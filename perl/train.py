
import torch
import os

from typing import List, Optional
from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from fire import Fire

from perl.utils.logging import init_logger, logger
from perl.data import load_dataset
from perl.config.config import TrainConfig
from perl.lora.adapter import apply_peft

def fuzzy_jobs(
    args: TrainConfig
):
    init_logger()
    args.training.output_dir = args.training.output_dir or "output"
    args.training.run_name = args.training.run_name or args.training.output_dir # training run name is the output_dir
    if not os.path.exists(args.training.output_dir): # check if output_dir exists
        os.makedirs(args.training.output_dir, exist_ok=True)
    else:
        logger.info(f"Output directory {args.training.output_dir} already exists, using it")
    set_seed(args.common.seed)

    if args.common.debug:
        args.training.report_to = []

    # only initialize for rank 0 when process group is available
    is_main_process = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0

    if is_main_process:
        if "trackio" in args.training.report_to:
            import trackio
            trackio.init(
                project=args.logging.trackio_project,
                space_id=args.logging.trackio_space_id,
                config=vars(args.training)
            )
            logger.info(f"Trackio initialized successfully")
        elif "wandb" in args.training.report_to:
            import wandb
            wandb.init(
                name=args.training.run_name,
                config=vars(args.training),
            )
            logger.info(f"Wandb initialized successfully")

    return args

def train(
    config: TrainConfig = None
):
    # 0. parse args and prepare logger
    print(config)
    args = fuzzy_jobs(config)

    # 1. load tokenizer and dataset
    logger.info(f"Loading tokenizer from {args.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    tokenizer.padding_side = "left"  # Configure for decoder-only architecture: use left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    logger.info(f"Loading dataset from {args.dataset.dataset_name_or_path}")
    dataset = load_dataset(
        args.dataset.dataset_name_or_path,
        example_numbers=args.dataset.example_numbers,
        tokenizer=tokenizer
    )
    train_dataset = dataset["train_dataset"]
    test_dataset = dataset["test_dataset"]
    reward_functions = dataset["reward_functions"]

    if "reward_weights" in dataset:
        reward_weights = dataset["reward_weights"]
    else:
        reward_weights = [1.0] * len(reward_functions)
    args.training.reward_weights = reward_weights

    # 2. load and configure model
    logger.info(f"Loading model from {args.model.model_name_or_path}")

    # Determine attention implementation (with fallback to SDPA if flash_attention_2 fails)
    attn_impl = args.model.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model.model_name_or_path,
            torch_dtype= torch.bfloat16 if args.model.dtype == "bfloat16" else torch.float16,
            attn_implementation=attn_impl
        )
        logger.info(f"Model loaded successfully with {attn_impl} attention")
    except ImportError as e:
        if attn_impl == "flash_attention_2":
            logger.warning(f"flash_attention_2 failed ({e}), falling back to sdpa")
            attn_impl = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                args.model.model_name_or_path,
                torch_dtype= torch.bfloat16 if args.model.dtype == "bfloat16" else torch.float16,
                attn_implementation=attn_impl
            )
            logger.info(f"Model loaded successfully with {attn_impl} attention (fallback)")
        else:
            raise

    # 3. configure PEFT adapter
    if args.peft.use_peft:
        logger.info(f"Detected PEFT configuration, applying {args.peft.type} adapter")
        optimizer, model = apply_peft(model, args)
        logger.info(f"PEFT adapter ({args.peft.type}) configured successfully")

    # 4. Setup mechanistic tracking (if enabled)
    callbacks = []
    spectral_tracker = None
    gradient_tracker = None

    # Determine tracking output directory
    tracking_base_dir = args.tracker.tracking_output_dir or args.training.output_dir

    # Print tracking summary if any tracking is enabled
    if args.tracker.enable_spectral_tracking or args.tracker.enable_gradient_tracking:
        from perl.utils.utils import print_tracking_info
        print_tracking_info(args)

    if args.tracker.enable_spectral_tracking and args.peft.use_peft:
        from perl.trackers import SpectralTracker, SpectralTrackingCallback
        spectral_save_dir = os.path.join(tracking_base_dir, 'spectral_logs')
        spectral_tracker = SpectralTracker(
            model=model,
            log_frequency=args.tracker.spectral_log_frequency,
            save_dir=spectral_save_dir,
            peft_type=args.peft.type,
            compute_full_svd=args.tracker.compute_full_svd,
            max_layers_to_track=args.tracker.max_layers_to_track,
        )
        callbacks.append(SpectralTrackingCallback(
            tracker=spectral_tracker,
            save_on_train_end=True,
            log_to_wandb=("wandb" in args.training.report_to or "trackio" in args.training.report_to),
        ))
        logger.info(f"Spectral tracking: {len(spectral_tracker.adapter_layers)} layers, "
                    f"every {args.tracker.spectral_log_frequency} steps -> {spectral_save_dir}")

    if args.tracker.enable_gradient_tracking and args.peft.use_peft:
        from perl.trackers import GradientFlowTracker, GradientFlowCallback
        gradient_save_dir = os.path.join(tracking_base_dir, 'gradient_logs')
        gradient_tracker = GradientFlowTracker(
            model=model,
            log_frequency=args.tracker.gradient_log_frequency,
            save_dir=gradient_save_dir,
            track_adapter_only=args.tracker.track_adapter_only,
            peft_type=args.peft.type,
        )
        callbacks.append(GradientFlowCallback(
            tracker=gradient_tracker,
            save_on_train_end=True,
            log_to_wandb=("wandb" in args.training.report_to or "trackio" in args.training.report_to),
            create_heatmaps=True,
        ))
        logger.info(f"Gradient tracking: {len(gradient_tracker.adapter_params)} params, "
                    f"every {args.tracker.gradient_log_frequency} steps -> {gradient_save_dir}")

    # 5. Training configuration
    training_args = GRPOConfig(
        **vars(args.training),
    )

    # 6. Train
    logger.info(f"Training model with GRPO")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
        callbacks=callbacks if callbacks else None,
    )
    
    # 7. Support resuming from checkpoint
    resume_checkpoint = args.training.resume_from_checkpoint
    if resume_checkpoint == "true":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training completed successfully")

    # 8. Save model and spectral data
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save tracking data (also saved via callback, but ensure final save)
    tracking_base_dir = args.tracker.tracking_output_dir or args.training.output_dir

    if spectral_tracker is not None:
        final_spectral_path = os.path.join(tracking_base_dir, 'spectral_history_final.pt')
        spectral_tracker.save(final_spectral_path)
        logger.info(f"Final spectral history saved to {final_spectral_path}")

        # Print summary statistics
        summary = spectral_tracker.get_summary_stats()
        logger.info(f"Spectral tracking summary: {summary.get('total_steps_logged', 0)} steps logged, "
                    f"{summary.get('num_layers', 0)} layers tracked")

    if gradient_tracker is not None:
        final_gradient_path = os.path.join(tracking_base_dir, 'gradient_flow_final.pt')
        gradient_tracker.save(final_gradient_path)
        logger.info(f"Final gradient flow saved to {final_gradient_path}")

        # Print summary statistics
        summary = gradient_tracker.get_summary_stats()
        logger.info(f"Gradient tracking summary: {summary.get('total_steps_logged', 0)} steps logged, "
                    f"{summary.get('num_layers', 0)} layers tracked")

    if spectral_tracker is not None or gradient_tracker is not None:
        logger.info(f"All tracking data saved to {tracking_base_dir}")
