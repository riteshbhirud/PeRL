
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

def get_model_size(model_path: str) -> str:
    """Extract model size from model path for tagging."""
    model_path_lower = model_path.lower()
    if "7b" in model_path_lower:
        return "7B"
    elif "1.5b" in model_path_lower or "1_5b" in model_path_lower:
        return "1.5B"
    elif "3b" in model_path_lower:
        return "3B"
    elif "14b" in model_path_lower:
        return "14B"
    return "unknown"


def build_wandb_tags(args: TrainConfig) -> list:
    """Build automatic tags for WandB based on config."""
    tags = list(args.wandb.tags) if args.wandb.tags else []

    # Add PEFT method tag
    if args.peft.type:
        tags.append(args.peft.type)

    # Add seed tag
    tags.append(f"seed{args.common.seed}")

    # Add model size tag
    if args.model.model_name_or_path:
        model_size = get_model_size(args.model.model_name_or_path)
        tags.append(model_size)

    # Add rank tag
    if args.peft.r:
        tags.append(f"r{args.peft.r}")

    # Add special condition tags
    if args.peft.r and args.peft.r < 8:
        tags.append("ultra_low_rank")
    if args.peft.r and args.peft.r >= 128:
        tags.append("high_rank")

    if args.dataset.example_numbers and args.dataset.example_numbers < 5000:
        tags.append("data_scarcity")

    if args.tracker.enable_spectral_tracking:
        tags.append("spectral_tracking")
    if args.tracker.enable_gradient_tracking:
        tags.append("gradient_tracking")

    # Add stress test tag if detected from output dir
    if args.training.output_dir and "stress" in args.training.output_dir.lower():
        tags.append("stress_test")

    return list(set(tags))  # Remove duplicates


def init_wandb(args: TrainConfig) -> bool:
    """Initialize WandB with comprehensive config and tags."""
    if not args.wandb.use_wandb:
        return False

    if "wandb" not in args.training.report_to:
        return False

    try:
        import wandb
    except ImportError:
        logger.warning("WandB not installed. Install with: pip install wandb")
        return False

    # Set offline mode if requested
    if args.wandb.offline:
        os.environ["WANDB_MODE"] = "offline"
        logger.info("WandB running in offline mode")

    # Build run name
    model_size = get_model_size(args.model.model_name_or_path) if args.model.model_name_or_path else "unknown"
    run_name = args.training.run_name or f"{args.peft.type}_{model_size}_r{args.peft.r}_seed{args.common.seed}"

    # Build group name for related runs
    group = args.wandb.group
    if not group and args.peft.type:
        group = f"{model_size}_{args.peft.type}"

    # Build tags
    tags = build_wandb_tags(args)

    # Build comprehensive config
    wandb_config = {
        # Model settings
        "model/name": args.model.model_name_or_path,
        "model/dtype": args.model.dtype,
        "model/attn_implementation": args.model.attn_implementation,
        "model/size": model_size,

        # PEFT settings
        "peft/type": args.peft.type,
        "peft/rank": args.peft.r,
        "peft/alpha": args.peft.lora_alpha,
        "peft/dropout": args.peft.lora_dropout,
        "peft/target_modules": args.peft.target_modules,

        # Training settings
        "training/learning_rate": args.training.learning_rate,
        "training/max_steps": args.training.max_steps,
        "training/batch_size": args.training.per_device_train_batch_size,
        "training/gradient_accumulation": args.training.gradient_accumulation_steps,
        "training/num_generations": args.training.num_generations,
        "training/loss_type": args.training.loss_type,
        "training/lr_scheduler": args.training.lr_scheduler_type,

        # Dataset settings
        "dataset/name": args.dataset.dataset_name_or_path,
        "dataset/example_numbers": args.dataset.example_numbers,

        # Tracker settings
        "tracker/spectral_enabled": args.tracker.enable_spectral_tracking,
        "tracker/gradient_enabled": args.tracker.enable_gradient_tracking,
        "tracker/spectral_frequency": args.tracker.spectral_log_frequency,
        "tracker/gradient_frequency": args.tracker.gradient_log_frequency,

        # Common settings
        "seed": args.common.seed,
    }

    # Initialize WandB
    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        name=run_name,
        group=group,
        tags=tags,
        notes=args.wandb.notes,
        config=wandb_config,
        reinit=True,  # Allow multiple inits in same process
    )

    logger.info(f"WandB initialized: project={args.wandb.project}, run={run_name}")
    logger.info(f"WandB tags: {tags}")
    logger.info(f"WandB URL: {wandb.run.get_url()}")

    return True


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
            # Use enhanced WandB initialization
            init_wandb(args)

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
