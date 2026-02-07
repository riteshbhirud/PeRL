"""
Checkpoint Loading for PeRL Evaluation.

Handles loading PEFT checkpoints with metadata extraction.
Supports all PEFT types: LoRA, DoRA, AdaLoRA, PiSSA, MiLoRA, VeRA, etc.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("perl.evaluation.checkpoint")


@dataclass
class CheckpointInfo:
    """Information about a loaded checkpoint."""
    path: str
    base_model: str
    peft_type: str
    rank: int
    alpha: int
    seed: Optional[int]
    step: Optional[int]
    target_modules: list
    extra_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Generate a descriptive name for this checkpoint."""
        parts = [self.peft_type]
        if self.rank:
            parts.append(f"r{self.rank}")
        if self.seed is not None:
            parts.append(f"s{self.seed}")
        if self.step is not None:
            parts.append(f"step{self.step}")
        return "_".join(parts)


def get_checkpoint_metadata(checkpoint_path: str) -> CheckpointInfo:
    """
    Extract metadata from a checkpoint without loading the model.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        CheckpointInfo with extracted metadata
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Look for adapter_config.json
    adapter_config_path = path / "adapter_config.json"
    if not adapter_config_path.exists():
        # Try parent directory (in case we're in a step directory)
        adapter_config_path = path.parent / "adapter_config.json"

    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            config = json.load(f)
    else:
        # No adapter config - might be a full model
        config = {}

    # Extract PEFT type
    peft_type = config.get("peft_type", "unknown")
    if peft_type == "LORA":
        # Check for DoRA
        if config.get("use_dora", False):
            peft_type = "dora"
        else:
            peft_type = "lora"
    elif peft_type:
        peft_type = peft_type.lower()

    # Extract rank and alpha
    rank = config.get("r", config.get("rank", 0))
    alpha = config.get("lora_alpha", config.get("alpha", rank * 2 if rank else 0))

    # Extract base model
    base_model = config.get("base_model_name_or_path", "")

    # Extract target modules
    target_modules = config.get("target_modules", [])

    # Try to extract seed and step from path
    seed = None
    step = None

    # Look for seed pattern (s42, seed42, _42)
    seed_match = re.search(r'[_/]s(?:eed)?(\d+)', str(path))
    if seed_match:
        seed = int(seed_match.group(1))

    # Look for step pattern (checkpoint-1000, step-1000)
    step_match = re.search(r'(?:checkpoint|step)[_-](\d+)', str(path))
    if step_match:
        step = int(step_match.group(1))

    return CheckpointInfo(
        path=str(path),
        base_model=base_model,
        peft_type=peft_type,
        rank=rank,
        alpha=alpha,
        seed=seed,
        step=step,
        target_modules=target_modules,
        extra_config=config,
    )


def load_checkpoint(
    checkpoint_path: str,
    base_model_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[Any, Any, CheckpointInfo]:
    """
    Load a PEFT checkpoint with its tokenizer.

    Args:
        checkpoint_path: Path to the PEFT checkpoint
        base_model_path: Override base model path (uses adapter_config if None)
        device_map: Device mapping strategy
        torch_dtype: Model dtype (bfloat16, float16, float32)
        trust_remote_code: Trust remote code in model
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization

    Returns:
        Tuple of (model, tokenizer, checkpoint_info)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Get checkpoint metadata first
    info = get_checkpoint_metadata(checkpoint_path)

    # Determine base model path
    if base_model_path is None:
        base_model_path = info.base_model

    if not base_model_path:
        raise ValueError(
            f"No base model path found in checkpoint and none provided. "
            f"Please specify --base_model"
        )

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    logger.info(f"  PEFT type: {info.peft_type}")
    logger.info(f"  Rank: {info.rank}")
    logger.info(f"  Base model: {base_model_path}")

    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    logger.info("Loading base model...")
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs,
    )

    # Check if this is a PEFT checkpoint
    adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
    if adapter_config_path.exists():
        logger.info("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            torch_dtype=dtype,
        )
        model.eval()
    else:
        # This might be a full model checkpoint
        logger.info("No adapter found, using as full model checkpoint...")
        model = base_model
        model.eval()

    logger.info("Checkpoint loaded successfully")

    return model, tokenizer, info


def merge_checkpoint(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
) -> str:
    """
    Merge a PEFT adapter into its base model and save.

    Args:
        checkpoint_path: Path to the PEFT checkpoint
        output_path: Where to save merged model (auto-generated if None)
        base_model_path: Override base model path

    Returns:
        Path to the merged model
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    info = get_checkpoint_metadata(checkpoint_path)

    if base_model_path is None:
        base_model_path = info.base_model

    if not base_model_path:
        raise ValueError("No base model path found")

    logger.info(f"Merging checkpoint: {checkpoint_path}")

    # Load on CPU for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    merged_model = model.merge_and_unload()

    # Determine output path
    if output_path is None:
        checkpoint_name = Path(checkpoint_path).name
        output_path = str(Path(checkpoint_path).parent / f"{checkpoint_name}_merged")

    logger.info(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    return output_path


def list_checkpoints(experiment_dir: str, pattern: str = "checkpoint-*") -> list:
    """
    List all checkpoints in an experiment directory.

    Args:
        experiment_dir: Path to experiment output directory
        pattern: Glob pattern for checkpoint directories

    Returns:
        List of checkpoint paths sorted by step number
    """
    from glob import glob

    path = Path(experiment_dir)
    checkpoints = []

    for cp_path in path.glob(pattern):
        if cp_path.is_dir():
            try:
                info = get_checkpoint_metadata(str(cp_path))
                checkpoints.append((info.step or 0, str(cp_path)))
            except Exception:
                continue

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])

    return [cp[1] for cp in checkpoints]
