"""
Utility functions for PeRL evaluation.

Provides stage management, logging setup, and model merging utilities.
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    name: str = "perl.eval"
) -> logging.Logger:
    """
    Configure logging for evaluation.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Stage Context Manager
# =============================================================================

class StageContext:
    """
    Context manager for evaluation stages with timing and logging.

    Usage:
        with StageContext("Model Loading", logger) as stage:
            # do work
            stage.log("Loaded model successfully")
    """

    def __init__(
        self,
        stage_name: str,
        logger: Optional[logging.Logger] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize stage context.

        Args:
            stage_name: Name of the evaluation stage
            logger: Logger instance (uses default if None)
            log_level: Logging level for stage messages
        """
        self.stage_name = stage_name
        self.logger = logger or logging.getLogger("perl.eval")
        self.log_level = log_level
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Enter the stage context."""
        self.start_time = time.time()
        self.logger.log(self.log_level, f"[Stage] Starting: {self.stage_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the stage context."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is not None:
            self.logger.error(
                f"[Stage] Failed: {self.stage_name} ({duration:.2f}s) - {exc_val}"
            )
            return False  # Re-raise exception

        self.logger.log(
            self.log_level,
            f"[Stage] Completed: {self.stage_name} ({duration:.2f}s)"
        )
        return False

    def log(self, message: str, level: int = None):
        """Log a message within the stage context."""
        level = level or self.log_level
        self.logger.log(level, f"  [{self.stage_name}] {message}")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


# =============================================================================
# Model Merging
# =============================================================================

def merge_model_if_needed(
    model_path: str,
    adapter_path: str,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Merge LoRA/PEFT adapter into base model if adapter path is provided.

    Args:
        model_path: Path to base model
        adapter_path: Path to LoRA/PEFT adapter (empty string = no merge)
        output_path: Optional path to save merged model
        logger: Logger instance

    Returns:
        Path to the model to use (merged path or original model path)
    """
    logger = logger or logging.getLogger("perl.eval")

    if not adapter_path or adapter_path.strip() == "":
        logger.info(f"No adapter specified, using base model: {model_path}")
        return model_path

    logger.info(f"Merging adapter into base model...")
    logger.info(f"  Base model: {model_path}")
    logger.info(f"  Adapter: {adapter_path}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load on CPU for merging
            trust_remote_code=True,
        )

        # Load and merge adapter
        logger.info("Loading and merging adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()

        # Determine output path
        if output_path is None:
            adapter_name = Path(adapter_path).name
            output_path = str(Path(adapter_path).parent / f"{adapter_name}_merged")

        # Save merged model
        logger.info(f"Saving merged model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)

        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Merge complete: {output_path}")
        return output_path

    except ImportError as e:
        logger.error(f"Failed to import required libraries: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to merge model: {e}")
        raise


def check_model_exists(model_path: str) -> bool:
    """Check if a model exists at the given path."""
    path = Path(model_path)

    # Check for local directory
    if path.is_dir():
        # Look for model files
        model_files = [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "adapter_config.json",
            "adapter_model.safetensors",
        ]
        return any((path / f).exists() for f in model_files)

    # Assume HuggingFace Hub model
    return True  # Will be validated on load


def get_device_info() -> dict:
    """Get information about available devices."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_devices": [],
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "index": i,
                "name": props.name,
                "memory_gb": props.total_memory / (1024 ** 3),
            })

    return info
