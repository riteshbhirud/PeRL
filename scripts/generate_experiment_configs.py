#!/usr/bin/env python3
"""
Phase 2A: Generate YAML experiment configurations for PeRL mechanistic analysis.

This script generates 4 config files with a total of 103 experiments:
- core_1.5B.yaml: 24 experiments (8 methods × 3 seeds)
- core_7B.yaml: 15 experiments (5 methods × 3 seeds)
- stress_rank.yaml: 32 experiments (4 methods × 4 ranks × 2 seeds)
- stress_data.yaml: 32 experiments (4 methods × 4 data sizes × 2 seeds)

Usage:
    python scripts/generate_experiment_configs.py
    python scripts/generate_experiment_configs.py --output-dir configs/experiments
    python scripts/generate_experiment_configs.py --dry-run
"""

import os
import argparse
from typing import Dict, List, Any
from datetime import datetime

# Try to use ruamel.yaml for better YAML formatting, fallback to PyYAML
try:
    from ruamel.yaml import YAML
    USE_RUAMEL = True
except ImportError:
    import yaml
    USE_RUAMEL = False


# =============================================================================
# Configuration Constants (Based on Yin et al. paper specifications)
# =============================================================================

# Models
MODELS = {
    "1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

# Dataset
DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
FULL_DATASET_SIZE = 17000  # Approximate size of full dataset

# PEFT Methods with their default configurations
PEFT_METHODS = {
    "lora": {"type": "lora", "description": "Standard LoRA"},
    "dora": {"type": "dora", "description": "Weight-Decomposed LoRA"},
    "adalora": {"type": "adalora", "description": "Adaptive rank LoRA"},
    "pissa": {"type": "pissa", "description": "Principal Singular values LoRA"},
    "milora": {"type": "milora", "description": "Minor components LoRA"},
    "vera": {"type": "vera", "description": "Vector-based Random Matrix Adaptation"},
    "lora_plus": {"type": "lora_plus", "description": "LoRA with different A/B learning rates"},
    "miss": {"type": "miss", "description": "MiSS adapter"},
}

# Core experiment seeds
SEEDS = [42, 123, 456]

# Stress test configurations
STRESS_RANKS = [4, 16, 64, 256]
STRESS_DATA_SIZES = [1000, 4000, 8500, 17000]  # 1k, 4k, 8.5k, 17k

# Training settings (based on Yin et al.)
TRAINING_DEFAULTS = {
    "max_steps": 1000,
    "save_steps": 100,
    "logging_steps": 10,
    "gradient_accumulation_steps": 8,
    "num_generations": 8,
    "use_vllm": False,
    "use_liger_kernel": False,
}

# Model-specific batch sizes
BATCH_SIZES = {
    "1.5B": 4,
    "7B": 2,
}

# Learning rates by method (based on paper recommendations)
LEARNING_RATES = {
    "lora": 1e-5,
    "dora": 1e-5,
    "adalora": 1e-5,
    "pissa": 1e-5,
    "milora": 1e-5,
    "vera": 1e-4,  # VeRA typically uses higher LR
    "lora_plus": 1e-5,
    "miss": 1e-5,
}

# Default rank/alpha settings
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32


# =============================================================================
# Config Generation Functions
# =============================================================================

def create_base_config(
    model_size: str,
    description: str,
    tracking_frequency: int = 100,
) -> Dict[str, Any]:
    """Create the base configuration section."""
    return {
        "description": description,
        "model": {
            "model_name_or_path": MODELS[model_size],
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",  # Safe default, flash_attention_2 if available
        },
        "dataset": {
            "dataset_name_or_path": DATASET,
        },
        "training": {
            "max_steps": TRAINING_DEFAULTS["max_steps"],
            "save_steps": TRAINING_DEFAULTS["save_steps"],
            "logging_steps": TRAINING_DEFAULTS["logging_steps"],
            "per_device_train_batch_size": BATCH_SIZES[model_size],
            "gradient_accumulation_steps": TRAINING_DEFAULTS["gradient_accumulation_steps"],
            "num_generations": TRAINING_DEFAULTS["num_generations"],
            "use_vllm": TRAINING_DEFAULTS["use_vllm"],
            "use_liger_kernel": TRAINING_DEFAULTS["use_liger_kernel"],
            "report_to": ["wandb"],
        },
        "tracker": {
            "enable_spectral_tracking": True,
            "enable_gradient_tracking": True,
            "spectral_log_frequency": tracking_frequency,
            "gradient_log_frequency": tracking_frequency,
        },
        "wandb": {
            "use_wandb": True,
            "project": "peft-rlvr-mechanistic",
            "log_spectral_images": True,
            "log_gradient_images": True,
        },
    }


def create_experiment(
    name: str,
    peft_type: str,
    seed: int,
    rank: int = DEFAULT_RANK,
    alpha: int = None,
    learning_rate: float = None,
    example_numbers: int = None,
    extra_peft_args: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Create a single experiment configuration."""
    if alpha is None:
        alpha = rank * 2  # Default: alpha = 2 * rank

    if learning_rate is None:
        learning_rate = LEARNING_RATES.get(peft_type, 1e-5)

    experiment = {
        "name": name,
        "peft": {
            "type": peft_type,
            "r": rank,
            "lora_alpha": alpha,
        },
        "training": {
            "learning_rate": learning_rate,
        },
        "common": {
            "seed": seed,
        },
    }

    # Add example_numbers if specified (for data stress tests)
    if example_numbers is not None:
        experiment["dataset"] = {"example_numbers": example_numbers}

    # Add extra PEFT-specific arguments
    if extra_peft_args:
        experiment["peft"].update(extra_peft_args)

    return experiment


def generate_core_1_5b_config() -> Dict[str, Any]:
    """Generate core_1.5B.yaml: 8 methods × 3 seeds = 24 experiments."""
    methods = ["lora", "dora", "adalora", "pissa", "milora", "vera", "lora_plus", "miss"]

    config = {
        "base": create_base_config(
            model_size="1.5B",
            description="Core PEFT method comparison on 1.5B model (8 methods × 3 seeds = 24 experiments)",
        ),
        "experiments": [],
    }

    for method in methods:
        for seed in SEEDS:
            name = f"{method}_r{DEFAULT_RANK}_s{seed}"
            experiment = create_experiment(
                name=name,
                peft_type=method,
                seed=seed,
                rank=DEFAULT_RANK,
            )
            config["experiments"].append(experiment)

    return config


def generate_core_7b_config() -> Dict[str, Any]:
    """Generate core_7B.yaml: 5 methods × 3 seeds = 15 experiments."""
    # Subset of methods for 7B (more compute-intensive)
    methods = ["lora", "dora", "adalora", "pissa", "milora"]

    config = {
        "base": create_base_config(
            model_size="7B",
            description="Core PEFT method comparison on 7B model (5 methods × 3 seeds = 15 experiments)",
        ),
        "experiments": [],
    }

    for method in methods:
        for seed in SEEDS:
            name = f"{method}_r{DEFAULT_RANK}_s{seed}"
            experiment = create_experiment(
                name=name,
                peft_type=method,
                seed=seed,
                rank=DEFAULT_RANK,
            )
            config["experiments"].append(experiment)

    return config


def generate_stress_rank_config() -> Dict[str, Any]:
    """Generate stress_rank.yaml: 4 methods × 4 ranks × 2 seeds = 32 experiments."""
    methods = ["lora", "dora", "pissa", "milora"]
    seeds = [42, 123]  # 2 seeds for stress tests

    config = {
        "base": create_base_config(
            model_size="1.5B",
            description="Rank stress test (4 methods × 4 ranks × 2 seeds = 32 experiments)",
        ),
        "experiments": [],
    }

    for method in methods:
        for rank in STRESS_RANKS:
            for seed in seeds:
                name = f"{method}_r{rank}_s{seed}"
                experiment = create_experiment(
                    name=name,
                    peft_type=method,
                    seed=seed,
                    rank=rank,
                )
                config["experiments"].append(experiment)

    return config


def generate_stress_data_config() -> Dict[str, Any]:
    """Generate stress_data.yaml: 4 methods × 4 data sizes × 2 seeds = 32 experiments."""
    methods = ["lora", "dora", "pissa", "milora"]
    seeds = [42, 123]  # 2 seeds for stress tests

    config = {
        "base": create_base_config(
            model_size="1.5B",
            description="Data size stress test (4 methods × 4 sizes × 2 seeds = 32 experiments)",
        ),
        "experiments": [],
    }

    for method in methods:
        for data_size in STRESS_DATA_SIZES:
            for seed in seeds:
                # Format data size for name (e.g., 1000 -> 1k, 17000 -> 17k)
                size_str = f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                name = f"{method}_d{size_str}_s{seed}"
                experiment = create_experiment(
                    name=name,
                    peft_type=method,
                    seed=seed,
                    rank=DEFAULT_RANK,
                    example_numbers=data_size,
                )
                config["experiments"].append(experiment)

    return config


def write_yaml(config: Dict[str, Any], filepath: str) -> None:
    """Write config to YAML file with nice formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Add header comment
    header = f"""# PeRL Experiment Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# Usage:
#   python scripts/run_experiments.py --config {os.path.basename(filepath)}
#   python scripts/validate_configs.py --config {filepath}
#
"""

    with open(filepath, 'w') as f:
        f.write(header)

        if USE_RUAMEL:
            yaml_handler = YAML()
            yaml_handler.default_flow_style = False
            yaml_handler.indent(mapping=2, sequence=4, offset=2)
            yaml_handler.dump(config, f)
        else:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)


def generate_all_configs(output_dir: str, dry_run: bool = False) -> Dict[str, int]:
    """Generate all experiment configuration files."""
    configs = {
        "core_1.5B.yaml": generate_core_1_5b_config(),
        "core_7B.yaml": generate_core_7b_config(),
        "stress_rank.yaml": generate_stress_rank_config(),
        "stress_data.yaml": generate_stress_data_config(),
    }

    stats = {}

    for filename, config in configs.items():
        filepath = os.path.join(output_dir, filename)
        num_experiments = len(config["experiments"])
        stats[filename] = num_experiments

        if dry_run:
            print(f"[DRY-RUN] Would write {filepath}")
            print(f"          {num_experiments} experiments")
            print(f"          Base model: {config['base']['model']['model_name_or_path']}")
        else:
            write_yaml(config, filepath)
            print(f"✓ Generated {filepath}")
            print(f"  {num_experiments} experiments")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML experiment configurations for PeRL mechanistic analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/experiments",
        help="Output directory for config files (default: configs/experiments)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )

    args = parser.parse_args()

    # Make output_dir relative to script location if not absolute
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(project_root, args.output_dir)

    print("=" * 60)
    print("PeRL Experiment Config Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print()

    stats = generate_all_configs(args.output_dir, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    total = sum(stats.values())
    for filename, count in stats.items():
        print(f"  {filename}: {count} experiments")
    print(f"  {'─' * 40}")
    print(f"  Total: {total} experiments")

    if not args.dry_run:
        print()
        print("Next steps:")
        print("  1. Validate configs: python scripts/validate_configs.py")
        print("  2. Run experiments:  python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml")


if __name__ == "__main__":
    main()
