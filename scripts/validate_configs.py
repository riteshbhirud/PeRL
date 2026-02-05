#!/usr/bin/env python3
"""
Validate PeRL experiment configuration files.

This script validates YAML config files for:
- Required fields presence
- Valid PEFT method types
- Valid parameter ranges
- Unique experiment names
- Config consistency

Usage:
    python scripts/validate_configs.py
    python scripts/validate_configs.py --config configs/experiments/core_1.5B.yaml
    python scripts/validate_configs.py --all --verbose
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)


# =============================================================================
# Validation Constants
# =============================================================================

VALID_PEFT_TYPES = {
    "lora", "dora", "adalora", "pissa", "milora", "milora_plus",
    "vera", "lora_plus", "miss", "lorafa", "IA3", "layernorm",
    "hra", "rslora", "slicefine",
}

VALID_ATTN_IMPLEMENTATIONS = {"flash_attention_2", "sdpa", "eager"}

VALID_DTYPES = {"bfloat16", "float16", "float32"}

# Parameter ranges
RANK_RANGE = (1, 512)
ALPHA_RANGE = (1, 1024)
LR_RANGE = (1e-8, 1.0)  # Wide range, PEFT typically uses 1e-6 to 1e-4
MAX_STEPS_RANGE = (1, 100000)
BATCH_SIZE_RANGE = (1, 128)

# Required fields in base config
REQUIRED_BASE_FIELDS = {
    "model": ["model_name_or_path"],
    "dataset": ["dataset_name_or_path"],
    "training": ["max_steps"],
}

# Required fields in experiment config
REQUIRED_EXPERIMENT_FIELDS = {
    "name": None,
    "peft": ["type", "r"],
}


# =============================================================================
# Validation Classes
# =============================================================================

@dataclass
class ValidationError:
    """Represents a validation error."""
    level: str  # "error" or "warning"
    config_file: str
    experiment_name: Optional[str]
    field: str
    message: str

    def __str__(self):
        exp_part = f" [{self.experiment_name}]" if self.experiment_name else ""
        return f"[{self.level.upper()}] {self.config_file}{exp_part}: {self.field} - {self.message}"


class ConfigValidator:
    """Validates PeRL experiment configuration files."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def add_error(self, config_file: str, field: str, message: str, experiment_name: str = None):
        """Add a validation error."""
        error = ValidationError("error", config_file, experiment_name, field, message)
        self.errors.append(error)

    def add_warning(self, config_file: str, field: str, message: str, experiment_name: str = None):
        """Add a validation warning."""
        warning = ValidationError("warning", config_file, experiment_name, field, message)
        self.warnings.append(warning)

    def validate_file(self, filepath: str) -> bool:
        """Validate a single config file."""
        filename = os.path.basename(filepath)
        self.log(f"\nValidating: {filepath}")

        # Check file exists
        if not os.path.exists(filepath):
            self.add_error(filename, "file", f"File not found: {filepath}")
            return False

        # Load YAML
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.add_error(filename, "yaml", f"Invalid YAML syntax: {e}")
            return False

        if config is None:
            self.add_error(filename, "content", "Config file is empty")
            return False

        # Validate structure
        self._validate_structure(config, filename)

        # Validate base config
        if "base" in config:
            self._validate_base(config["base"], filename)

        # Validate experiments
        if "experiments" in config:
            self._validate_experiments(config["experiments"], filename)

        return len([e for e in self.errors if e.config_file == filename]) == 0

    def _validate_structure(self, config: Dict, filename: str):
        """Validate top-level structure."""
        if "base" not in config:
            self.add_error(filename, "structure", "Missing 'base' section")

        if "experiments" not in config:
            self.add_error(filename, "structure", "Missing 'experiments' section")
        elif not isinstance(config["experiments"], list):
            self.add_error(filename, "experiments", "Experiments must be a list")
        elif len(config["experiments"]) == 0:
            self.add_warning(filename, "experiments", "No experiments defined")

    def _validate_base(self, base: Dict, filename: str):
        """Validate base configuration."""
        # Check required fields
        for section, fields in REQUIRED_BASE_FIELDS.items():
            if section not in base:
                self.add_error(filename, f"base.{section}", f"Missing required section '{section}'")
            elif fields:
                for field in fields:
                    if field not in base[section]:
                        self.add_error(filename, f"base.{section}.{field}",
                                       f"Missing required field '{field}'")

        # Validate model settings
        if "model" in base:
            model = base["model"]
            if "dtype" in model and model["dtype"] not in VALID_DTYPES:
                self.add_error(filename, "base.model.dtype",
                               f"Invalid dtype '{model['dtype']}', must be one of {VALID_DTYPES}")

            if "attn_implementation" in model:
                if model["attn_implementation"] not in VALID_ATTN_IMPLEMENTATIONS:
                    self.add_error(filename, "base.model.attn_implementation",
                                   f"Invalid attn_implementation '{model['attn_implementation']}', "
                                   f"must be one of {VALID_ATTN_IMPLEMENTATIONS}")

        # Validate training settings
        if "training" in base:
            training = base["training"]
            self._validate_training(training, filename, "base")

        # Validate tracker settings
        if "tracker" in base:
            tracker = base["tracker"]
            for freq_field in ["spectral_log_frequency", "gradient_log_frequency"]:
                if freq_field in tracker:
                    freq = tracker[freq_field]
                    if not isinstance(freq, int) or freq < 1:
                        self.add_error(filename, f"base.tracker.{freq_field}",
                                       f"Frequency must be a positive integer, got {freq}")

    def _validate_training(self, training: Dict, filename: str, prefix: str):
        """Validate training configuration."""
        # Validate max_steps
        if "max_steps" in training:
            max_steps = training["max_steps"]
            if not isinstance(max_steps, int) or not (RANK_RANGE[0] <= max_steps <= MAX_STEPS_RANGE[1]):
                self.add_error(filename, f"{prefix}.training.max_steps",
                               f"max_steps must be integer in range {MAX_STEPS_RANGE}")

        # Validate batch size
        if "per_device_train_batch_size" in training:
            batch_size = training["per_device_train_batch_size"]
            if not isinstance(batch_size, int) or not (BATCH_SIZE_RANGE[0] <= batch_size <= BATCH_SIZE_RANGE[1]):
                self.add_error(filename, f"{prefix}.training.per_device_train_batch_size",
                               f"batch_size must be integer in range {BATCH_SIZE_RANGE}")

        # Validate learning_rate
        if "learning_rate" in training:
            lr = training["learning_rate"]
            # Convert string scientific notation to float if needed
            if isinstance(lr, str):
                try:
                    lr = float(lr)
                except ValueError:
                    self.add_error(filename, f"{prefix}.training.learning_rate",
                                   f"Invalid learning_rate value: {lr}")
                    return
            if not isinstance(lr, (int, float)) or not (LR_RANGE[0] <= lr <= LR_RANGE[1]):
                self.add_warning(filename, f"{prefix}.training.learning_rate",
                                 f"learning_rate {lr} outside typical range {LR_RANGE}")

    def _validate_experiments(self, experiments: List[Dict], filename: str):
        """Validate all experiments."""
        seen_names = set()

        for i, exp in enumerate(experiments):
            exp_name = exp.get("name", f"experiment_{i}")

            # Check for required name field
            if "name" not in exp:
                self.add_error(filename, f"experiments[{i}].name",
                               "Missing required field 'name'", exp_name)

            # Check for duplicate names
            if exp_name in seen_names:
                self.add_error(filename, f"experiments[{i}].name",
                               f"Duplicate experiment name: {exp_name}", exp_name)
            seen_names.add(exp_name)

            # Validate peft config
            if "peft" not in exp:
                self.add_error(filename, f"experiments[{i}].peft",
                               "Missing required 'peft' section", exp_name)
            else:
                self._validate_peft(exp["peft"], filename, exp_name)

            # Validate training overrides
            if "training" in exp:
                self._validate_training(exp["training"], filename, f"experiments.{exp_name}")

            # Validate common (seed)
            if "common" in exp:
                common = exp["common"]
                if "seed" in common:
                    seed = common["seed"]
                    if not isinstance(seed, int):
                        self.add_error(filename, f"experiments.{exp_name}.common.seed",
                                       f"Seed must be an integer, got {type(seed).__name__}", exp_name)

    def _validate_peft(self, peft: Dict, filename: str, exp_name: str):
        """Validate PEFT configuration."""
        # Check type
        if "type" not in peft:
            self.add_error(filename, f"experiments.{exp_name}.peft.type",
                           "Missing required field 'type'", exp_name)
        else:
            peft_type = peft["type"]
            if peft_type not in VALID_PEFT_TYPES:
                self.add_error(filename, f"experiments.{exp_name}.peft.type",
                               f"Invalid PEFT type '{peft_type}', must be one of {VALID_PEFT_TYPES}",
                               exp_name)

        # Check rank
        if "r" not in peft:
            self.add_warning(filename, f"experiments.{exp_name}.peft.r",
                             "Missing rank 'r', will use default", exp_name)
        else:
            r = peft["r"]
            if not isinstance(r, int) or not (RANK_RANGE[0] <= r <= RANK_RANGE[1]):
                self.add_error(filename, f"experiments.{exp_name}.peft.r",
                               f"Rank must be integer in range {RANK_RANGE}", exp_name)

        # Check lora_alpha
        if "lora_alpha" in peft:
            alpha = peft["lora_alpha"]
            if not isinstance(alpha, int) or not (ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]):
                self.add_error(filename, f"experiments.{exp_name}.peft.lora_alpha",
                               f"lora_alpha must be integer in range {ALPHA_RANGE}", exp_name)

    def validate_directory(self, directory: str) -> Tuple[int, int]:
        """Validate all YAML files in a directory."""
        yaml_files = [f for f in os.listdir(directory) if f.endswith('.yaml') or f.endswith('.yml')]

        if not yaml_files:
            print(f"No YAML config files found in {directory}")
            return 0, 0

        valid_count = 0
        for filename in sorted(yaml_files):
            filepath = os.path.join(directory, filename)
            if self.validate_file(filepath):
                valid_count += 1

        return valid_count, len(yaml_files)

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")

        print()


def count_experiments(filepath: str) -> int:
    """Count experiments in a config file."""
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return len(config.get("experiments", []))
    except:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate PeRL experiment configuration files"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a specific config file to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all configs in configs/experiments/",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="configs/experiments",
        help="Directory containing config files (default: configs/experiments)",
    )

    args = parser.parse_args()

    # Default to --all if no specific config is provided
    if not args.config and not args.all:
        args.all = True

    # Make directory path relative to script location if not absolute
    if not os.path.isabs(args.directory):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.directory = os.path.join(project_root, args.directory)

    validator = ConfigValidator(verbose=args.verbose)

    print("=" * 60)
    print("PeRL Config Validator")
    print("=" * 60)

    if args.config:
        # Validate single file
        filepath = args.config
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.getcwd(), filepath)

        is_valid = validator.validate_file(filepath)
        num_experiments = count_experiments(filepath)
        print(f"\nExperiments in file: {num_experiments}")

    elif args.all:
        # Validate all files in directory
        print(f"Scanning: {args.directory}")

        if not os.path.exists(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            sys.exit(1)

        valid_count, total_count = validator.validate_directory(args.directory)

        # Count total experiments
        total_experiments = 0
        for f in os.listdir(args.directory):
            if f.endswith('.yaml') or f.endswith('.yml'):
                total_experiments += count_experiments(os.path.join(args.directory, f))

        print(f"\nFiles validated: {valid_count}/{total_count}")
        print(f"Total experiments: {total_experiments}")

    validator.print_summary()

    # Exit with error code if there were errors
    sys.exit(1 if validator.errors else 0)


if __name__ == "__main__":
    main()
