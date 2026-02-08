#!/usr/bin/env python3
"""
Generate Stress Test Configuration Files for PeRL.

Creates YAML config files for systematic stress testing:
1. Ultra-low rank stress tests (r=1,2,4,8)
2. Data scarcity stress tests (n=1K,2K,4K,8K)

These tests help characterize failure modes for the paper's
"Failure Mode Taxonomy" section.

Usage:
    python scripts/generate_stress_configs.py
    python scripts/generate_stress_configs.py --validate
    python scripts/generate_stress_configs.py --dry_run
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Try to use ruamel.yaml for better formatting, fall back to PyYAML
try:
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.preserve_quotes = True
    USE_RUAMEL = True
except ImportError:
    import yaml as pyyaml
    yaml = pyyaml
    USE_RUAMEL = False


def write_yaml(data: Dict, path: Path):
    """Write YAML file with proper formatting."""
    with open(path, 'w') as f:
        if USE_RUAMEL:
            yaml.dump(data, f)
        else:
            pyyaml.dump(data, f, default_flow_style=False, sort_keys=False)


def generate_stress_rank_config() -> Dict[str, Any]:
    """
    Generate ultra-low rank stress test configuration.

    Tests 4 methods × 4 ranks × 2 seeds = 32 experiments

    Purpose: Identify at what rank each method catastrophically fails.
    Hypothesis: PiSSA/MiLoRA fail at low rank, DoRA/LoRA degrade gracefully.
    """
    methods = ['lora', 'dora', 'pissa', 'adalora']
    ranks = [1, 2, 4, 8]
    seeds = [42, 43]

    base = {
        'model': {
            'model_name_or_path': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
            'attn_implementation': 'flash_attention_2',
            'dtype': 'bfloat16',
        },
        'dataset': {
            'dataset_name_or_path': 'open-r1/DAPO-Math-17k-Processed',
        },
        'training': {
            'max_steps': 512,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 8,
            'learning_rate': 1.0e-5,
            'warmup_ratio': 0.1,
            'save_steps': 128,
            'logging_steps': 10,
            'eval_steps': 128,
            'use_liger_kernel': False,
            'gradient_checkpointing': True,
            'report_to': ['wandb'],
        },
        'tracker': {
            'enable_spectral_tracking': True,
            'enable_gradient_tracking': True,
            'spectral_log_frequency': 100,
            'gradient_log_frequency': 100,
            'save_trajectory': True,
        },
        'wandb': {
            'use_wandb': True,
            'project': 'peft-rlvr-mechanistic',
            'tags': ['stress-test', 'low-rank'],
        },
        'common': {
            'seed': 42,
            'output_dir': 'output/stress_rank',
        },
    }

    experiments = []
    for method in methods:
        for rank in ranks:
            for seed in seeds:
                exp = {
                    'name': f'{method}_r{rank}_seed{seed}',
                    'peft': {
                        'type': method,
                        'r': rank,
                        'lora_alpha': rank * 2,  # Keep alpha = 2*r ratio
                        'lora_dropout': 0.0,
                        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj',
                                          'up_proj', 'down_proj', 'gate_proj'],
                    },
                    'common': {
                        'seed': seed,
                    },
                    'wandb': {
                        'run_name': f'{method}_r{rank}_seed{seed}',
                        'tags': ['stress-test', 'low-rank', f'r{rank}', method],
                    },
                }
                experiments.append(exp)

    return {'base': base, 'experiments': experiments}


def generate_stress_data_config() -> Dict[str, Any]:
    """
    Generate data scarcity stress test configuration.

    Tests 4 methods × 4 data sizes × 2 seeds = 32 experiments

    Purpose: Identify how much data each method needs to perform well.
    Hypothesis: SVD-based methods (PiSSA, MiLoRA) need more data to initialize well.
    """
    methods = ['lora', 'dora', 'pissa', 'adalora']
    data_sizes = [1000, 2000, 4000, 8000]
    seeds = [42, 43]

    base = {
        'model': {
            'model_name_or_path': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
            'attn_implementation': 'flash_attention_2',
            'dtype': 'bfloat16',
        },
        'dataset': {
            'dataset_name_or_path': 'open-r1/DAPO-Math-17k-Processed',
        },
        'training': {
            'max_steps': 512,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 8,
            'learning_rate': 1.0e-5,
            'warmup_ratio': 0.1,
            'save_steps': 128,
            'logging_steps': 10,
            'eval_steps': 128,
            'use_liger_kernel': False,
            'gradient_checkpointing': True,
            'report_to': ['wandb'],
        },
        'peft': {
            'r': 32,  # Fixed rank - we're testing data scarcity, not rank
            'lora_alpha': 64,
            'lora_dropout': 0.05,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj',
                              'up_proj', 'down_proj', 'gate_proj'],
        },
        'tracker': {
            'enable_spectral_tracking': True,
            'enable_gradient_tracking': True,
            'spectral_log_frequency': 100,
            'gradient_log_frequency': 100,
            'save_trajectory': True,
        },
        'wandb': {
            'use_wandb': True,
            'project': 'peft-rlvr-mechanistic',
            'tags': ['stress-test', 'data-scarcity'],
        },
        'common': {
            'seed': 42,
            'output_dir': 'output/stress_data',
        },
    }

    experiments = []
    for method in methods:
        for n_samples in data_sizes:
            for seed in seeds:
                exp = {
                    'name': f'{method}_n{n_samples}_seed{seed}',
                    'peft': {
                        'type': method,
                        'r': 32,  # Explicit rank (same as base, for validator)
                        'lora_alpha': 64,
                    },
                    'dataset': {
                        'max_train_samples': n_samples,
                    },
                    'common': {
                        'seed': seed,
                    },
                    'wandb': {
                        'run_name': f'{method}_n{n_samples}_seed{seed}',
                        'tags': ['stress-test', 'data-scarcity', f'n{n_samples}', method],
                    },
                }
                experiments.append(exp)

    return {'base': base, 'experiments': experiments}


def validate_config(config: Dict[str, Any], name: str) -> bool:
    """Validate a stress test configuration."""
    errors = []

    # Check base structure
    if 'base' not in config:
        errors.append("Missing 'base' section")
    if 'experiments' not in config:
        errors.append("Missing 'experiments' section")

    if errors:
        print(f"  ✗ {name}: {', '.join(errors)}")
        return False

    base = config['base']
    experiments = config['experiments']

    # Check required base sections
    required_sections = ['model', 'dataset', 'training', 'tracker', 'wandb', 'common']
    for section in required_sections:
        if section not in base:
            errors.append(f"Missing base.{section}")

    # Check experiments
    if len(experiments) == 0:
        errors.append("No experiments defined")
    else:
        # Check each experiment has required fields
        for i, exp in enumerate(experiments):
            if 'name' not in exp:
                errors.append(f"Experiment {i} missing 'name'")
            if 'peft' not in exp:
                errors.append(f"Experiment '{exp.get('name', i)}' missing 'peft'")
            if 'common' not in exp:
                errors.append(f"Experiment '{exp.get('name', i)}' missing 'common'")

    # Check for duplicate experiment names
    names = [exp.get('name', '') for exp in experiments]
    duplicates = [n for n in set(names) if names.count(n) > 1]
    if duplicates:
        errors.append(f"Duplicate experiment names: {duplicates}")

    if errors:
        print(f"  ✗ {name}:")
        for error in errors:
            print(f"      - {error}")
        return False

    print(f"  ✓ {name}: {len(experiments)} experiments validated")
    return True


def print_dry_run(config: Dict[str, Any], name: str):
    """Print dry-run information for a config."""
    experiments = config.get('experiments', [])
    base = config.get('base', {})

    print(f"\n{'='*60}")
    print(f"DRY RUN: {name}")
    print(f"{'='*60}")

    print(f"\nBase Settings:")
    print(f"  Model: {base.get('model', {}).get('model_name_or_path', 'N/A')}")
    print(f"  Dataset: {base.get('dataset', {}).get('dataset_name_or_path', 'N/A')}")
    print(f"  Max Steps: {base.get('training', {}).get('max_steps', 'N/A')}")
    print(f"  Batch Size: {base.get('training', {}).get('per_device_train_batch_size', 'N/A')}")
    print(f"  Tracking: spectral={base.get('tracker', {}).get('enable_spectral_tracking', False)}, "
          f"gradient={base.get('tracker', {}).get('enable_gradient_tracking', False)}")

    print(f"\nExperiments ({len(experiments)} total):")

    # Group by method
    by_method = {}
    for exp in experiments:
        method = exp.get('peft', {}).get('type', 'unknown')
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(exp['name'])

    for method, names in sorted(by_method.items()):
        print(f"\n  {method} ({len(names)} experiments):")
        for name in names:
            print(f"    - {name}")

    # Estimate time and cost
    # Rough estimates: 512 steps ~15 minutes per experiment on A100
    time_per_exp = 0.25  # hours
    cost_per_hour = 3.0  # $/hour for A100

    total_hours = len(experiments) * time_per_exp
    total_cost = total_hours * cost_per_hour

    print(f"\n{'='*60}")
    print(f"ESTIMATES:")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Estimated time: ~{total_hours:.1f} hours")
    print(f"  Estimated cost: ~${total_cost:.0f} (A100 @ ${cost_per_hour}/hr)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stress test configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate generated configs'
    )
    parser.add_argument(
        '--dry_run', '-d',
        action='store_true',
        help='Show what would be generated without writing files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=Path,
        default=Path('configs/experiments'),
        help='Output directory for config files'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating Stress Test Configurations")
    print("=" * 60)
    print()

    # Generate configs
    print("1. Generating ultra-low rank stress test config...")
    stress_rank_config = generate_stress_rank_config()
    print(f"   Generated {len(stress_rank_config['experiments'])} experiments")

    print("\n2. Generating data scarcity stress test config...")
    stress_data_config = generate_stress_data_config()
    print(f"   Generated {len(stress_data_config['experiments'])} experiments")

    # Validate
    if args.validate or args.dry_run:
        print("\n3. Validating configurations...")
        valid_rank = validate_config(stress_rank_config, "stress_rank.yaml")
        valid_data = validate_config(stress_data_config, "stress_data.yaml")

        if not (valid_rank and valid_data):
            print("\n✗ Validation failed!")
            sys.exit(1)

    # Dry run
    if args.dry_run:
        print_dry_run(stress_rank_config, "stress_rank.yaml")
        print_dry_run(stress_data_config, "stress_data.yaml")

        total_experiments = (len(stress_rank_config['experiments']) +
                           len(stress_data_config['experiments']))
        print(f"\n{'='*60}")
        print(f"TOTAL STRESS TESTS: {total_experiments} experiments")
        print(f"{'='*60}")
        return

    # Write files
    print("\n3. Writing configuration files...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank_path = args.output_dir / 'stress_rank.yaml'
    write_yaml(stress_rank_config, rank_path)
    print(f"   ✓ {rank_path}")

    data_path = args.output_dir / 'stress_data.yaml'
    write_yaml(stress_data_config, data_path)
    print(f"   ✓ {data_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  stress_rank.yaml: {len(stress_rank_config['experiments'])} experiments")
    print(f"    - 4 methods (lora, dora, pissa, adalora)")
    print(f"    - 4 ranks (r=1, 2, 4, 8)")
    print(f"    - 2 seeds (42, 43)")
    print()
    print(f"  stress_data.yaml: {len(stress_data_config['experiments'])} experiments")
    print(f"    - 4 methods (lora, dora, pissa, adalora)")
    print(f"    - 4 data sizes (n=1K, 2K, 4K, 8K)")
    print(f"    - 2 seeds (42, 43)")
    print()
    print(f"  Total: {len(stress_rank_config['experiments']) + len(stress_data_config['experiments'])} experiments")
    print()
    print("To validate:")
    print(f"  python {__file__} --validate")
    print()
    print("To see dry-run:")
    print(f"  python {__file__} --dry_run")
    print("=" * 60)


if __name__ == '__main__':
    main()
