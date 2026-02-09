#!/usr/bin/env python3
"""
PeRL Experiment Runner - Batch execution of experiments from YAML configs.

This script loads experiment configurations from YAML files and executes them
systematically with support for:
- Sequential or parallel execution
- Progress tracking and status persistence
- Resume from failures
- Dry run mode with time estimates
- Selective experiment execution
- Per-experiment metadata logging

Usage:
    # Run all experiments from a config
    python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml

    # Run specific experiments by name
    python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
        --only "lora_r16_s42,dora_r16_s42"

    # Skip completed experiments (resume)
    python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
        --skip_completed

    # Parallel execution with 4 GPUs
    python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
        --parallel 4

    # Dry run (preview commands with time estimates)
    python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
        --dry_run
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


# =============================================================================
# Constants
# =============================================================================

# Estimated time per experiment (in minutes) based on model size and steps
# These are rough estimates for A100 GPUs
ESTIMATED_TIME_MINUTES = {
    "1.5B": {
        1000: 45,   # 1000 steps
        500: 25,
        100: 5,
    },
    "7B": {
        1000: 120,  # 1000 steps
        500: 65,
        100: 15,
    },
}

# Cost per A100 GPU-hour (rough estimate)
A100_COST_PER_HOUR = 3.0  # $3/hour typical cloud pricing


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentStatus:
    """Status of a single experiment."""
    name: str
    status: str  # "pending", "running", "completed", "failed", "skipped"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    output_dir: Optional[str] = None
    log_file: Optional[str] = None
    command: Optional[str] = None


@dataclass
class RunStatus:
    """Overall status of a batch run."""
    config_file: str
    start_time: str
    end_time: Optional[str] = None
    total_experiments: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    pending: int = 0
    remaining: int = 0
    experiments: Dict[str, ExperimentStatus] = field(default_factory=dict)


# =============================================================================
# Config Loading and Merging
# =============================================================================

def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def merge_experiment_config(base: Dict, experiment: Dict) -> Dict:
    """
    Merge base config with experiment-specific overrides.

    Args:
        base: Base configuration from YAML
        experiment: Experiment-specific configuration

    Returns:
        Complete merged configuration for the experiment
    """
    merged = {}

    # Start with base config sections
    for section in ['model', 'dataset', 'training', 'tracker', 'peft', 'common']:
        if section in base:
            merged[section] = base[section].copy()

    # Merge experiment overrides
    for section, values in experiment.items():
        if section == 'name':
            continue  # Skip name, not a config section
        if section in merged:
            merged[section] = deep_merge(merged[section], values)
        else:
            merged[section] = values

    return merged


def build_command(
    experiment_name: str,
    config: Dict[str, Any],
    base_output_dir: str,
    config_name: str,
    accelerate_config: Optional[str] = None,
    main_process_port: int = 29500,
) -> Tuple[List[str], str, str]:
    """
    Build the command line arguments for running an experiment.

    Args:
        experiment_name: Name of the experiment
        config: Merged configuration dictionary
        base_output_dir: Base directory for outputs
        config_name: Name of the config file (for organizing outputs)
        accelerate_config: Path to accelerate config file (enables multi-GPU)
        main_process_port: Port for distributed training coordination

    Returns:
        Tuple of (command list, output directory path, command string)
    """
    # Determine output directory for this experiment
    output_dir = os.path.join(base_output_dir, config_name, experiment_name)

    # Build command with accelerate launch for multi-GPU training
    if accelerate_config:
        cmd = [
            "accelerate", "launch",
            "--main_process_port", str(main_process_port),
            "--config_file", accelerate_config,
            "run.py", "train"
        ]
    else:
        cmd = ["python", "run.py", "train"]

    # Add all config sections as command line arguments
    def add_args(prefix: str, d: Dict):
        for key, value in d.items():
            arg_name = f"--config.{prefix}.{key}" if prefix else f"--config.{key}"

            if isinstance(value, bool):
                cmd.extend([arg_name, str(value).lower()])
            elif isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings or JSON
                cmd.extend([arg_name, json.dumps(value)])
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                add_args(f"{prefix}.{key}" if prefix else key, value)
            else:
                cmd.extend([arg_name, str(value)])

    # Add each config section
    for section, values in config.items():
        if isinstance(values, dict):
            add_args(section, values)

    # Override output directory
    cmd.extend(["--config.training.output_dir", output_dir])

    # Build command string for logging
    cmd_str = " ".join(cmd)

    return cmd, output_dir, cmd_str


def estimate_experiment_time(config: Dict[str, Any]) -> float:
    """
    Estimate experiment runtime in minutes.

    Args:
        config: Merged experiment config

    Returns:
        Estimated time in minutes
    """
    # Get model size from path
    model_path = config.get('model', {}).get('model_name_or_path', '')
    if '7B' in model_path or '7b' in model_path:
        model_size = "7B"
    else:
        model_size = "1.5B"

    # Get max steps
    max_steps = config.get('training', {}).get('max_steps', 1000)

    # Look up estimated time
    time_map = ESTIMATED_TIME_MINUTES.get(model_size, ESTIMATED_TIME_MINUTES["1.5B"])

    # Find closest step count
    closest_steps = min(time_map.keys(), key=lambda x: abs(x - max_steps))
    base_time = time_map[closest_steps]

    # Scale by steps ratio
    estimated = base_time * (max_steps / closest_steps)

    return estimated


# =============================================================================
# Status Tracking
# =============================================================================

def get_status_file_path(config_file: str, output_dir: str) -> str:
    """Get the path to the status file for a config."""
    config_name = Path(config_file).stem
    return os.path.join(output_dir, config_name, "experiment_status.json")


def load_run_status(status_file: str) -> Optional[RunStatus]:
    """Load existing run status from file."""
    if not os.path.exists(status_file):
        return None

    try:
        with open(status_file, 'r') as f:
            data = json.load(f)

        # Reconstruct RunStatus
        experiments = {}
        for name, exp_data in data.get('experiments', {}).items():
            experiments[name] = ExperimentStatus(**exp_data)

        return RunStatus(
            config_file=data['config_file'],
            start_time=data['start_time'],
            end_time=data.get('end_time'),
            total_experiments=data['total_experiments'],
            completed=data['completed'],
            failed=data['failed'],
            skipped=data['skipped'],
            pending=data['pending'],
            remaining=data.get('remaining', data['pending']),
            experiments=experiments,
        )
    except Exception as e:
        print(f"Warning: Could not load status file: {e}")
        return None


def save_run_status(status: RunStatus, status_file: str):
    """Save run status to file."""
    os.makedirs(os.path.dirname(status_file), exist_ok=True)

    # Convert to serializable dict
    data = {
        'config_file': status.config_file,
        'start_time': status.start_time,
        'end_time': status.end_time,
        'total_experiments': status.total_experiments,
        'completed': status.completed,
        'failed': status.failed,
        'skipped': status.skipped,
        'pending': status.pending,
        'remaining': status.remaining,
        'experiments': {
            name: asdict(exp) for name, exp in status.experiments.items()
        }
    }

    with open(status_file, 'w') as f:
        json.dump(data, f, indent=2)


def save_experiment_metadata(
    experiment_name: str,
    output_dir: str,
    config: Dict[str, Any],
    command: str,
    status: str,
    start_time: str,
    end_time: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    error_message: Optional[str] = None,
):
    """Save per-experiment metadata to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "experiment_name": experiment_name,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "status": status,
        "command": command,
        "config": config,
    }

    if error_message:
        metadata["error_message"] = error_message

    metadata_path = os.path.join(output_dir, "experiment_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def is_experiment_completed(output_dir: str) -> bool:
    """Check if an experiment has completed successfully."""
    # Check for experiment_metadata.json with completed status
    metadata_path = os.path.join(output_dir, "experiment_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if metadata.get('status') == 'completed':
                return True
        except:
            pass

    # Check for adapter_model files (PEFT)
    adapter_files = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_config.json",
    ]
    for f in adapter_files:
        if os.path.exists(os.path.join(output_dir, f)):
            return True

    return False


# =============================================================================
# Progress Display
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m:.0f}m {s:.0f}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:.0f}h {m:.0f}m"


def format_duration_minutes(minutes: float) -> str:
    """Format duration from minutes."""
    if minutes < 60:
        return f"~{minutes:.0f} minutes"
    else:
        hours = minutes / 60
        return f"~{hours:.1f} hours"


def print_experiment_start(name: str, index: int, total: int, output_dir: str, command: str):
    """Print experiment start message."""
    print()
    print("=" * 60)
    print(f"Running experiment {index}/{total}: {name}")
    print(f"Command: python run.py --config...")
    print(f"Output dir: {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def print_experiment_end(status: ExperimentStatus, remaining: int, total: int, estimated_per_exp: float):
    """Print experiment end message."""
    duration_str = format_duration(status.duration_seconds) if status.duration_seconds else "N/A"

    if status.status == "completed":
        print(f"\n✓ Completed: {status.name} ({duration_str})")
    elif status.status == "failed":
        print(f"\n✗ Failed: {status.name}")
        if status.error_message:
            print(f"  Error: {status.error_message}")
    else:
        print(f"\n⊘ {status.status.capitalize()}: {status.name}")

    # Show remaining info
    print(f"  Remaining: {remaining}/{total} experiments")
    if remaining > 0 and estimated_per_exp > 0:
        estimated_remaining = remaining * estimated_per_exp
        print(f"  Estimated time remaining: {format_duration_minutes(estimated_remaining)}")

    if status.log_file:
        print(f"  Log: {status.log_file}")
    print("-" * 60)


def print_summary(run_status: RunStatus, failed_experiments: List[Tuple[str, str]]):
    """Print final summary of the run."""
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Completed: {run_status.completed}/{run_status.total_experiments}")
    print(f"Failed: {run_status.failed}/{run_status.total_experiments}")

    if failed_experiments:
        print()
        for name, error in failed_experiments:
            print(f"  - {name} ({error})")
        print()
        failed_names = ",".join([name for name, _ in failed_experiments])
        print("To retry failed experiments:")
        print(f'  python scripts/run_experiments.py --config {run_status.config_file} --only "{failed_names}"')

    print("=" * 60)


def print_dry_run_summary(experiments: List[Tuple[str, str, str, Dict, float]], config_file: str):
    """Print dry run summary with time estimates."""
    print()
    print("=" * 60)
    print("[DRY RUN] Would execute {} experiments:".format(len(experiments)))
    print("=" * 60)

    total_time = 0
    for name, output_dir, cmd_str, config, est_time in experiments:
        print()
        print(f"  {name}")
        print(f"    Command: python run.py --config.model...")
        print(f"    Output: {output_dir}")
        print(f"    Estimated time: {format_duration_minutes(est_time)}")
        total_time += est_time

    print()
    print("-" * 60)
    print(f"Total experiments: {len(experiments)}")
    print(f"Total estimated time: {format_duration_minutes(total_time)}")

    # Calculate GPU-hours and cost
    gpu_hours = total_time / 60
    estimated_cost = gpu_hours * A100_COST_PER_HOUR
    print(f"Total estimated cost: {gpu_hours:.1f} A100-hours (${estimated_cost:.0f})")
    print("=" * 60)


# =============================================================================
# Experiment Execution
# =============================================================================

def run_experiment(
    experiment_name: str,
    cmd: List[str],
    cmd_str: str,
    output_dir: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    gpu_id: Optional[int] = None,
    cuda_devices: Optional[str] = None,
) -> ExperimentStatus:
    """
    Run a single experiment.

    Args:
        experiment_name: Name of the experiment
        cmd: Command to execute as list
        cmd_str: Command as string for logging
        output_dir: Output directory for the experiment
        config: Merged config for metadata
        dry_run: If True, just print the command
        gpu_id: GPU ID to use (for single-GPU CUDA_VISIBLE_DEVICES)
        cuda_devices: CUDA_VISIBLE_DEVICES string for multi-GPU (e.g., "0,1,2,3")

    Returns:
        ExperimentStatus with results
    """
    start_time = datetime.now().isoformat()

    status = ExperimentStatus(
        name=experiment_name,
        status="running",
        start_time=start_time,
        output_dir=output_dir,
        command=cmd_str,
    )

    # Log file goes in the experiment output directory
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "experiment.log")
    status.log_file = log_file

    if dry_run:
        status.status = "skipped"
        status.end_time = datetime.now().isoformat()
        return status

    # Save initial metadata
    save_experiment_metadata(
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=config,
        command=cmd_str,
        status="running",
        start_time=start_time,
    )

    # Prepare environment for distributed training
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        env["ACCELERATE_LOG_LEVEL"] = "info"
    elif gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Run the experiment
        with open(log_file, 'w') as log_f:
            log_f.write(f"# Experiment: {experiment_name}\n")
            log_f.write(f"# Started: {start_time}\n")
            log_f.write(f"# Command: {cmd_str}\n")
            log_f.write("=" * 80 + "\n\n")
            log_f.flush()

            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Project root
            )

            # Wait for completion
            exit_code = process.wait()

            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(f"# Finished: {datetime.now().isoformat()}\n")
            log_f.write(f"# Exit code: {exit_code}\n")

        status.end_time = datetime.now().isoformat()
        status.exit_code = exit_code

        if exit_code == 0:
            status.status = "completed"
        else:
            status.status = "failed"
            status.error_message = f"Process exited with code {exit_code}"

    except Exception as e:
        status.end_time = datetime.now().isoformat()
        status.status = "failed"
        status.error_message = str(e)

    # Calculate duration
    if status.start_time and status.end_time:
        start = datetime.fromisoformat(status.start_time)
        end = datetime.fromisoformat(status.end_time)
        status.duration_seconds = (end - start).total_seconds()

    # Save final metadata
    save_experiment_metadata(
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=config,
        command=cmd_str,
        status=status.status,
        start_time=status.start_time,
        end_time=status.end_time,
        duration_seconds=status.duration_seconds,
        error_message=status.error_message,
    )

    return status


def run_experiment_wrapper(args: tuple) -> ExperimentStatus:
    """Wrapper for parallel execution."""
    return run_experiment(*args)


# =============================================================================
# Main Runner Functions
# =============================================================================

def create_progress_bar_description(name: str, elapsed_minutes: float = 0, eta_minutes: float = 0) -> str:
    """Create a description string for the progress bar."""
    if eta_minutes > 0:
        if eta_minutes < 60:
            eta_str = f"{eta_minutes:.0f}m"
        else:
            eta_str = f"{eta_minutes/60:.1f}h"
        return f"Current: {name} | ETA: {eta_str}"
    return f"Current: {name}"


def run_experiments_sequential(
    experiments: List[Tuple[str, List[str], str, str, Dict, float]],
    run_status: RunStatus,
    status_file: str,
    dry_run: bool = False,
    use_progress_bar: bool = True,
    cuda_devices: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Run experiments one at a time.

    Args:
        experiments: List of (name, cmd, cmd_str, output_dir, config, est_time)
        run_status: Status tracker
        status_file: Path to save status
        dry_run: If True, don't actually run
        use_progress_bar: If True, show tqdm progress bar
        cuda_devices: CUDA_VISIBLE_DEVICES string for multi-GPU

    Returns:
        List of (name, error_message) for failed experiments
    """
    failed = []
    total = len(experiments)

    # Calculate average estimated time
    avg_est_time = sum(e[5] for e in experiments) / total if total > 0 else 45
    total_est_time = sum(e[5] for e in experiments)

    # Create progress bar if tqdm available
    pbar = None
    if TQDM_AVAILABLE and use_progress_bar and not dry_run:
        pbar = tqdm(
            total=total,
            desc="Running experiments",
            unit="exp",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
        )

    completed_times = []  # Track actual completion times for better estimates

    for i, (name, cmd, cmd_str, output_dir, config, est_time) in enumerate(experiments, 1):
        remaining = total - i
        start_time = time.time()

        # Update progress bar description
        if pbar:
            # Calculate ETA based on actual times if available
            if completed_times:
                actual_avg = sum(completed_times) / len(completed_times)
                eta_minutes = remaining * actual_avg
            else:
                eta_minutes = remaining * avg_est_time
            pbar.set_description(create_progress_bar_description(name, eta_minutes=eta_minutes))

        if not pbar:
            print_experiment_start(name, i, total, output_dir, cmd_str)

        status = run_experiment(
            experiment_name=name,
            cmd=cmd,
            cmd_str=cmd_str,
            output_dir=output_dir,
            config=config,
            dry_run=dry_run,
            cuda_devices=cuda_devices,
        )

        run_status.experiments[name] = status

        # Track actual time
        if status.duration_seconds:
            completed_times.append(status.duration_seconds / 60)

        if status.status == "completed":
            run_status.completed += 1
            if pbar:
                pbar.write(f"  ✓ {name} ({format_duration(status.duration_seconds) if status.duration_seconds else 'N/A'})")
        elif status.status == "failed":
            run_status.failed += 1
            failed.append((name, status.error_message or "Unknown error"))
            if pbar:
                pbar.write(f"  ✗ {name} - {status.error_message or 'Unknown error'}")

        run_status.pending -= 1
        run_status.remaining = remaining

        # Update progress bar
        if pbar:
            pbar.update(1)
        else:
            print_experiment_end(status, remaining, total, avg_est_time)

        if not dry_run:
            save_run_status(run_status, status_file)

    # Close progress bar
    if pbar:
        pbar.close()
        print()  # Add newline after progress bar

    return failed


def run_experiments_parallel(
    experiments: List[Tuple[str, List[str], str, str, Dict, float]],
    run_status: RunStatus,
    status_file: str,
    n_parallel: int,
    gpu_ids: Optional[List[int]] = None,
    use_progress_bar: bool = True,
) -> List[Tuple[str, str]]:
    """
    Run experiments in parallel.

    Args:
        experiments: List of (name, cmd, cmd_str, output_dir, config, est_time)
        run_status: Status tracker
        status_file: Path to save status
        n_parallel: Number of parallel workers
        gpu_ids: List of GPU IDs to use
        use_progress_bar: If True, show tqdm progress bar

    Returns:
        List of (name, error_message) for failed experiments
    """
    failed = []
    total = len(experiments)

    if gpu_ids is None:
        gpu_ids = list(range(n_parallel))

    print(f"\nRunning {total} experiments in parallel with {n_parallel} workers")
    print(f"GPU IDs: {gpu_ids}")
    print("-" * 60)

    # Create progress bar if tqdm available
    pbar = None
    if TQDM_AVAILABLE and use_progress_bar:
        pbar = tqdm(
            total=total,
            desc="Parallel execution",
            unit="exp",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
        )

    with ProcessPoolExecutor(max_workers=n_parallel) as executor:
        # Submit all experiments
        futures = {}
        for i, (name, cmd, cmd_str, output_dir, config, est_time) in enumerate(experiments):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(
                run_experiment,
                name, cmd, cmd_str, output_dir, config, False, gpu_id
            )
            futures[future] = (name, i + 1)

        # Process results as they complete
        completed_count = 0
        for future in as_completed(futures):
            name, index = futures[future]
            completed_count += 1

            try:
                status = future.result()
                run_status.experiments[name] = status

                if status.status == "completed":
                    run_status.completed += 1
                    duration_str = format_duration(status.duration_seconds) if status.duration_seconds else ""
                    if pbar:
                        pbar.write(f"  ✓ {name} ({duration_str})")
                    else:
                        print(f"✓ [{completed_count}/{total}] Completed: {name} ({duration_str})")
                elif status.status == "failed":
                    run_status.failed += 1
                    failed.append((name, status.error_message or "Unknown error"))
                    if pbar:
                        pbar.write(f"  ✗ {name} - {status.error_message or 'Unknown error'}")
                    else:
                        print(f"✗ [{completed_count}/{total}] Failed: {name}")

                run_status.pending -= 1
                run_status.remaining = total - completed_count
                save_run_status(run_status, status_file)

                if pbar:
                    pbar.update(1)

            except Exception as e:
                if pbar:
                    pbar.write(f"  ✗ {name} - Error: {e}")
                else:
                    print(f"✗ [{completed_count}/{total}] Error in {name}: {e}")
                run_status.experiments[name] = ExperimentStatus(
                    name=name,
                    status="failed",
                    error_message=str(e),
                )
                run_status.failed += 1
                failed.append((name, str(e)))
                run_status.pending -= 1
                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()
        print()  # Add newline after progress bar

    return failed


def run_experiments(
    config_file: str,
    output_dir: str = "output",
    only: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    skip_completed: bool = False,
    parallel: int = 1,
    dry_run: bool = False,
    gpu_ids: Optional[List[int]] = None,
    use_progress_bar: bool = True,
    accelerate_config: Optional[str] = None,
    cuda_devices: str = "0,1,2,3",
    main_process_port: int = 29500,
) -> RunStatus:
    """
    Run experiments from a YAML config file.

    Args:
        config_file: Path to YAML config file
        output_dir: Base directory for outputs
        only: List of experiment names to run (None = all)
        exclude: List of experiment names to exclude
        skip_completed: Skip experiments that have completed
        parallel: Number of parallel executions (1 = sequential)
        dry_run: If True, just print commands without executing
        gpu_ids: List of GPU IDs to use for parallel execution
        use_progress_bar: If True, show tqdm progress bar
        accelerate_config: Path to accelerate config file for multi-GPU
        cuda_devices: CUDA_VISIBLE_DEVICES string
        main_process_port: Base port for distributed training

    Returns:
        RunStatus with results
    """
    # Load config
    print(f"Loading config: {config_file}")
    config = load_yaml_config(config_file)

    if 'base' not in config or 'experiments' not in config:
        raise ValueError("Config must have 'base' and 'experiments' sections")

    base_config = config['base']
    experiments = config['experiments']
    config_name = Path(config_file).stem

    # Setup status tracking
    status_file = get_status_file_path(config_file, output_dir)

    # Load existing status if resuming
    run_status = None
    if skip_completed:
        run_status = load_run_status(status_file)

    if run_status is None:
        run_status = RunStatus(
            config_file=config_file,
            start_time=datetime.now().isoformat(),
            total_experiments=len(experiments),
        )

    # Build list of experiments to run
    experiments_to_run = []
    skipped_count = 0
    exp_index = 0  # Track experiment index for port assignment

    for exp in experiments:
        name = exp['name']

        # Check only/exclude filters
        if only and name not in only:
            continue
        if exclude and name in exclude:
            continue

        # Merge config
        merged_config = merge_experiment_config(base_config, exp)

        # Build command with accelerate support
        # Each experiment gets a unique port to avoid conflicts
        exp_port = main_process_port + exp_index
        cmd, exp_output_dir, cmd_str = build_command(
            name, merged_config, output_dir, config_name,
            accelerate_config=accelerate_config,
            main_process_port=exp_port
        )
        est_time = estimate_experiment_time(merged_config)
        exp_index += 1

        # Check if already completed
        if skip_completed:
            if name in run_status.experiments:
                if run_status.experiments[name].status == "completed":
                    print(f"  Skipping completed: {name}")
                    skipped_count += 1
                    continue
            elif is_experiment_completed(exp_output_dir):
                print(f"  Skipping completed (found outputs): {name}")
                run_status.experiments[name] = ExperimentStatus(
                    name=name,
                    status="completed",
                    output_dir=exp_output_dir,
                )
                skipped_count += 1
                continue

        experiments_to_run.append((name, cmd, cmd_str, exp_output_dir, merged_config, est_time))

        # Initialize status
        if name not in run_status.experiments:
            run_status.experiments[name] = ExperimentStatus(
                name=name,
                status="pending",
            )

    run_status.pending = len(experiments_to_run)
    run_status.remaining = len(experiments_to_run)
    run_status.skipped = skipped_count

    print(f"\nExperiments to run: {len(experiments_to_run)}/{len(experiments)}")
    if skipped_count > 0:
        print(f"Skipped (already completed): {skipped_count}")

    if not experiments_to_run:
        print("\nNo experiments to run.")
        return run_status

    # Dry run mode
    if dry_run:
        dry_run_data = [
            (name, output_dir, cmd_str, config, est_time)
            for name, cmd, cmd_str, output_dir, config, est_time in experiments_to_run
        ]
        print_dry_run_summary(dry_run_data, config_file)
        return run_status

    # Save initial status
    save_run_status(run_status, status_file)

    # Run experiments
    if parallel > 1:
        failed = run_experiments_parallel(
            experiments_to_run,
            run_status,
            status_file,
            parallel,
            gpu_ids,
            use_progress_bar=use_progress_bar,
        )
    else:
        failed = run_experiments_sequential(
            experiments_to_run,
            run_status,
            status_file,
            dry_run=False,
            use_progress_bar=use_progress_bar,
            cuda_devices=cuda_devices,
        )

    # Finalize
    run_status.end_time = datetime.now().isoformat()
    save_run_status(run_status, status_file)

    print_summary(run_status, failed)

    return run_status


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run PeRL experiments from YAML config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List experiments in a config
  python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml --list

  # Dry run (preview with time estimates)
  python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml --dry_run

  # Run all experiments sequentially
  python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml

  # Run specific experiments only
  python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \\
      --only "lora_r16_s42,dora_r16_s42"

  # Resume from previous run (skip completed)
  python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \\
      --skip_completed

  # Parallel execution on 4 GPUs
  python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \\
      --parallel 4 --gpu_ids "0,1,2,3"
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="output",
        help="Base output directory (default: output)",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Comma-separated list of experiment names to run",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="Comma-separated list of experiment names to exclude",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip experiments that have already completed",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel experiments (default: 1 = sequential)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        help="Comma-separated GPU IDs for parallel execution (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and estimates without executing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_experiments",
        help="List experiments in the config without running",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar (use detailed output instead)",
    )
    parser.add_argument(
        "--accelerate_config",
        type=str,
        default="scripts/trl/accelerate/ds_zero2_4gpu.yaml",
        help="Path to accelerate config file for multi-GPU training (default: scripts/trl/accelerate/ds_zero2_4gpu.yaml)",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1,2,3",
        help="CUDA_VISIBLE_DEVICES to use (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--no_accelerate",
        action="store_true",
        help="Use single-GPU mode (python run.py) instead of accelerate launch",
    )
    parser.add_argument(
        "--main_process_port",
        type=int,
        default=29500,
        help="Base port for distributed training (default: 29500, incremented per experiment)",
    )

    args = parser.parse_args()

    # Make paths absolute
    if not os.path.isabs(args.config):
        args.config = os.path.abspath(args.config)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)

    # Check config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Parse only/exclude lists
    only = args.only.split(",") if args.only else None
    exclude = args.exclude.split(",") if args.exclude else None
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None

    # List mode
    if args.list_experiments:
        config = load_yaml_config(args.config)
        print(f"\nExperiments in {args.config}:\n")
        for i, exp in enumerate(config.get('experiments', []), 1):
            name = exp.get('name', f'experiment_{i}')
            peft_type = exp.get('peft', {}).get('type', 'unknown')
            print(f"  {i:3d}. {name} (peft: {peft_type})")
        print(f"\nTotal: {len(config.get('experiments', []))} experiments")
        sys.exit(0)

    # Determine accelerate config
    accelerate_config = None if args.no_accelerate else args.accelerate_config

    # Run experiments
    print("=" * 60)
    print("PeRL Experiment Runner")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {'dry-run' if args.dry_run else 'parallel x' + str(args.parallel) if args.parallel > 1 else 'sequential'}")
    if accelerate_config:
        print(f"Accelerate: {accelerate_config}")
        print(f"CUDA devices: {args.cuda_devices}")
    else:
        print("Accelerate: disabled (single-GPU mode)")
    if args.parallel > 1:
        print(f"Workers: {args.parallel}")

    try:
        run_status = run_experiments(
            config_file=args.config,
            output_dir=args.output_dir,
            only=only,
            exclude=exclude,
            skip_completed=args.skip_completed,
            parallel=args.parallel,
            dry_run=args.dry_run,
            gpu_ids=gpu_ids,
            use_progress_bar=not args.no_progress,
            accelerate_config=accelerate_config,
            cuda_devices=args.cuda_devices,
            main_process_port=args.main_process_port,
        )

        # Exit with error code if any experiments failed
        if run_status.failed > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
