#!/usr/bin/env python3
"""
Experiment Status Checker for PeRL.

Check the status of running, completed, and failed experiments without
logging into the cluster.

Usage:
    python scripts/check_experiment_status.py --output_dir output/core_1.5B
    python scripts/check_experiment_status.py --output_dir output/core_1.5B --verbose
    python scripts/check_experiment_status.py --output_dir output/core_1.5B --json
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentStatus:
    """Status information for a single experiment."""
    name: str
    status: str  # 'completed', 'running', 'failed', 'not_started'
    output_dir: str
    duration_minutes: Optional[float] = None
    elapsed_minutes: Optional[float] = None
    estimated_remaining_minutes: Optional[float] = None
    checkpoint_count: int = 0
    final_checkpoint_exists: bool = False
    size_gb: Optional[float] = None
    last_log_time: Optional[datetime] = None
    error_type: Optional[str] = None
    final_loss: Optional[float] = None
    final_step: Optional[int] = None


@dataclass
class BatchStatus:
    """Overall status of a batch of experiments."""
    output_dir: str
    total: int = 0
    completed: int = 0
    running: int = 0
    failed: int = 0
    not_started: int = 0
    experiments: List[ExperimentStatus] = field(default_factory=list)
    total_duration_minutes: float = 0.0
    avg_duration_minutes: float = 0.0
    total_size_gb: float = 0.0
    estimated_completion: Optional[datetime] = None
    estimated_remaining_hours: Optional[float] = None


def get_directory_size(path: str) -> float:
    """Get total size of a directory in GB."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except (OSError, PermissionError):
        pass
    return total_size / (1024 ** 3)  # Convert to GB


def parse_log_file(log_path: str) -> Dict[str, Any]:
    """Parse experiment log file for useful information."""
    info = {
        'last_step': None,
        'last_loss': None,
        'error_type': None,
        'last_line_time': None,
        'is_training': False,
    }

    if not os.path.exists(log_path):
        return info

    try:
        with open(log_path, 'r', errors='ignore') as f:
            lines = f.readlines()

        # Get last modification time
        info['last_line_time'] = datetime.fromtimestamp(os.path.getmtime(log_path))

        # Parse lines for information
        for line in reversed(lines[-100:]):  # Check last 100 lines
            line_lower = line.lower()

            # Check for errors
            if info['error_type'] is None:
                if 'cuda out of memory' in line_lower or 'outofmemoryerror' in line_lower:
                    info['error_type'] = 'CUDA out of memory'
                elif 'runtimeerror' in line_lower:
                    info['error_type'] = 'RuntimeError'
                elif 'valueerror' in line_lower:
                    info['error_type'] = 'ValueError'
                elif 'filenotfounderror' in line_lower:
                    info['error_type'] = 'FileNotFoundError'
                elif 'connectionerror' in line_lower or 'timeout' in line_lower:
                    info['error_type'] = 'Connection/Timeout error'
                elif 'killed' in line_lower or 'sigkill' in line_lower:
                    info['error_type'] = 'Process killed (OOM?)'
                elif 'error' in line_lower and 'traceback' in line_lower:
                    info['error_type'] = 'Unknown error'

            # Check for training step info
            if info['last_step'] is None:
                # Common TRL/transformers log format: {'loss': 0.123, ...} or step 100/1000
                if "'loss':" in line or '"loss":' in line:
                    try:
                        # Extract loss value
                        import re
                        loss_match = re.search(r"['\"]loss['\"]:\s*([\d.]+)", line)
                        if loss_match:
                            info['last_loss'] = float(loss_match.group(1))
                        step_match = re.search(r"['\"]global_step['\"]:\s*(\d+)", line)
                        if step_match:
                            info['last_step'] = int(step_match.group(1))
                    except (ValueError, AttributeError):
                        pass

                # Alternative format: Step 100/1000
                if info['last_step'] is None:
                    import re
                    step_match = re.search(r'step\s*(\d+)', line_lower)
                    if step_match:
                        info['last_step'] = int(step_match.group(1))

            # Check if training is actively happening
            if 'training' in line_lower and ('step' in line_lower or 'epoch' in line_lower):
                info['is_training'] = True

    except Exception:
        pass

    return info


def check_experiment_status(exp_dir: str, exp_name: str, avg_duration: Optional[float] = None) -> ExperimentStatus:
    """Check the status of a single experiment."""
    status = ExperimentStatus(
        name=exp_name,
        status='not_started',
        output_dir=exp_dir,
    )

    if not os.path.exists(exp_dir):
        return status

    # Get directory size
    status.size_gb = get_directory_size(exp_dir)

    # Check for checkpoints
    checkpoint_dirs = glob.glob(os.path.join(exp_dir, 'checkpoint-*'))
    status.checkpoint_count = len(checkpoint_dirs)

    # Check for final checkpoint or completion markers
    final_markers = [
        os.path.join(exp_dir, 'adapter_model.safetensors'),
        os.path.join(exp_dir, 'adapter_model.bin'),
        os.path.join(exp_dir, 'pytorch_model.bin'),
        os.path.join(exp_dir, 'model.safetensors'),
    ]
    status.final_checkpoint_exists = any(os.path.exists(m) for m in final_markers)

    # Check for experiment metadata
    metadata_path = os.path.join(exp_dir, 'experiment_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if metadata.get('status') == 'completed':
                status.status = 'completed'
                if 'duration_seconds' in metadata:
                    status.duration_minutes = metadata['duration_seconds'] / 60
            elif metadata.get('status') == 'failed':
                status.status = 'failed'
                status.error_type = metadata.get('error_type', 'Unknown')
        except (json.JSONDecodeError, IOError):
            pass

    # Parse log file for more info
    log_paths = [
        os.path.join(exp_dir, 'training.log'),
        os.path.join(exp_dir, 'experiment.log'),
        os.path.join(os.path.dirname(exp_dir), 'logs', f'{exp_name}.log'),
    ]

    log_info = {}
    for log_path in log_paths:
        if os.path.exists(log_path):
            log_info = parse_log_file(log_path)
            status.last_log_time = log_info.get('last_line_time')
            status.final_loss = log_info.get('last_loss')
            status.final_step = log_info.get('last_step')
            if log_info.get('error_type') and status.status != 'completed':
                status.error_type = log_info['error_type']
            break

    # Determine status based on evidence
    if status.status == 'not_started':
        if status.final_checkpoint_exists and status.checkpoint_count > 0:
            status.status = 'completed'
        elif status.error_type:
            status.status = 'failed'
        elif status.checkpoint_count > 0 or (status.last_log_time and
              (datetime.now() - status.last_log_time).total_seconds() < 600):  # 10 min threshold
            status.status = 'running'
            # Calculate elapsed time
            if status.last_log_time:
                # Find start time from first checkpoint or metadata
                start_time = None
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        if 'start_time' in metadata:
                            start_time = datetime.fromisoformat(metadata['start_time'])
                    except:
                        pass

                if start_time is None:
                    # Use directory creation time as fallback
                    start_time = datetime.fromtimestamp(os.path.getctime(exp_dir))

                status.elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60

                # Estimate remaining time
                if avg_duration and status.elapsed_minutes:
                    status.estimated_remaining_minutes = max(0, avg_duration - status.elapsed_minutes)
        elif status.size_gb and status.size_gb > 0.001:  # Some content exists
            # Check if log was recently updated
            if status.last_log_time:
                time_since_update = (datetime.now() - status.last_log_time).total_seconds()
                if time_since_update > 3600:  # More than 1 hour since last update
                    status.status = 'failed'
                    status.error_type = status.error_type or 'Stalled (no recent activity)'
                else:
                    status.status = 'running'

    return status


def scan_experiments(output_dir: str, config_path: Optional[str] = None) -> BatchStatus:
    """Scan output directory for all experiments and their statuses."""
    batch_status = BatchStatus(output_dir=output_dir)

    # Try to get experiment list from config or status file
    experiment_names = []

    # Check for run status file
    status_file = os.path.join(output_dir, 'experiment_status.json')
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                run_status = json.load(f)
            experiment_names = list(run_status.get('experiments', {}).keys())
        except (json.JSONDecodeError, IOError):
            pass

    # If no status file, scan directories
    if not experiment_names:
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path) and item not in ['logs', 'wandb', '__pycache__']:
                    experiment_names.append(item)

    if not experiment_names:
        return batch_status

    # First pass: get completed durations for average
    completed_durations = []
    for exp_name in experiment_names:
        exp_dir = os.path.join(output_dir, exp_name)
        metadata_path = os.path.join(exp_dir, 'experiment_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if metadata.get('status') == 'completed' and 'duration_seconds' in metadata:
                    completed_durations.append(metadata['duration_seconds'] / 60)
            except:
                pass

    avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else None

    # Second pass: check all experiments
    for exp_name in sorted(experiment_names):
        exp_dir = os.path.join(output_dir, exp_name)
        exp_status = check_experiment_status(exp_dir, exp_name, avg_duration)
        batch_status.experiments.append(exp_status)

        # Update counters
        batch_status.total += 1
        if exp_status.status == 'completed':
            batch_status.completed += 1
            if exp_status.duration_minutes:
                batch_status.total_duration_minutes += exp_status.duration_minutes
        elif exp_status.status == 'running':
            batch_status.running += 1
        elif exp_status.status == 'failed':
            batch_status.failed += 1
        else:
            batch_status.not_started += 1

        if exp_status.size_gb:
            batch_status.total_size_gb += exp_status.size_gb

    # Calculate averages and estimates
    if batch_status.completed > 0:
        batch_status.avg_duration_minutes = batch_status.total_duration_minutes / batch_status.completed

        # Estimate completion time
        remaining = batch_status.not_started + batch_status.running
        if remaining > 0 and batch_status.avg_duration_minutes > 0:
            # Account for running experiments
            running_remaining = 0
            for exp in batch_status.experiments:
                if exp.status == 'running' and exp.estimated_remaining_minutes:
                    running_remaining += exp.estimated_remaining_minutes

            # Total remaining time
            total_remaining_minutes = (batch_status.not_started * batch_status.avg_duration_minutes) + running_remaining
            batch_status.estimated_remaining_hours = total_remaining_minutes / 60
            batch_status.estimated_completion = datetime.now() + timedelta(minutes=total_remaining_minutes)

    return batch_status


def format_duration(minutes: Optional[float]) -> str:
    """Format duration in human-readable form."""
    if minutes is None:
        return "N/A"
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} hr"
    days = hours / 24
    return f"{days:.1f} days"


def format_size(gb: Optional[float]) -> str:
    """Format size in human-readable form."""
    if gb is None:
        return "N/A"
    if gb < 0.001:
        return "< 1 MB"
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.1f} GB"


def create_progress_bar(progress: float, width: int = 30) -> str:
    """Create ASCII progress bar."""
    filled = int(width * progress)
    empty = width - filled
    return f"[{'#' * filled}{'-' * empty}]"


def print_status(batch_status: BatchStatus, verbose: bool = False):
    """Print formatted status report."""
    print("=" * 50)
    print(f"Experiment Status: {batch_status.output_dir}")
    print("=" * 50)

    print(f"\nTotal experiments: {batch_status.total}")
    print(f"  Completed: {batch_status.completed} \033[92m✓\033[0m")
    print(f"  Running:   {batch_status.running} \033[93m⧗\033[0m")
    print(f"  Failed:    {batch_status.failed} \033[91m✗\033[0m")
    print(f"  Not started: {batch_status.not_started} ○")

    # Completed experiments
    completed_exps = [e for e in batch_status.experiments if e.status == 'completed']
    if completed_exps:
        print(f"\n\033[92mCompleted:\033[0m")
        for exp in completed_exps[:10]:  # Show first 10
            duration_str = format_duration(exp.duration_minutes)
            size_str = format_size(exp.size_gb)
            loss_str = f", loss={exp.final_loss:.4f}" if exp.final_loss else ""
            print(f"  ✓ {exp.name:<25} ({duration_str}, {size_str}{loss_str})")
        if len(completed_exps) > 10:
            print(f"  ... ({len(completed_exps) - 10} more)")

    # Running experiments
    running_exps = [e for e in batch_status.experiments if e.status == 'running']
    if running_exps:
        print(f"\n\033[93mRunning:\033[0m")
        for exp in running_exps:
            elapsed_str = format_duration(exp.elapsed_minutes)
            remaining_str = f", ~{format_duration(exp.estimated_remaining_minutes)} remaining" if exp.estimated_remaining_minutes else ""
            step_str = f", step {exp.final_step}" if exp.final_step else ""
            print(f"  ⧗ {exp.name:<25} ({elapsed_str} elapsed{remaining_str}{step_str})")

    # Failed experiments
    failed_exps = [e for e in batch_status.experiments if e.status == 'failed']
    if failed_exps:
        print(f"\n\033[91mFailed:\033[0m")
        for exp in failed_exps:
            error_str = f" - {exp.error_type}" if exp.error_type else ""
            print(f"  ✗ {exp.name:<25}{error_str}")

    # Not started
    not_started_exps = [e for e in batch_status.experiments if e.status == 'not_started']
    if not_started_exps:
        print(f"\n○ Not started:")
        if verbose:
            for exp in not_started_exps:
                print(f"  ○ {exp.name}")
        else:
            for exp in not_started_exps[:5]:
                print(f"  ○ {exp.name}")
            if len(not_started_exps) > 5:
                print(f"  ... ({len(not_started_exps) - 5} more)")

    # Progress bar
    if batch_status.total > 0:
        progress = batch_status.completed / batch_status.total
        progress_bar = create_progress_bar(progress)
        print(f"\nProgress: {progress_bar} {progress * 100:.1f}%")

    # Estimates
    if batch_status.estimated_completion and batch_status.completed < batch_status.total:
        print(f"\nEstimated completion: {batch_status.estimated_completion.strftime('%Y-%m-%d %H:%M')}")
        if batch_status.estimated_remaining_hours:
            if batch_status.estimated_remaining_hours < 1:
                print(f"Time remaining: ~{batch_status.estimated_remaining_hours * 60:.0f} minutes")
            elif batch_status.estimated_remaining_hours < 24:
                print(f"Time remaining: ~{batch_status.estimated_remaining_hours:.1f} hours")
            else:
                print(f"Time remaining: ~{batch_status.estimated_remaining_hours / 24:.1f} days")

    # Storage
    if batch_status.total_size_gb > 0:
        avg_size = batch_status.total_size_gb / max(1, batch_status.completed + batch_status.running)
        estimated_total = avg_size * batch_status.total
        print(f"\nStorage: {format_size(batch_status.total_size_gb)} used")
        if batch_status.completed < batch_status.total:
            print(f"Estimated total: ~{format_size(estimated_total)}")

    # Average duration
    if batch_status.avg_duration_minutes > 0:
        print(f"\nAverage experiment duration: {format_duration(batch_status.avg_duration_minutes)}")


def print_json(batch_status: BatchStatus):
    """Print status as JSON."""
    output = {
        'output_dir': batch_status.output_dir,
        'total': batch_status.total,
        'completed': batch_status.completed,
        'running': batch_status.running,
        'failed': batch_status.failed,
        'not_started': batch_status.not_started,
        'total_size_gb': batch_status.total_size_gb,
        'avg_duration_minutes': batch_status.avg_duration_minutes,
        'estimated_remaining_hours': batch_status.estimated_remaining_hours,
        'estimated_completion': batch_status.estimated_completion.isoformat() if batch_status.estimated_completion else None,
        'experiments': [asdict(e) for e in batch_status.experiments],
    }

    # Convert datetime objects to strings
    for exp in output['experiments']:
        if exp['last_log_time']:
            exp['last_log_time'] = exp['last_log_time'].isoformat()

    print(json.dumps(output, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Check the status of PeRL experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status of experiments
  python scripts/check_experiment_status.py --output_dir output/core_1.5B

  # Verbose output (show all experiments)
  python scripts/check_experiment_status.py --output_dir output/core_1.5B --verbose

  # JSON output for scripting
  python scripts/check_experiment_status.py --output_dir output/core_1.5B --json

  # Watch mode (refresh every 30 seconds)
  python scripts/check_experiment_status.py --output_dir output/core_1.5B --watch
"""
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory containing experiments"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all experiments including not started"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output status as JSON"
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch mode: refresh status every 30 seconds"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval in seconds for watch mode (default: 30)"
    )

    args = parser.parse_args()

    # Resolve output directory
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(project_root, args.output_dir)

    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory does not exist: {args.output_dir}")
        sys.exit(1)

    if args.watch:
        import time
        try:
            while True:
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')

                batch_status = scan_experiments(args.output_dir)

                if args.json:
                    print_json(batch_status)
                else:
                    print_status(batch_status, args.verbose)
                    print(f"\n[Refreshing every {args.interval}s. Press Ctrl+C to stop]")

                # Check if all done
                if batch_status.completed + batch_status.failed == batch_status.total:
                    print("\n\033[92mAll experiments finished!\033[0m")
                    break

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        batch_status = scan_experiments(args.output_dir)

        if args.json:
            print_json(batch_status)
        else:
            print_status(batch_status, args.verbose)


if __name__ == "__main__":
    main()
