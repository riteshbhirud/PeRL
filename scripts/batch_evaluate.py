#!/usr/bin/env python3
"""
Batch Checkpoint Evaluation for PeRL.

Evaluates all checkpoints in a directory on multiple benchmarks.
Supports parallel execution across GPUs, resume functionality, and result aggregation.

Usage:
    # Evaluate all checkpoints
    python scripts/batch_evaluate.py \
        --checkpoint_dir outputs/core_1.5B \
        --benchmarks "aime2024,math500" \
        --output_dir results/evaluations

    # Evaluate specific methods only
    python scripts/batch_evaluate.py \
        --checkpoint_dir outputs/core_1.5B \
        --only "lora,dora,pissa" \
        --benchmarks "aime2024,math500"

    # Parallel evaluation on multiple GPUs
    python scripts/batch_evaluate.py \
        --checkpoint_dir outputs/core_1.5B \
        --benchmarks "aime2024,math500" \
        --parallel 4 \
        --gpu_ids "0,1,2,3"

    # Resume from failures
    python scripts/batch_evaluate.py \
        --checkpoint_dir outputs/core_1.5B \
        --benchmarks "aime2024,math500" \
        --skip_completed
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("perl.batch_evaluate")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckpointMeta:
    """Metadata about a discovered checkpoint."""
    path: str
    experiment_name: str
    peft_method: str
    seed: Optional[int]
    rank: Optional[int]
    step: Optional[int]

    @property
    def id(self) -> str:
        """Unique identifier for this checkpoint."""
        parts = [self.experiment_name]
        if self.step:
            parts.append(f"step{self.step}")
        return "_".join(parts)


@dataclass
class EvaluationTask:
    """A single evaluation task (checkpoint + benchmark)."""
    checkpoint: CheckpointMeta
    benchmark: str
    output_path: str
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class EvaluationStatus:
    """Overall evaluation status."""
    checkpoint_dir: str
    benchmarks: List[str]
    total_tasks: int
    completed: int
    failed: int
    pending: int
    running: int
    tasks: Dict[str, dict] = field(default_factory=dict)
    start_time: str = ""
    last_updated: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationStatus":
        return cls(**data)


# =============================================================================
# Checkpoint Discovery
# =============================================================================

def discover_checkpoints(
    checkpoint_dir: str,
    only_methods: Optional[List[str]] = None,
    exclude_methods: Optional[List[str]] = None,
    only_experiments: Optional[List[str]] = None,
) -> List[CheckpointMeta]:
    """
    Discover all checkpoints in a directory.

    Args:
        checkpoint_dir: Root directory to scan
        only_methods: Only include these PEFT methods
        exclude_methods: Exclude these PEFT methods
        only_experiments: Only include these experiment names

    Returns:
        List of CheckpointMeta objects
    """
    checkpoints = []
    root = Path(checkpoint_dir)

    if not root.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    logger.info(f"Scanning for checkpoints in: {checkpoint_dir}")

    # Look for adapter_config.json files
    for adapter_config in root.rglob("adapter_config.json"):
        checkpoint_path = adapter_config.parent

        # Parse metadata from path and config
        meta = parse_checkpoint_metadata(checkpoint_path)
        if meta is None:
            continue

        # Apply filters
        if only_methods and meta.peft_method.lower() not in [m.lower() for m in only_methods]:
            continue
        if exclude_methods and meta.peft_method.lower() in [m.lower() for m in exclude_methods]:
            continue
        if only_experiments and meta.experiment_name not in only_experiments:
            continue

        checkpoints.append(meta)

    # Sort by experiment name and step
    checkpoints.sort(key=lambda x: (x.experiment_name, x.step or 0))

    logger.info(f"Found {len(checkpoints)} checkpoints")

    return checkpoints


def parse_checkpoint_metadata(checkpoint_path: Path) -> Optional[CheckpointMeta]:
    """Parse checkpoint metadata from path and config."""
    try:
        # Load adapter config
        config_path = checkpoint_path / "adapter_config.json"
        if not config_path.exists():
            return None

        with open(config_path) as f:
            config = json.load(f)

        # Extract PEFT type
        peft_type = config.get("peft_type", "unknown").lower()
        if peft_type == "lora":
            if config.get("use_dora", False):
                peft_type = "dora"

        # Extract rank
        rank = config.get("r", config.get("rank"))

        # Parse path for experiment name, seed, step
        path_str = str(checkpoint_path)

        # Extract seed (s42, seed42, _s42)
        seed = None
        seed_match = re.search(r'[_/]s(?:eed)?(\d+)', path_str)
        if seed_match:
            seed = int(seed_match.group(1))

        # Extract step (checkpoint-1000)
        step = None
        step_match = re.search(r'checkpoint[_-](\d+)', checkpoint_path.name)
        if step_match:
            step = int(step_match.group(1))

        # Extract experiment name
        # Look for pattern like: outputs/core_1.5B/lora_r16_s42/checkpoint-1000
        parts = checkpoint_path.parts
        experiment_name = None

        for i, part in enumerate(parts):
            if part.startswith("checkpoint"):
                if i > 0:
                    experiment_name = parts[i - 1]
                break

        if not experiment_name:
            experiment_name = checkpoint_path.parent.name

        return CheckpointMeta(
            path=str(checkpoint_path),
            experiment_name=experiment_name,
            peft_method=peft_type,
            seed=seed,
            rank=rank,
            step=step,
        )

    except Exception as e:
        logger.debug(f"Failed to parse checkpoint {checkpoint_path}: {e}")
        return None


def group_checkpoints_by_experiment(
    checkpoints: List[CheckpointMeta]
) -> Dict[str, List[CheckpointMeta]]:
    """Group checkpoints by experiment name."""
    groups = {}
    for cp in checkpoints:
        if cp.experiment_name not in groups:
            groups[cp.experiment_name] = []
        groups[cp.experiment_name].append(cp)
    return groups


# =============================================================================
# Evaluation Planning
# =============================================================================

def create_evaluation_plan(
    checkpoints: List[CheckpointMeta],
    benchmarks: List[str],
    output_dir: str,
    skip_completed: bool = False,
) -> List[EvaluationTask]:
    """
    Create evaluation plan with all (checkpoint, benchmark) pairs.

    Args:
        checkpoints: List of checkpoints to evaluate
        benchmarks: List of benchmarks to evaluate on
        output_dir: Directory for results
        skip_completed: Skip already completed evaluations

    Returns:
        List of EvaluationTask objects
    """
    tasks = []
    skipped = 0

    for checkpoint in checkpoints:
        for benchmark in benchmarks:
            # Generate output path
            output_path = generate_result_path(checkpoint, benchmark, output_dir)

            # Check if already completed
            if skip_completed and Path(output_path).exists():
                skipped += 1
                continue

            task = EvaluationTask(
                checkpoint=checkpoint,
                benchmark=benchmark,
                output_path=output_path,
            )
            tasks.append(task)

    if skipped > 0:
        logger.info(f"Skipped {skipped} already completed evaluations")

    logger.info(f"Created evaluation plan with {len(tasks)} tasks")

    return tasks


def generate_result_path(
    checkpoint: CheckpointMeta,
    benchmark: str,
    output_dir: str,
) -> str:
    """Generate standardized result path."""
    # Create subdirectory structure: output_dir/experiment_name/checkpoint_step/benchmark.json
    subdir = checkpoint.experiment_name
    if checkpoint.step:
        subdir = f"{subdir}/step_{checkpoint.step}"

    benchmark_clean = benchmark.replace("/", "_").replace(" ", "_").lower()
    filename = f"{benchmark_clean}.json"

    return str(Path(output_dir) / subdir / filename)


def print_evaluation_plan(
    tasks: List[EvaluationTask],
    checkpoints: List[CheckpointMeta],
    benchmarks: List[str],
):
    """Print summary of evaluation plan."""
    print("\n" + "=" * 70)
    print("EVALUATION PLAN")
    print("=" * 70)

    # Group by experiment
    experiments = group_checkpoints_by_experiment(checkpoints)

    print(f"\nCheckpoints: {len(checkpoints)}")
    print(f"Benchmarks:  {len(benchmarks)} ({', '.join(benchmarks)})")
    print(f"Total tasks: {len(tasks)}")

    print(f"\nExperiments:")
    for exp_name, exp_checkpoints in sorted(experiments.items()):
        steps = [cp.step for cp in exp_checkpoints if cp.step]
        step_str = f"steps: {min(steps)}-{max(steps)}" if steps else "final"
        print(f"  {exp_name:<30} ({len(exp_checkpoints)} checkpoints, {step_str})")

    print("\n" + "=" * 70)


# =============================================================================
# Evaluation Execution
# =============================================================================

def run_evaluation(
    task: EvaluationTask,
    base_model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_problems: Optional[int] = None,
    gpu_id: Optional[int] = None,
    save_generations: bool = True,
) -> EvaluationTask:
    """
    Run a single evaluation task.

    Args:
        task: EvaluationTask to execute
        base_model: Override base model
        temperature: Generation temperature
        max_tokens: Max tokens to generate
        max_problems: Limit problems (for testing)
        gpu_id: GPU to use
        save_generations: Whether to save full responses

    Returns:
        Updated EvaluationTask with status
    """
    task.status = "running"
    task.start_time = datetime.now().isoformat()

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate_checkpoint.py"),
        "--checkpoint", task.checkpoint.path,
        "--benchmark", task.benchmark,
        "--output_dir", str(Path(task.output_path).parent.parent),
        "--temperature", str(temperature),
        "--max_tokens", str(max_tokens),
        "--no_tqdm",
    ]

    if base_model:
        cmd.extend(["--base_model", base_model])

    if max_problems:
        cmd.extend(["--max_problems", str(max_problems)])

    if not save_generations:
        cmd.append("--no_save_generations")

    # Set environment for GPU
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        logger.info(f"Evaluating {task.checkpoint.experiment_name} on {task.benchmark}")

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            task.status = "completed"
            logger.info(f"  ✓ Completed: {task.checkpoint.experiment_name} on {task.benchmark}")
        else:
            task.status = "failed"
            task.error = result.stderr[:1000] if result.stderr else "Unknown error"
            logger.error(f"  ✗ Failed: {task.checkpoint.experiment_name} on {task.benchmark}")
            logger.error(f"    Error: {task.error[:200]}")

    except subprocess.TimeoutExpired:
        task.status = "failed"
        task.error = "Timeout after 1 hour"
        logger.error(f"  ✗ Timeout: {task.checkpoint.experiment_name} on {task.benchmark}")

    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        logger.error(f"  ✗ Error: {task.checkpoint.experiment_name} on {task.benchmark}: {e}")

    task.end_time = datetime.now().isoformat()
    return task


def run_evaluation_worker(args: tuple) -> EvaluationTask:
    """Worker function for parallel evaluation."""
    task, kwargs = args
    return run_evaluation(task, **kwargs)


def run_sequential(
    tasks: List[EvaluationTask],
    status_path: str,
    **eval_kwargs,
) -> List[EvaluationTask]:
    """Run evaluations sequentially."""
    results = []
    status = create_status(tasks)

    try:
        from tqdm import tqdm
        pbar = tqdm(tasks, desc="Evaluating", unit="task")
    except ImportError:
        pbar = tasks

    for task in pbar:
        result = run_evaluation(task, **eval_kwargs)
        results.append(result)

        # Update status
        update_status(status, result)
        save_status(status, status_path)

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                "completed": status.completed,
                "failed": status.failed,
            })

    return results


def run_parallel(
    tasks: List[EvaluationTask],
    status_path: str,
    num_workers: int = 4,
    gpu_ids: Optional[List[int]] = None,
    **eval_kwargs,
) -> List[EvaluationTask]:
    """Run evaluations in parallel across GPUs."""
    results = []
    status = create_status(tasks)

    # Assign GPUs to workers
    if gpu_ids is None:
        gpu_ids = list(range(num_workers))

    # Prepare worker arguments
    worker_args = []
    for i, task in enumerate(tasks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        kwargs = {**eval_kwargs, "gpu_id": gpu_id}
        worker_args.append((task, kwargs))

    logger.info(f"Starting parallel evaluation with {num_workers} workers")
    logger.info(f"GPUs: {gpu_ids}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_evaluation_worker, args): args[0]
            for args in worker_args
        }

        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks), desc="Evaluating", unit="task")
        except ImportError:
            pbar = None

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)

                # Update status
                update_status(status, result)
                save_status(status, status_path)

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        "completed": status.completed,
                        "failed": status.failed,
                    })

            except Exception as e:
                task = futures[future]
                task.status = "failed"
                task.error = str(e)
                results.append(task)
                logger.error(f"Worker error: {e}")

        if pbar:
            pbar.close()

    return results


# =============================================================================
# Status Management
# =============================================================================

def create_status(tasks: List[EvaluationTask]) -> EvaluationStatus:
    """Create initial evaluation status."""
    benchmarks = list(set(t.benchmark for t in tasks))
    checkpoint_dirs = list(set(t.checkpoint.path for t in tasks))

    return EvaluationStatus(
        checkpoint_dir=checkpoint_dirs[0] if checkpoint_dirs else "",
        benchmarks=benchmarks,
        total_tasks=len(tasks),
        completed=0,
        failed=0,
        pending=len(tasks),
        running=0,
        start_time=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
    )


def update_status(status: EvaluationStatus, task: EvaluationTask):
    """Update status after task completion."""
    task_id = f"{task.checkpoint.id}_{task.benchmark}"

    status.tasks[task_id] = {
        "checkpoint": task.checkpoint.path,
        "benchmark": task.benchmark,
        "status": task.status,
        "error": task.error,
        "start_time": task.start_time,
        "end_time": task.end_time,
    }

    # Recount
    statuses = [t["status"] for t in status.tasks.values()]
    status.completed = statuses.count("completed")
    status.failed = statuses.count("failed")
    status.running = statuses.count("running")
    status.pending = status.total_tasks - status.completed - status.failed - status.running
    status.last_updated = datetime.now().isoformat()


def save_status(status: EvaluationStatus, path: str):
    """Save status to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(status.to_dict(), f, indent=2)


def load_status(path: str) -> Optional[EvaluationStatus]:
    """Load status from JSON file."""
    if not Path(path).exists():
        return None
    with open(path) as f:
        return EvaluationStatus.from_dict(json.load(f))


# =============================================================================
# Result Aggregation (Basic)
# =============================================================================

def aggregate_results(output_dir: str) -> dict:
    """
    Aggregate evaluation results from output directory.

    Returns dictionary of aggregated results by method and benchmark.
    """
    from perl.evaluation import load_results

    results_by_method = {}
    output_path = Path(output_dir)

    for result_file in output_path.rglob("*.json"):
        if result_file.name == "evaluation_status.json":
            continue
        if result_file.name == "aggregated_results.json":
            continue

        try:
            result = load_results(result_file)
            method = result.metadata.peft_method
            benchmark = result.metadata.benchmark
            seed = result.metadata.seed
            accuracy = result.statistics.accuracy

            if method not in results_by_method:
                results_by_method[method] = {}

            if benchmark not in results_by_method[method]:
                results_by_method[method][benchmark] = {"seeds": {}, "accuracies": []}

            results_by_method[method][benchmark]["seeds"][seed] = accuracy
            results_by_method[method][benchmark]["accuracies"].append(accuracy)

        except Exception as e:
            logger.debug(f"Failed to load {result_file}: {e}")

    # Compute statistics
    import numpy as np

    for method in results_by_method:
        for benchmark in results_by_method[method]:
            accs = results_by_method[method][benchmark]["accuracies"]
            results_by_method[method][benchmark].update({
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)) if len(accs) > 1 else 0.0,
                "min": float(np.min(accs)),
                "max": float(np.max(accs)),
                "n": len(accs),
            })

    return results_by_method


def print_aggregated_results(results: dict):
    """Print aggregated results as a table."""
    if not results:
        print("No results to display")
        return

    # Get all benchmarks
    benchmarks = set()
    for method_results in results.values():
        benchmarks.update(method_results.keys())
    benchmarks = sorted(benchmarks)

    # Print header
    header = f"{'Method':<15}"
    for b in benchmarks:
        header += f" | {b[:12]:<14}"
    print("\n" + "=" * len(header))
    print("AGGREGATED RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Print rows
    for method in sorted(results.keys()):
        row = f"{method:<15}"
        for benchmark in benchmarks:
            if benchmark in results[method]:
                stats = results[method][benchmark]
                if stats["std"] > 0:
                    val = f"{stats['mean']:.3f}±{stats['std']:.3f}"
                else:
                    val = f"{stats['mean']:.3f}"
                row += f" | {val:<14}"
            else:
                row += f" | {'-':<14}"
        print(row)

    print("=" * len(header) + "\n")


def save_aggregated_results(results: dict, output_path: str):
    """Save aggregated results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved aggregated results to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch checkpoint evaluation for PeRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Checkpoint discovery
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoints to evaluate",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only evaluate these PEFT methods (comma-separated)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Exclude these PEFT methods (comma-separated)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Only evaluate these experiments (comma-separated)",
    )

    # Benchmarks
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="aime2024,math500",
        help="Benchmarks to evaluate on (comma-separated)",
    )

    # Output
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results/evaluations",
        help="Directory for evaluation results",
    )

    # Execution
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (1 for sequential)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="GPU IDs to use (comma-separated)",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip already completed evaluations",
    )

    # Generation settings
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Override base model path",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Limit problems per benchmark (for testing)",
    )
    parser.add_argument(
        "--no_save_generations",
        action="store_true",
        help="Don't save full model responses",
    )

    # Control
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show plan without executing",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    # Parse arguments
    only_methods = [m.strip() for m in args.only.split(",")] if args.only else None
    exclude_methods = [m.strip() for m in args.exclude.split(",")] if args.exclude else None
    only_experiments = [e.strip() for e in args.experiments.split(",")] if args.experiments else None
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None

    # Discover checkpoints
    checkpoints = discover_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        only_methods=only_methods,
        exclude_methods=exclude_methods,
        only_experiments=only_experiments,
    )

    if not checkpoints:
        logger.error("No checkpoints found!")
        sys.exit(1)

    # Create evaluation plan
    tasks = create_evaluation_plan(
        checkpoints=checkpoints,
        benchmarks=benchmarks,
        output_dir=args.output_dir,
        skip_completed=args.skip_completed,
    )

    if not tasks:
        logger.info("No tasks to run (all completed or no checkpoints found)")
        return

    # Print plan
    print_evaluation_plan(tasks, checkpoints, benchmarks)

    if args.dry_run:
        print("\n[DRY RUN] Would execute the above plan")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    status_path = os.path.join(args.output_dir, "evaluation_status.json")

    # Prepare evaluation kwargs
    eval_kwargs = {
        "base_model": args.base_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_problems": args.max_problems,
        "save_generations": not args.no_save_generations,
    }

    # Run evaluations
    start_time = time.time()

    if args.parallel > 1:
        results = run_parallel(
            tasks=tasks,
            status_path=status_path,
            num_workers=args.parallel,
            gpu_ids=gpu_ids,
            **eval_kwargs,
        )
    else:
        results = run_sequential(
            tasks=tasks,
            status_path=status_path,
            **eval_kwargs,
        )

    elapsed = time.time() - start_time

    # Print summary
    completed = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total tasks:  {len(results)}")
    print(f"Completed:    {completed}")
    print(f"Failed:       {failed}")
    print(f"Total time:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results:      {args.output_dir}")

    # Aggregate and display results
    if completed > 0:
        aggregated = aggregate_results(args.output_dir)
        print_aggregated_results(aggregated)

        # Save aggregated results
        agg_path = os.path.join(args.output_dir, "aggregated_results.json")
        save_aggregated_results(aggregated, agg_path)

    # Print failed tasks
    if failed > 0:
        print("\nFailed evaluations:")
        for r in results:
            if r.status == "failed":
                print(f"  - {r.checkpoint.experiment_name} on {r.benchmark}")
                if r.error:
                    print(f"    Error: {r.error[:100]}")


if __name__ == "__main__":
    main()
