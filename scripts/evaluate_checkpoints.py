#!/usr/bin/env python3
"""
Batch Checkpoint Evaluator for PeRL.

Evaluates multiple checkpoints on specified datasets and generates learning curves.

Usage:
    python scripts/evaluate_checkpoints.py \
        --checkpoints output/lora_r16_s42/checkpoint-* \
        --datasets aime2024,math500 \
        --rollout-n 4 \
        --output eval_results/

    # Evaluate specific steps
    python scripts/evaluate_checkpoints.py \
        --checkpoints output/lora_r16_s42/checkpoint-100 \
                      output/lora_r16_s42/checkpoint-200 \
        --datasets math500

    # Dry run (preview without execution)
    python scripts/evaluate_checkpoints.py \
        --checkpoints output/*/checkpoint-* \
        --dry-run
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from perl.eval import (
    StageContext,
    setup_logging,
    merge_model_if_needed,
    grade_answer_perl,
    extract_vllm_args,
    start_vllm_processes,
    stop_vllm_processes,
    wait_for_vllm_ready,
    generate_with_vllm_async,
)
from perl.eval.vllm import VLLMConfig

logger = logging.getLogger("perl.eval.checkpoints")


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASETS = {
    "aime2024": {
        "path": "HuggingFaceH4/aime_2024",
        "split": "train",
        "prompt_field": "problem",
        "answer_field": "answer",
    },
    "aime2025": {
        "path": "yentinglin/aime_2025",
        "split": "train",
        "prompt_field": "problem",
        "answer_field": "answer",
    },
    "amc2023": {
        "path": "zwhe99/amc23",
        "split": "test",
        "prompt_field": "problem",
        "answer_field": "answer",
    },
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "split": "test",
        "prompt_field": "problem",
        "answer_field": "answer",
    },
    "minerva": {
        "path": "math-ai/minervamath",
        "split": "test",
        "prompt_field": "problem",
        "answer_field": "answer",
    },
    "hmmt2025": {
        "path": "FlagEval/HMMT_2025",
        "split": "train",
        "prompt_field": "problem",
        "answer_field": "answer",
    },
}

# System prompt for evaluation
SYSTEM_PROMPT = """You are a helpful assistant that solves math problems step by step.
Think through the problem carefully, showing your work.
Put your final answer in \\boxed{} format."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: str
    step: int
    experiment_name: str

    @classmethod
    def from_path(cls, path: str) -> "CheckpointInfo":
        """Parse checkpoint info from path."""
        path = str(Path(path).resolve())

        # Extract step from checkpoint-N pattern
        step_match = re.search(r'checkpoint-(\d+)', path)
        step = int(step_match.group(1)) if step_match else 0

        # Extract experiment name from parent directory
        parent = Path(path).parent.name
        if parent.startswith("checkpoint"):
            parent = Path(path).parent.parent.name

        return cls(path=path, step=step, experiment_name=parent)


@dataclass
class EvalResult:
    """Evaluation result for a single checkpoint-dataset pair."""
    checkpoint: str
    step: int
    dataset: str
    accuracy: float
    format_score: float
    num_problems: int
    num_correct: int
    pass_at_k: float
    avg_at_k: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    problem_results: List[Dict] = field(default_factory=list)


@dataclass
class EvalConfig:
    """Configuration for batch evaluation."""
    checkpoints: List[str]
    datasets: List[str]
    output_dir: str
    base_model: Optional[str] = None
    rollout_n: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 16384
    dp_size: int = 1
    tp_size: int = 1
    serve_port: int = 8000
    max_concurrent: int = 16
    dry_run: bool = False
    skip_existing: bool = False
    log_level: str = "INFO"


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(dataset_name: str) -> List[Dict]:
    """Load evaluation dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]

    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info(f"Loading dataset: {config['path']} ({config['split']})")
        ds = hf_load_dataset(config["path"], split=config["split"])

        problems = []
        for i, item in enumerate(ds):
            problems.append({
                "id": i,
                "prompt": item[config["prompt_field"]],
                "answer": str(item[config["answer_field"]]),
            })

        logger.info(f"Loaded {len(problems)} problems from {dataset_name}")
        return problems

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def format_prompt(problem: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Format problem with system prompt for evaluation."""
    return f"{system_prompt}\n\nProblem: {problem}\n\nSolution:"


# =============================================================================
# Evaluation Logic
# =============================================================================

async def evaluate_checkpoint(
    checkpoint: CheckpointInfo,
    dataset_name: str,
    problems: List[Dict],
    config: EvalConfig,
    vllm_configs: List[VLLMConfig],
) -> EvalResult:
    """Evaluate a single checkpoint on a dataset."""

    logger.info(f"Evaluating {checkpoint.experiment_name} step {checkpoint.step} on {dataset_name}")

    # Generate prompts
    prompts = [format_prompt(p["prompt"]) for p in problems]

    # Expand prompts for rollout_n
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * config.rollout_n)

    # Generate responses
    logger.info(f"Generating {len(expanded_prompts)} responses...")
    responses = await generate_with_vllm_async(
        configs=vllm_configs,
        prompts=expanded_prompts,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        max_concurrent_per_server=config.max_concurrent,
    )

    # Grade responses
    problem_results = []
    total_correct = 0
    total_format = 0.0
    num_pass = 0

    for i, problem in enumerate(problems):
        start_idx = i * config.rollout_n
        end_idx = start_idx + config.rollout_n
        problem_responses = responses[start_idx:end_idx]

        scores = []
        format_scores = []

        for response in problem_responses:
            acc, fmt = grade_answer_perl(response, problem["answer"])
            scores.append(acc)
            format_scores.append(fmt)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        avg_format = sum(format_scores) / len(format_scores) if format_scores else 0.0

        total_correct += avg_score
        total_format += avg_format
        if max_score > 0:
            num_pass += 1

        problem_results.append({
            "id": problem["id"],
            "prompt": problem["prompt"],
            "answer": problem["answer"],
            "responses": problem_responses,
            "scores": scores,
            "avg": avg_score,
            "max": max_score,
            "format_avg": avg_format,
        })

    # Compute aggregate metrics
    n = len(problems)
    accuracy = total_correct / n if n > 0 else 0.0
    format_score = total_format / n if n > 0 else 0.0
    pass_at_k = num_pass / n if n > 0 else 0.0

    return EvalResult(
        checkpoint=checkpoint.path,
        step=checkpoint.step,
        dataset=dataset_name,
        accuracy=accuracy,
        format_score=format_score,
        num_problems=n,
        num_correct=int(total_correct),
        pass_at_k=pass_at_k,
        avg_at_k=accuracy,
        problem_results=problem_results,
    )


def save_result(result: EvalResult, output_dir: str):
    """Save evaluation result to disk."""
    # Create output directory
    result_dir = Path(output_dir) / f"step_{result.step}" / result.dataset
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        "checkpoint": result.checkpoint,
        "step": result.step,
        "dataset": result.dataset,
        "accuracy": result.accuracy,
        "format_score": result.format_score,
        "num_problems": result.num_problems,
        "num_correct": result.num_correct,
        "pass_at_k": result.pass_at_k,
        "avg_at_k": result.avg_at_k,
        "timestamp": result.timestamp,
    }

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    with open(result_dir / "results.jsonl", "w") as f:
        for pr in result.problem_results:
            f.write(json.dumps(pr) + "\n")

    logger.info(f"Saved results to {result_dir}")


def check_existing_result(step: int, dataset: str, output_dir: str) -> bool:
    """Check if result already exists."""
    result_path = Path(output_dir) / f"step_{step}" / dataset / "summary.json"
    return result_path.exists()


# =============================================================================
# Learning Curve Generation
# =============================================================================

def generate_learning_curves(output_dir: str):
    """Generate learning curve data from all results."""
    output_path = Path(output_dir)

    # Collect all results
    curves = {}  # dataset -> [(step, accuracy)]

    for step_dir in sorted(output_path.glob("step_*")):
        step = int(step_dir.name.split("_")[1])

        for dataset_dir in step_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            summary_path = dataset_dir / "summary.json"
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                summary = json.load(f)

            dataset = summary["dataset"]
            if dataset not in curves:
                curves[dataset] = []

            curves[dataset].append({
                "step": step,
                "accuracy": summary["accuracy"],
                "pass_at_k": summary["pass_at_k"],
                "format_score": summary["format_score"],
            })

    # Sort by step
    for dataset in curves:
        curves[dataset].sort(key=lambda x: x["step"])

    # Save learning curves
    curves_path = output_path / "learning_curves.json"
    with open(curves_path, "w") as f:
        json.dump(curves, f, indent=2)

    logger.info(f"Generated learning curves: {curves_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Learning Curves Summary")
    print("=" * 70)

    for dataset, points in curves.items():
        print(f"\n{dataset}:")
        print(f"  {'Step':>8}  {'Accuracy':>10}  {'Pass@k':>10}  {'Format':>10}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
        for p in points:
            print(f"  {p['step']:>8}  {p['accuracy']:>10.4f}  {p['pass_at_k']:>10.4f}  {p['format_score']:>10.4f}")

    return curves


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

async def run_evaluation(config: EvalConfig):
    """Run batch checkpoint evaluation."""

    # Setup logging
    log_file = Path(config.output_dir) / "eval.log"
    os.makedirs(config.output_dir, exist_ok=True)
    setup_logging(config.log_level, str(log_file))

    # Parse checkpoints
    checkpoints = []
    for pattern in config.checkpoints:
        if "*" in pattern:
            matches = glob.glob(pattern)
            checkpoints.extend(matches)
        else:
            checkpoints.append(pattern)

    # Create CheckpointInfo objects
    checkpoint_infos = []
    for cp in checkpoints:
        if os.path.isdir(cp):
            checkpoint_infos.append(CheckpointInfo.from_path(cp))

    # Sort by step
    checkpoint_infos.sort(key=lambda x: x.step)

    if not checkpoint_infos:
        logger.error("No valid checkpoints found")
        return

    logger.info(f"Found {len(checkpoint_infos)} checkpoints to evaluate")
    for cp in checkpoint_infos:
        logger.info(f"  - Step {cp.step}: {cp.path}")

    # Preview datasets
    logger.info(f"Datasets: {config.datasets}")

    if config.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - No evaluation will be performed")
        print("=" * 70)
        print(f"\nCheckpoints ({len(checkpoint_infos)}):")
        for cp in checkpoint_infos:
            print(f"  Step {cp.step:>6}: {cp.path}")
        print(f"\nDatasets: {', '.join(config.datasets)}")
        print(f"Rollout N: {config.rollout_n}")
        print(f"Output: {config.output_dir}")

        total_evals = len(checkpoint_infos) * len(config.datasets)
        print(f"\nTotal evaluations: {total_evals}")
        return

    # Determine model path
    if config.base_model:
        model_path = config.base_model
    else:
        # Use first checkpoint's model (assuming adapter)
        # Try to find adapter_config.json to get base model
        first_cp = checkpoint_infos[0].path
        adapter_config = Path(first_cp) / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config) as f:
                ac = json.load(f)
                model_path = ac.get("base_model_name_or_path", first_cp)
        else:
            model_path = first_cp

    logger.info(f"Base model: {model_path}")

    # Load datasets
    datasets_data = {}
    for ds_name in config.datasets:
        datasets_data[ds_name] = load_dataset(ds_name)

    # Create vLLM configs (one per checkpoint evaluation)
    # We'll restart vLLM for each checkpoint to load the merged model

    results = []

    for cp_idx, checkpoint in enumerate(checkpoint_infos):
        logger.info(f"\n{'='*60}")
        logger.info(f"Checkpoint {cp_idx + 1}/{len(checkpoint_infos)}: Step {checkpoint.step}")
        logger.info(f"{'='*60}")

        # Skip if all datasets already evaluated
        if config.skip_existing:
            all_exist = all(
                check_existing_result(checkpoint.step, ds, config.output_dir)
                for ds in config.datasets
            )
            if all_exist:
                logger.info(f"Skipping step {checkpoint.step} (already evaluated)")
                continue

        # Merge adapter if needed
        with StageContext("Model Merging", logger) as stage:
            merged_path = merge_model_if_needed(
                model_path=model_path,
                adapter_path=checkpoint.path,
                output_path=None,  # Auto-generate
                logger=logger,
            )
            stage.log(f"Using model: {merged_path}")

        # Create vLLM config for merged model
        vllm_configs = []
        for i in range(config.dp_size):
            vllm_configs.append(VLLMConfig(
                model=merged_path,
                port=config.serve_port + i,
                tensor_parallel_size=config.tp_size,
            ))

        # Start vLLM servers
        with StageContext("Starting vLLM", logger) as stage:
            processes = start_vllm_processes(vllm_configs)
            stage.log(f"Started {len(processes)} vLLM server(s)")

        try:
            # Wait for servers to be ready
            with StageContext("Waiting for vLLM", logger) as stage:
                ready = await wait_for_vllm_ready(vllm_configs, timeout=300)
                if not ready:
                    raise RuntimeError("vLLM servers failed to start")
                stage.log("All servers ready")

            # Evaluate on each dataset
            for ds_name, problems in datasets_data.items():
                if config.skip_existing and check_existing_result(
                    checkpoint.step, ds_name, config.output_dir
                ):
                    logger.info(f"Skipping {ds_name} (already evaluated)")
                    continue

                with StageContext(f"Evaluating {ds_name}", logger) as stage:
                    result = await evaluate_checkpoint(
                        checkpoint=checkpoint,
                        dataset_name=ds_name,
                        problems=problems,
                        config=config,
                        vllm_configs=vllm_configs,
                    )

                    stage.log(f"Accuracy: {result.accuracy:.4f}, Pass@k: {result.pass_at_k:.4f}")

                    # Save result
                    save_result(result, config.output_dir)
                    results.append(result)

        finally:
            # Stop vLLM servers
            with StageContext("Stopping vLLM", logger):
                stop_vllm_processes(processes)

    # Generate learning curves
    generate_learning_curves(config.output_dir)

    # Print final summary
    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    print(f"Results saved to: {config.output_dir}")
    print(f"Total evaluations: {len(results)}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch checkpoint evaluator for PeRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint directories or glob patterns (e.g., output/*/checkpoint-*)",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="math500",
        help=f"Comma-separated dataset names. Available: {','.join(DATASETS.keys())}",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path (auto-detected from adapter_config.json if not specified)",
    )

    parser.add_argument(
        "--rollout-n",
        type=int,
        default=4,
        help="Number of generations per problem (default: 4)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate (default: 16384)",
    )

    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel size (number of vLLM instances)",
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size per vLLM instance",
    )

    parser.add_argument(
        "--serve-port",
        type=int,
        default=8000,
        help="Base port for vLLM servers",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Maximum concurrent requests per server",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints that have already been evaluated",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview evaluation plan without running",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(",")]

    # Validate datasets
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Error: Unknown dataset '{ds}'")
            print(f"Available datasets: {', '.join(DATASETS.keys())}")
            sys.exit(1)

    # Create config
    config = EvalConfig(
        checkpoints=args.checkpoints,
        datasets=datasets,
        output_dir=args.output,
        base_model=args.base_model,
        rollout_n=args.rollout_n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        serve_port=args.serve_port,
        max_concurrent=args.max_concurrent,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )

    # Run evaluation
    asyncio.run(run_evaluation(config))


if __name__ == "__main__":
    main()
