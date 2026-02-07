#!/usr/bin/env python3
"""
Long-Chain Problem Filter for PeRL.

Filters problems by reasoning chain length from evaluation results.
Creates filtered benchmarks for targeted evaluation.

Usage:
    # Extract long-chain problems from results
    python scripts/filter_long_chain.py \
        --results_file results/evaluations/lora_s42/math500.json \
        --min_steps 9 \
        --output_file benchmarks/math500_long_chain.json

    # Filter from multiple results
    python scripts/filter_long_chain.py \
        --results_dir results/evaluations \
        --min_steps 9 \
        --output_dir benchmarks/long_chain

    # Different step thresholds
    python scripts/filter_long_chain.py \
        --results_dir results/evaluations \
        --min_steps 5 \
        --max_steps 8 \
        --output_dir benchmarks/medium_chain

    # Analyze chain length distribution
    python scripts/filter_long_chain.py \
        --results_dir results/evaluations \
        --analyze
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("perl.filter_long_chain")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FilteredProblem:
    """A filtered problem with chain length info."""
    problem_id: str
    question: str
    answer: str
    reasoning_steps: int
    reasoning_tokens: int
    difficulty: Optional[str]
    category: Optional[str]
    benchmark: str
    # Performance across methods
    method_results: Dict[str, bool]  # method -> correct


@dataclass
class ChainLengthStats:
    """Statistics about chain lengths."""
    total_problems: int
    min_steps: int
    max_steps: int
    mean_steps: float
    median_steps: float
    std_steps: float
    distribution: Dict[int, int]  # steps -> count
    bucket_counts: Dict[str, int]  # short/medium/long -> count


# =============================================================================
# Chain Length Analysis
# =============================================================================

def analyze_chain_lengths(results_dir: str) -> Tuple[ChainLengthStats, List[Dict]]:
    """
    Analyze chain length distribution across all results.

    Returns:
        Tuple of (stats, all_problems)
    """
    from perl.evaluation import load_results

    all_problems = []
    step_counts = []
    distribution = defaultdict(int)

    results_path = Path(results_dir)

    for json_file in results_path.rglob("*.json"):
        if json_file.name in ["evaluation_status.json", "aggregated_results.json"]:
            continue
        if "ood" in json_file.name or "comparison" in json_file.name:
            continue

        try:
            result = load_results(json_file)

            for prob in result.results:
                steps = prob.reasoning_steps
                step_counts.append(steps)
                distribution[steps] += 1

                all_problems.append({
                    "problem_id": prob.problem_id,
                    "question": prob.question,
                    "answer": prob.ground_truth,
                    "reasoning_steps": steps,
                    "reasoning_tokens": prob.reasoning_tokens,
                    "difficulty": prob.difficulty,
                    "category": prob.category,
                    "correct": prob.correct,
                    "peft_method": result.metadata.peft_method,
                    "benchmark": result.metadata.benchmark,
                    "seed": result.metadata.seed,
                })

        except Exception as e:
            logger.debug(f"Failed to load {json_file}: {e}")

    if not step_counts:
        return ChainLengthStats(0, 0, 0, 0, 0, 0, {}, {}), []

    # Compute bucket counts
    bucket_counts = {"short": 0, "medium": 0, "long": 0}
    for steps in step_counts:
        if steps <= 3:
            bucket_counts["short"] += 1
        elif steps <= 8:
            bucket_counts["medium"] += 1
        else:
            bucket_counts["long"] += 1

    stats = ChainLengthStats(
        total_problems=len(step_counts),
        min_steps=min(step_counts),
        max_steps=max(step_counts),
        mean_steps=float(np.mean(step_counts)),
        median_steps=float(np.median(step_counts)),
        std_steps=float(np.std(step_counts)),
        distribution=dict(distribution),
        bucket_counts=bucket_counts,
    )

    return stats, all_problems


def print_chain_length_analysis(stats: ChainLengthStats):
    """Print chain length analysis to console."""
    print("\n" + "=" * 60)
    print("CHAIN LENGTH DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print(f"\nTotal problems: {stats.total_problems:,}")
    print(f"Min steps:      {stats.min_steps}")
    print(f"Max steps:      {stats.max_steps}")
    print(f"Mean steps:     {stats.mean_steps:.2f}")
    print(f"Median steps:   {stats.median_steps:.1f}")
    print(f"Std dev:        {stats.std_steps:.2f}")

    print("\nBucket Distribution:")
    print(f"  Short (1-3 steps):   {stats.bucket_counts['short']:>6,} "
          f"({100*stats.bucket_counts['short']/stats.total_problems:.1f}%)")
    print(f"  Medium (4-8 steps):  {stats.bucket_counts['medium']:>6,} "
          f"({100*stats.bucket_counts['medium']/stats.total_problems:.1f}%)")
    print(f"  Long (9+ steps):     {stats.bucket_counts['long']:>6,} "
          f"({100*stats.bucket_counts['long']/stats.total_problems:.1f}%)")

    print("\nDetailed Distribution:")
    for steps in sorted(stats.distribution.keys()):
        count = stats.distribution[steps]
        bar = "█" * min(50, count // 10)
        print(f"  {steps:>3} steps: {count:>5} {bar}")

    print("=" * 60)


# =============================================================================
# Filtering Functions
# =============================================================================

def filter_by_chain_length(
    all_problems: List[Dict],
    min_steps: int = 9,
    max_steps: Optional[int] = None,
) -> List[Dict]:
    """
    Filter problems by chain length.

    Args:
        all_problems: List of problem dictionaries
        min_steps: Minimum reasoning steps required
        max_steps: Maximum reasoning steps (optional)

    Returns:
        Filtered list of problems
    """
    filtered = []

    for prob in all_problems:
        steps = prob.get("reasoning_steps", 0)

        if steps >= min_steps:
            if max_steps is None or steps <= max_steps:
                filtered.append(prob)

    return filtered


def deduplicate_problems(problems: List[Dict]) -> List[FilteredProblem]:
    """
    Deduplicate problems across methods/seeds and aggregate results.

    Returns list of unique problems with method performance.
    """
    # Group by (problem_id, benchmark)
    grouped = defaultdict(lambda: {
        "question": None,
        "answer": None,
        "reasoning_steps": 0,
        "reasoning_tokens": 0,
        "difficulty": None,
        "category": None,
        "benchmark": None,
        "method_results": defaultdict(list),
    })

    for prob in problems:
        key = (prob["problem_id"], prob["benchmark"])
        entry = grouped[key]

        # Store problem info (take from first occurrence)
        if entry["question"] is None:
            entry["question"] = prob["question"]
            entry["answer"] = prob["answer"]
            entry["difficulty"] = prob.get("difficulty")
            entry["category"] = prob.get("category")
            entry["benchmark"] = prob["benchmark"]

        # Take max steps (they should be same across methods)
        entry["reasoning_steps"] = max(entry["reasoning_steps"], prob["reasoning_steps"])
        entry["reasoning_tokens"] = max(entry["reasoning_tokens"], prob["reasoning_tokens"])

        # Track correctness by method
        method = prob["peft_method"]
        entry["method_results"][method].append(prob["correct"])

    # Convert to FilteredProblem objects
    results = []
    for (prob_id, benchmark), entry in grouped.items():
        # Aggregate method results (majority vote across seeds)
        method_correct = {}
        for method, correctness_list in entry["method_results"].items():
            # Use majority vote
            method_correct[method] = sum(correctness_list) > len(correctness_list) / 2

        results.append(FilteredProblem(
            problem_id=prob_id,
            question=entry["question"],
            answer=entry["answer"],
            reasoning_steps=entry["reasoning_steps"],
            reasoning_tokens=entry["reasoning_tokens"],
            difficulty=entry["difficulty"],
            category=entry["category"],
            benchmark=entry["benchmark"],
            method_results=method_correct,
        ))

    return results


def create_filtered_benchmark(
    problems: List[FilteredProblem],
    benchmark_name: str,
    description: str,
) -> Dict:
    """
    Create a filtered benchmark file.

    Returns a dictionary suitable for saving as JSON.
    """
    return {
        "name": benchmark_name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "total_problems": len(problems),
        "min_steps": min(p.reasoning_steps for p in problems) if problems else 0,
        "max_steps": max(p.reasoning_steps for p in problems) if problems else 0,
        "problems": [
            {
                "id": p.problem_id,
                "question": p.question,
                "answer": p.answer,
                "reasoning_steps": p.reasoning_steps,
                "difficulty": p.difficulty,
                "category": p.category,
                "source_benchmark": p.benchmark,
            }
            for p in problems
        ],
    }


def save_filtered_benchmark(benchmark: Dict, output_path: str):
    """Save filtered benchmark to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    logger.info(f"Saved filtered benchmark: {output_path}")
    logger.info(f"  Problems: {benchmark['total_problems']}")
    logger.info(f"  Steps: {benchmark['min_steps']}-{benchmark['max_steps']}")


# =============================================================================
# Method Performance Analysis
# =============================================================================

def analyze_method_performance_by_length(
    all_problems: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Analyze method performance by chain length bucket.

    Returns:
        Dict: method -> {"short": acc, "medium": acc, "long": acc}
    """
    # Group by method and bucket
    grouped = defaultdict(lambda: defaultdict(list))

    for prob in all_problems:
        method = prob["peft_method"]
        steps = prob["reasoning_steps"]

        if steps <= 3:
            bucket = "short"
        elif steps <= 8:
            bucket = "medium"
        else:
            bucket = "long"

        grouped[method][bucket].append(prob["correct"])

    # Compute accuracies
    results = {}
    for method, buckets in grouped.items():
        results[method] = {}
        for bucket, correctness in buckets.items():
            if correctness:
                results[method][bucket] = sum(correctness) / len(correctness)
            else:
                results[method][bucket] = 0.0

    return results


def print_method_performance(performance: Dict[str, Dict[str, float]]):
    """Print method performance by chain length."""
    print("\n" + "-" * 60)
    print("METHOD PERFORMANCE BY CHAIN LENGTH")
    print("-" * 60)

    print(f"\n{'Method':<12} | {'Short':>8} | {'Medium':>8} | {'Long':>8} | {'Δ Long-Short':>12}")
    print("-" * 60)

    for method in sorted(performance.keys()):
        perf = performance[method]
        short = perf.get("short", 0)
        medium = perf.get("medium", 0)
        long = perf.get("long", 0)
        delta = long - short

        print(f"{method:<12} | {short:>7.1%} | {medium:>7.1%} | {long:>7.1%} | {delta:>+11.1%}")

    print("-" * 60)


# =============================================================================
# Single File Processing
# =============================================================================

def filter_single_file(
    results_file: str,
    min_steps: int = 9,
    max_steps: Optional[int] = None,
) -> List[Dict]:
    """Filter problems from a single results file."""
    from perl.evaluation import load_results

    result = load_results(results_file)

    filtered = []
    for prob in result.results:
        steps = prob.reasoning_steps

        if steps >= min_steps:
            if max_steps is None or steps <= max_steps:
                filtered.append({
                    "id": prob.problem_id,
                    "question": prob.question,
                    "answer": prob.ground_truth,
                    "reasoning_steps": steps,
                    "reasoning_tokens": prob.reasoning_tokens,
                    "difficulty": prob.difficulty,
                    "category": prob.category,
                    "correct": prob.correct,
                    "model_answer": prob.model_answer,
                })

    return filtered


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter problems by reasoning chain length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results_file",
        type=str,
        help="Single results JSON file to filter",
    )
    input_group.add_argument(
        "--results_dir",
        type=str,
        help="Directory containing results to analyze",
    )

    # Chain length options
    parser.add_argument(
        "--min_steps",
        type=int,
        default=9,
        help="Minimum reasoning steps (default: 9)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum reasoning steps (optional)",
    )

    # Output options
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for filtered benchmark (single file mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for filtered benchmarks (directory mode)",
    )

    # Analysis options
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze chain length distribution (no filtering)",
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

    # Single file mode
    if args.results_file:
        logger.info(f"Processing single file: {args.results_file}")

        filtered = filter_single_file(
            args.results_file,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
        )

        print(f"\nFiltered {len(filtered)} problems with {args.min_steps}+ steps")

        if args.output_file:
            benchmark = {
                "name": f"filtered_min{args.min_steps}",
                "description": f"Problems requiring {args.min_steps}+ reasoning steps",
                "source": args.results_file,
                "created_at": datetime.now().isoformat(),
                "total_problems": len(filtered),
                "problems": filtered,
            }
            save_filtered_benchmark(benchmark, args.output_file)
        else:
            # Print sample
            print("\nSample problems:")
            for p in filtered[:5]:
                print(f"  - {p['id']}: {p['reasoning_steps']} steps, "
                      f"correct={p['correct']}")

        return

    # Directory mode
    if args.results_dir:
        # Analyze chain length distribution
        stats, all_problems = analyze_chain_lengths(args.results_dir)

        if stats.total_problems == 0:
            logger.error("No problems found!")
            sys.exit(1)

        # Print analysis
        print_chain_length_analysis(stats)

        # Print method performance
        performance = analyze_method_performance_by_length(all_problems)
        print_method_performance(performance)

        if args.analyze:
            return  # Analysis only, no filtering

        # Filter problems
        filtered = filter_by_chain_length(
            all_problems,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
        )

        print(f"\nFiltered {len(filtered)} problems with "
              f"{args.min_steps}+ steps")

        # Deduplicate
        unique = deduplicate_problems(filtered)
        print(f"Unique problems: {len(unique)}")

        # Save if output specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

            # Group by source benchmark
            by_benchmark = defaultdict(list)
            for p in unique:
                by_benchmark[p.benchmark].append(p)

            for benchmark, problems in by_benchmark.items():
                benchmark_name = f"{benchmark}_long_chain"
                description = (f"Long-chain problems ({args.min_steps}+ steps) "
                               f"from {benchmark}")

                benchmark_data = create_filtered_benchmark(
                    problems, benchmark_name, description
                )

                output_file = os.path.join(
                    args.output_dir,
                    f"{benchmark}_min{args.min_steps}.json"
                )
                save_filtered_benchmark(benchmark_data, output_file)

            # Also save combined
            combined = create_filtered_benchmark(
                unique,
                f"combined_long_chain_min{args.min_steps}",
                f"All long-chain problems ({args.min_steps}+ steps)",
            )
            combined_path = os.path.join(
                args.output_dir,
                f"combined_min{args.min_steps}.json"
            )
            save_filtered_benchmark(combined, combined_path)

            print(f"\nSaved filtered benchmarks to: {args.output_dir}")


if __name__ == "__main__":
    main()
