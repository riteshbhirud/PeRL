#!/usr/bin/env python3
"""
Cross-Seed Result Aggregator for PeRL.

Aggregates evaluation results across multiple seeds and PEFT methods
to compute mean ± std statistics and generate comparison tables.

Usage:
    # Aggregate results from multiple experiments
    python scripts/aggregate_eval_results.py \
        --results output/*/eval_results/ \
        --output aggregated_results/

    # Filter by PEFT method
    python scripts/aggregate_eval_results.py \
        --results output/*/eval_results/ \
        --filter-method lora,dora

    # Generate markdown report
    python scripts/aggregate_eval_results.py \
        --results output/*/eval_results/ \
        --format markdown

    # Dry run (preview without writing)
    python scripts/aggregate_eval_results.py \
        --results output/*/eval_results/ \
        --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResultEntry:
    """Single evaluation result entry."""
    experiment_name: str
    peft_method: str
    seed: int
    step: int
    dataset: str
    accuracy: float
    pass_at_k: float
    format_score: float
    num_problems: int


@dataclass
class AggregatedResult:
    """Aggregated result across seeds."""
    peft_method: str
    step: int
    dataset: str
    accuracy_mean: float
    accuracy_std: float
    pass_at_k_mean: float
    pass_at_k_std: float
    format_score_mean: float
    format_score_std: float
    num_seeds: int
    seeds: List[int]


@dataclass
class MethodComparison:
    """Comparison of methods at a specific step."""
    step: int
    dataset: str
    methods: Dict[str, AggregatedResult]


# =============================================================================
# Parsing Utilities
# =============================================================================

def parse_experiment_name(name: str) -> Tuple[str, int]:
    """
    Parse PEFT method and seed from experiment name.

    Expected formats:
    - lora_r16_s42 -> ("lora", 42)
    - dora_r32_s123 -> ("dora", 123)
    - pissa_s42 -> ("pissa", 42)
    """
    # Try to extract seed (s followed by digits at end)
    seed_match = re.search(r'_s(\d+)$', name)
    seed = int(seed_match.group(1)) if seed_match else 0

    # Extract method (everything before _r or _s)
    method_match = re.match(r'^([a-z_]+?)(?:_r\d+)?(?:_s\d+)?$', name, re.IGNORECASE)
    method = method_match.group(1) if method_match else name.split("_")[0]

    return method, seed


def find_result_files(patterns: List[str]) -> List[Path]:
    """Find all result summary.json files matching patterns."""
    result_files = []

    for pattern in patterns:
        if "*" in pattern:
            matches = glob.glob(pattern, recursive=True)
        else:
            matches = [pattern]

        for match in matches:
            path = Path(match)
            if path.is_dir():
                # Look for summary.json files in step_*/*/
                for summary in path.glob("step_*/*/summary.json"):
                    result_files.append(summary)
            elif path.name == "summary.json":
                result_files.append(path)

    return result_files


def load_results(result_files: List[Path]) -> List[ResultEntry]:
    """Load all result entries from summary files."""
    entries = []

    for summary_path in result_files:
        try:
            with open(summary_path) as f:
                data = json.load(f)

            # Extract experiment name from path
            # Expected: .../experiment_name/eval_results/step_N/dataset/summary.json
            parts = summary_path.parts
            experiment_name = None

            # Find the eval_results directory and get parent
            for i, part in enumerate(parts):
                if part == "eval_results" and i > 0:
                    experiment_name = parts[i - 1]
                    break

            if not experiment_name:
                # Try getting from checkpoint path in data
                checkpoint = data.get("checkpoint", "")
                cp_parts = Path(checkpoint).parts
                for i, part in enumerate(cp_parts):
                    if part.startswith("checkpoint-"):
                        experiment_name = cp_parts[i - 1] if i > 0 else "unknown"
                        break

            if not experiment_name:
                experiment_name = "unknown"

            peft_method, seed = parse_experiment_name(experiment_name)

            entry = ResultEntry(
                experiment_name=experiment_name,
                peft_method=peft_method,
                seed=seed,
                step=data.get("step", 0),
                dataset=data.get("dataset", "unknown"),
                accuracy=data.get("accuracy", 0.0),
                pass_at_k=data.get("pass_at_k", 0.0),
                format_score=data.get("format_score", 0.0),
                num_problems=data.get("num_problems", 0),
            )
            entries.append(entry)

        except Exception as e:
            print(f"Warning: Failed to load {summary_path}: {e}")

    return entries


# =============================================================================
# Aggregation Logic
# =============================================================================

def aggregate_results(
    entries: List[ResultEntry],
    filter_methods: Optional[List[str]] = None,
    filter_steps: Optional[List[int]] = None,
    filter_datasets: Optional[List[str]] = None,
) -> List[AggregatedResult]:
    """Aggregate results across seeds."""

    # Group by (method, step, dataset)
    groups = defaultdict(list)

    for entry in entries:
        # Apply filters
        if filter_methods and entry.peft_method not in filter_methods:
            continue
        if filter_steps and entry.step not in filter_steps:
            continue
        if filter_datasets and entry.dataset not in filter_datasets:
            continue

        key = (entry.peft_method, entry.step, entry.dataset)
        groups[key].append(entry)

    # Compute statistics for each group
    aggregated = []

    for (method, step, dataset), group_entries in groups.items():
        accuracies = [e.accuracy for e in group_entries]
        pass_at_ks = [e.pass_at_k for e in group_entries]
        format_scores = [e.format_score for e in group_entries]
        seeds = [e.seed for e in group_entries]

        agg = AggregatedResult(
            peft_method=method,
            step=step,
            dataset=dataset,
            accuracy_mean=np.mean(accuracies),
            accuracy_std=np.std(accuracies) if len(accuracies) > 1 else 0.0,
            pass_at_k_mean=np.mean(pass_at_ks),
            pass_at_k_std=np.std(pass_at_ks) if len(pass_at_ks) > 1 else 0.0,
            format_score_mean=np.mean(format_scores),
            format_score_std=np.std(format_scores) if len(format_scores) > 1 else 0.0,
            num_seeds=len(seeds),
            seeds=sorted(seeds),
        )
        aggregated.append(agg)

    return aggregated


def create_comparison_tables(
    aggregated: List[AggregatedResult],
) -> Dict[Tuple[int, str], MethodComparison]:
    """Create comparison tables indexed by (step, dataset)."""
    comparisons = {}

    for agg in aggregated:
        key = (agg.step, agg.dataset)
        if key not in comparisons:
            comparisons[key] = MethodComparison(
                step=agg.step,
                dataset=agg.dataset,
                methods={},
            )
        comparisons[key].methods[agg.peft_method] = agg

    return comparisons


# =============================================================================
# Output Formatting
# =============================================================================

def format_value_with_std(mean: float, std: float, precision: int = 4) -> str:
    """Format value as mean ± std."""
    if std > 0:
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    return f"{mean:.{precision}f}"


def generate_markdown_report(
    aggregated: List[AggregatedResult],
    comparisons: Dict[Tuple[int, str], MethodComparison],
    output_path: Path,
):
    """Generate markdown report."""

    lines = [
        "# Evaluation Results Aggregation",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Summary statistics
    methods = sorted(set(a.peft_method for a in aggregated))
    datasets = sorted(set(a.dataset for a in aggregated))
    steps = sorted(set(a.step for a in aggregated))

    lines.extend([
        "## Summary",
        "",
        f"- **PEFT Methods**: {', '.join(methods)}",
        f"- **Datasets**: {', '.join(datasets)}",
        f"- **Steps**: {', '.join(map(str, steps))}",
        "",
    ])

    # Per-dataset tables
    for dataset in datasets:
        lines.extend([
            f"## {dataset}",
            "",
        ])

        # Build table header
        header = "| Step | " + " | ".join(methods) + " |"
        separator = "|------|" + "|".join(["------" for _ in methods]) + "|"
        lines.extend([header, separator])

        # Build table rows
        for step in steps:
            key = (step, dataset)
            if key not in comparisons:
                continue

            comp = comparisons[key]
            row = f"| {step} |"

            for method in methods:
                if method in comp.methods:
                    agg = comp.methods[method]
                    val = format_value_with_std(agg.accuracy_mean, agg.accuracy_std)
                    row += f" {val} |"
                else:
                    row += " - |"

            lines.append(row)

        lines.append("")

    # Best methods summary
    lines.extend([
        "## Best Methods by Dataset",
        "",
    ])

    for dataset in datasets:
        best_method = None
        best_accuracy = -1

        for agg in aggregated:
            if agg.dataset == dataset:
                if agg.accuracy_mean > best_accuracy:
                    best_accuracy = agg.accuracy_mean
                    best_method = agg.peft_method

        if best_method:
            lines.append(f"- **{dataset}**: {best_method} ({best_accuracy:.4f})")

    lines.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated markdown report: {output_path}")


def generate_json_output(
    aggregated: List[AggregatedResult],
    output_path: Path,
):
    """Generate JSON output."""
    data = {
        "generated": datetime.now().isoformat(),
        "results": [],
    }

    for agg in aggregated:
        data["results"].append({
            "peft_method": agg.peft_method,
            "step": agg.step,
            "dataset": agg.dataset,
            "accuracy_mean": agg.accuracy_mean,
            "accuracy_std": agg.accuracy_std,
            "pass_at_k_mean": agg.pass_at_k_mean,
            "pass_at_k_std": agg.pass_at_k_std,
            "format_score_mean": agg.format_score_mean,
            "format_score_std": agg.format_score_std,
            "num_seeds": agg.num_seeds,
            "seeds": agg.seeds,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated JSON output: {output_path}")


def generate_csv_output(
    aggregated: List[AggregatedResult],
    output_path: Path,
):
    """Generate CSV output."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "peft_method", "step", "dataset",
            "accuracy_mean", "accuracy_std",
            "pass_at_k_mean", "pass_at_k_std",
            "format_score_mean", "format_score_std",
            "num_seeds",
        ])

        # Data rows
        for agg in aggregated:
            writer.writerow([
                agg.peft_method, agg.step, agg.dataset,
                f"{agg.accuracy_mean:.6f}", f"{agg.accuracy_std:.6f}",
                f"{agg.pass_at_k_mean:.6f}", f"{agg.pass_at_k_std:.6f}",
                f"{agg.format_score_mean:.6f}", f"{agg.format_score_std:.6f}",
                agg.num_seeds,
            ])

    print(f"Generated CSV output: {output_path}")


def print_summary_table(aggregated: List[AggregatedResult]):
    """Print summary table to console."""

    # Group by dataset
    by_dataset = defaultdict(list)
    for agg in aggregated:
        by_dataset[agg.dataset].append(agg)

    for dataset, aggs in sorted(by_dataset.items()):
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}")
        print(f"{'='*70}")

        # Get all methods and steps
        methods = sorted(set(a.peft_method for a in aggs))
        steps = sorted(set(a.step for a in aggs))

        # Print header
        header = f"{'Method':<12}"
        for step in steps:
            header += f" | Step {step:>5}"
        print(header)
        print("-" * len(header))

        # Print rows
        for method in methods:
            row = f"{method:<12}"
            for step in steps:
                matching = [a for a in aggs if a.peft_method == method and a.step == step]
                if matching:
                    agg = matching[0]
                    val = f"{agg.accuracy_mean:.4f}"
                    if agg.accuracy_std > 0:
                        val += f"±{agg.accuracy_std:.3f}"
                    row += f" | {val:>11}"
                else:
                    row += f" | {'-':>11}"
            print(row)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_aggregation(
    result_patterns: List[str],
    output_dir: str,
    output_format: str = "all",
    filter_methods: Optional[List[str]] = None,
    filter_steps: Optional[List[int]] = None,
    filter_datasets: Optional[List[str]] = None,
    dry_run: bool = False,
):
    """Run the aggregation pipeline."""

    # Find result files
    print("Searching for result files...")
    result_files = find_result_files(result_patterns)

    if not result_files:
        print("No result files found!")
        print("Searched patterns:")
        for p in result_patterns:
            print(f"  - {p}")
        return

    print(f"Found {len(result_files)} result files")

    # Load results
    print("Loading results...")
    entries = load_results(result_files)

    if not entries:
        print("No valid result entries loaded!")
        return

    print(f"Loaded {len(entries)} result entries")

    # Print summary of loaded data
    methods = sorted(set(e.peft_method for e in entries))
    datasets = sorted(set(e.dataset for e in entries))
    steps = sorted(set(e.step for e in entries))
    seeds = sorted(set(e.seed for e in entries))

    print(f"\nData summary:")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Steps: {steps}")
    print(f"  Seeds: {seeds}")

    if dry_run:
        print("\n[DRY RUN] Would aggregate the above data")
        print(f"[DRY RUN] Output directory: {output_dir}")
        return

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(
        entries,
        filter_methods=filter_methods,
        filter_steps=filter_steps,
        filter_datasets=filter_datasets,
    )

    if not aggregated:
        print("No results after filtering!")
        return

    print(f"Generated {len(aggregated)} aggregated results")

    # Create comparisons
    comparisons = create_comparison_tables(aggregated)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    if output_format in ("all", "json"):
        generate_json_output(aggregated, output_path / "aggregated.json")

    if output_format in ("all", "csv"):
        generate_csv_output(aggregated, output_path / "aggregated.csv")

    if output_format in ("all", "markdown"):
        generate_markdown_report(aggregated, comparisons, output_path / "REPORT.md")

    # Print summary to console
    print_summary_table(aggregated)

    print(f"\nResults saved to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Result directories or glob patterns",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="aggregated_results",
        help="Output directory (default: aggregated_results)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["all", "json", "csv", "markdown"],
        default="all",
        help="Output format (default: all)",
    )

    parser.add_argument(
        "--filter-method",
        type=str,
        default=None,
        help="Comma-separated list of PEFT methods to include",
    )

    parser.add_argument(
        "--filter-step",
        type=str,
        default=None,
        help="Comma-separated list of steps to include",
    )

    parser.add_argument(
        "--filter-dataset",
        type=str,
        default=None,
        help="Comma-separated list of datasets to include",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing output",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse filters
    filter_methods = None
    if args.filter_method:
        filter_methods = [m.strip() for m in args.filter_method.split(",")]

    filter_steps = None
    if args.filter_step:
        filter_steps = [int(s.strip()) for s in args.filter_step.split(",")]

    filter_datasets = None
    if args.filter_dataset:
        filter_datasets = [d.strip() for d in args.filter_dataset.split(",")]

    run_aggregation(
        result_patterns=args.results,
        output_dir=args.output,
        output_format=args.format,
        filter_methods=filter_methods,
        filter_steps=filter_steps,
        filter_datasets=filter_datasets,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
