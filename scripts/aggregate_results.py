#!/usr/bin/env python3
"""
Result Aggregation for PeRL Evaluation.

Aggregates evaluation results across seeds and experiments,
computes statistics, and generates comparison reports.

Usage:
    # Aggregate all results in a directory
    python scripts/aggregate_results.py \
        --results_dir results/evaluations \
        --output aggregated_results

    # Filter by methods
    python scripts/aggregate_results.py \
        --results_dir results/evaluations \
        --methods "lora,dora,pissa" \
        --output aggregated_results

    # Generate markdown report
    python scripts/aggregate_results.py \
        --results_dir results/evaluations \
        --format markdown \
        --output aggregated_results

    # Include confidence intervals
    python scripts/aggregate_results.py \
        --results_dir results/evaluations \
        --confidence 0.95 \
        --output aggregated_results
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional scipy for statistical tests
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_stats = None
    SCIPY_AVAILABLE = False

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("perl.aggregate_results")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResultEntry:
    """Single evaluation result entry."""
    experiment_name: str
    peft_method: str
    rank: int
    seed: int
    step: int
    benchmark: str
    accuracy: float
    pass_at_k: float
    format_score: float
    avg_reasoning_steps: float
    avg_generation_time: float
    total_problems: int
    correct_count: int
    result_path: str


@dataclass
class AggregatedStats:
    """Aggregated statistics for a method-benchmark pair."""
    peft_method: str
    benchmark: str
    seeds: List[int]
    accuracies: List[float]
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    ci_lower: float  # Confidence interval lower
    ci_upper: float  # Confidence interval upper
    n: int

    def to_dict(self) -> dict:
        return {
            "peft_method": self.peft_method,
            "benchmark": self.benchmark,
            "seeds": self.seeds,
            "accuracies": self.accuracies,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "median": self.median,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n": self.n,
        }


@dataclass
class ComparisonReport:
    """Complete comparison report."""
    generated_at: str
    total_results: int
    methods: List[str]
    benchmarks: List[str]
    results_by_method: Dict[str, Dict[str, AggregatedStats]]
    best_method_by_benchmark: Dict[str, str]
    ranking_by_benchmark: Dict[str, List[Tuple[str, float]]]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_results": self.total_results,
            "methods": self.methods,
            "benchmarks": self.benchmarks,
            "results_by_method": {
                m: {b: s.to_dict() for b, s in benchmarks.items()}
                for m, benchmarks in self.results_by_method.items()
            },
            "best_method_by_benchmark": self.best_method_by_benchmark,
            "ranking_by_benchmark": self.ranking_by_benchmark,
        }


# =============================================================================
# Result Loading
# =============================================================================

def load_all_results(results_dir: str) -> List[ResultEntry]:
    """
    Load all evaluation results from a directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of ResultEntry objects
    """
    from perl.evaluation import load_results

    entries = []
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []

    logger.info(f"Scanning for results in: {results_dir}")

    for json_file in results_path.rglob("*.json"):
        # Skip status and aggregated files
        if json_file.name in ["evaluation_status.json", "aggregated_results.json"]:
            continue
        if "aggregated" in json_file.name:
            continue

        try:
            result = load_results(json_file)
            m = result.metadata
            s = result.statistics

            entry = ResultEntry(
                experiment_name=Path(m.checkpoint).parent.name if m.checkpoint else "unknown",
                peft_method=m.peft_method,
                rank=m.rank,
                seed=m.seed or 0,
                step=m.checkpoint_step or 0,
                benchmark=m.benchmark,
                accuracy=s.accuracy,
                pass_at_k=s.accuracy,  # Using same value for now
                format_score=getattr(s, 'format_score_mean', 0.0) if hasattr(s, 'format_score_mean') else 0.0,
                avg_reasoning_steps=s.avg_reasoning_steps,
                avg_generation_time=s.avg_generation_time,
                total_problems=s.total_count,
                correct_count=s.correct_count,
                result_path=str(json_file),
            )
            entries.append(entry)

        except Exception as e:
            logger.debug(f"Failed to load {json_file}: {e}")

    logger.info(f"Loaded {len(entries)} result entries")

    return entries


def filter_results(
    entries: List[ResultEntry],
    methods: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    steps: Optional[List[int]] = None,
) -> List[ResultEntry]:
    """Filter results by various criteria."""
    filtered = entries

    if methods:
        methods_lower = [m.lower() for m in methods]
        filtered = [e for e in filtered if e.peft_method.lower() in methods_lower]

    if benchmarks:
        benchmarks_lower = [b.lower() for b in benchmarks]
        filtered = [e for e in filtered if e.benchmark.lower() in benchmarks_lower]

    if seeds:
        filtered = [e for e in filtered if e.seed in seeds]

    if steps:
        filtered = [e for e in filtered if e.step in steps]

    return filtered


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_by_method_benchmark(
    entries: List[ResultEntry],
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, AggregatedStats]]:
    """
    Aggregate results by PEFT method and benchmark.

    Args:
        entries: List of result entries
        confidence_level: Confidence level for CI

    Returns:
        Nested dict: method -> benchmark -> AggregatedStats
    """
    # Group by method and benchmark
    groups = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        groups[entry.peft_method][entry.benchmark].append(entry)

    # Compute statistics for each group
    results = {}
    _scipy_fallback_warned = False

    for method, benchmarks in groups.items():
        results[method] = {}

        for benchmark, group_entries in benchmarks.items():
            accuracies = [e.accuracy for e in group_entries]
            seeds = [e.seed for e in group_entries]

            # Basic stats
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
            median_acc = np.median(accuracies)

            # Confidence interval
            if len(accuracies) >= 2 and SCIPY_AVAILABLE:
                ci = scipy_stats.t.interval(
                    confidence_level,
                    len(accuracies) - 1,
                    loc=mean_acc,
                    scale=scipy_stats.sem(accuracies)
                )
                ci_lower, ci_upper = ci
            else:
                # Fallback: use mean ± 2*std as approximate 95% CI
                if not SCIPY_AVAILABLE and len(accuracies) >= 2 and not _scipy_fallback_warned:
                    logger.warning(
                        "*** USING FALLBACK: scipy not available, "
                        "using mean ± 2*std for confidence intervals instead of t-distribution ***"
                    )
                    _scipy_fallback_warned = True
                ci_lower = mean_acc - 2 * std_acc
                ci_upper = mean_acc + 2 * std_acc

            stats = AggregatedStats(
                peft_method=method,
                benchmark=benchmark,
                seeds=sorted(seeds),
                accuracies=accuracies,
                mean=float(mean_acc),
                std=float(std_acc),
                min_val=float(np.min(accuracies)),
                max_val=float(np.max(accuracies)),
                median=float(median_acc),
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
                n=len(accuracies),
            )
            results[method][benchmark] = stats

    return results


def find_best_methods(
    results_by_method: Dict[str, Dict[str, AggregatedStats]]
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, float]]]]:
    """
    Find best method for each benchmark and create rankings.

    Returns:
        Tuple of (best_method_by_benchmark, ranking_by_benchmark)
    """
    best_by_benchmark = {}
    ranking_by_benchmark = {}

    # Get all benchmarks
    all_benchmarks = set()
    for method_results in results_by_method.values():
        all_benchmarks.update(method_results.keys())

    for benchmark in all_benchmarks:
        # Collect (method, accuracy) pairs
        method_scores = []
        for method, benchmarks in results_by_method.items():
            if benchmark in benchmarks:
                method_scores.append((method, benchmarks[benchmark].mean))

        # Sort by accuracy descending
        method_scores.sort(key=lambda x: x[1], reverse=True)

        if method_scores:
            best_by_benchmark[benchmark] = method_scores[0][0]
            ranking_by_benchmark[benchmark] = method_scores

    return best_by_benchmark, ranking_by_benchmark


def create_comparison_report(
    entries: List[ResultEntry],
    confidence_level: float = 0.95,
) -> ComparisonReport:
    """Create a complete comparison report."""
    results_by_method = aggregate_by_method_benchmark(entries, confidence_level)
    best_methods, rankings = find_best_methods(results_by_method)

    methods = sorted(results_by_method.keys())
    benchmarks = sorted(set(
        b for m_results in results_by_method.values()
        for b in m_results.keys()
    ))

    return ComparisonReport(
        generated_at=datetime.now().isoformat(),
        total_results=len(entries),
        methods=methods,
        benchmarks=benchmarks,
        results_by_method=results_by_method,
        best_method_by_benchmark=best_methods,
        ranking_by_benchmark=rankings,
    )


# =============================================================================
# Output Formatting
# =============================================================================

def format_accuracy(mean: float, std: float, n: int) -> str:
    """Format accuracy as mean ± std."""
    if n > 1 and std > 0:
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.3f}"


def print_comparison_table(report: ComparisonReport):
    """Print comparison table to console."""
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    print(f"Generated: {report.generated_at}")
    print(f"Total results: {report.total_results}")
    print(f"Methods: {len(report.methods)}")
    print(f"Benchmarks: {len(report.benchmarks)}")

    # Build header
    header = f"{'Method':<12}"
    for b in report.benchmarks:
        header += f" | {b[:15]:<17}"
    print("\n" + header)
    print("-" * len(header))

    # Print rows
    for method in sorted(report.methods):
        row = f"{method:<12}"
        for benchmark in report.benchmarks:
            if benchmark in report.results_by_method.get(method, {}):
                stats = report.results_by_method[method][benchmark]
                val = format_accuracy(stats.mean, stats.std, stats.n)
                # Highlight best
                if report.best_method_by_benchmark.get(benchmark) == method:
                    val = f"*{val}*"
                row += f" | {val:<17}"
            else:
                row += f" | {'-':<17}"
        print(row)

    print("\n* = Best method for benchmark")

    # Print rankings
    print("\n" + "-" * 80)
    print("RANKINGS BY BENCHMARK")
    print("-" * 80)

    for benchmark in report.benchmarks:
        if benchmark in report.ranking_by_benchmark:
            ranking = report.ranking_by_benchmark[benchmark]
            print(f"\n{benchmark}:")
            for i, (method, acc) in enumerate(ranking, 1):
                print(f"  {i}. {method:<12} {acc:.4f}")


def generate_markdown_report(report: ComparisonReport) -> str:
    """Generate markdown report."""
    lines = [
        "# PeRL Evaluation Results",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Summary",
        "",
        f"- Total evaluations: {report.total_results}",
        f"- PEFT methods: {', '.join(report.methods)}",
        f"- Benchmarks: {', '.join(report.benchmarks)}",
        "",
        "## Results Table",
        "",
    ]

    # Table header
    header = "| Method |"
    separator = "|--------|"
    for b in report.benchmarks:
        header += f" {b} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    # Table rows
    for method in sorted(report.methods):
        row = f"| {method} |"
        for benchmark in report.benchmarks:
            if benchmark in report.results_by_method.get(method, {}):
                stats = report.results_by_method[method][benchmark]
                val = format_accuracy(stats.mean, stats.std, stats.n)
                if report.best_method_by_benchmark.get(benchmark) == method:
                    val = f"**{val}**"
                row += f" {val} |"
            else:
                row += " - |"
        lines.append(row)

    lines.append("")
    lines.append("**Bold** = Best method for benchmark")
    lines.append("")

    # Rankings
    lines.append("## Rankings")
    lines.append("")

    for benchmark in report.benchmarks:
        lines.append(f"### {benchmark}")
        lines.append("")
        if benchmark in report.ranking_by_benchmark:
            for i, (method, acc) in enumerate(report.ranking_by_benchmark[benchmark], 1):
                lines.append(f"{i}. **{method}**: {acc:.4f}")
        lines.append("")

    # Detailed statistics
    lines.append("## Detailed Statistics")
    lines.append("")

    for method in sorted(report.methods):
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Benchmark | Mean | Std | Min | Max | 95% CI | N |")
        lines.append("|-----------|------|-----|-----|-----|--------|---|")

        for benchmark in report.benchmarks:
            if benchmark in report.results_by_method.get(method, {}):
                s = report.results_by_method[method][benchmark]
                lines.append(
                    f"| {benchmark} | {s.mean:.4f} | {s.std:.4f} | "
                    f"{s.min_val:.4f} | {s.max_val:.4f} | "
                    f"[{s.ci_lower:.4f}, {s.ci_upper:.4f}] | {s.n} |"
                )
        lines.append("")

    return "\n".join(lines)


def generate_csv_report(report: ComparisonReport) -> str:
    """Generate CSV report."""
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "method", "benchmark", "mean", "std", "min", "max",
        "ci_lower", "ci_upper", "n", "seeds"
    ])

    # Data rows
    for method in sorted(report.methods):
        for benchmark in report.benchmarks:
            if benchmark in report.results_by_method.get(method, {}):
                s = report.results_by_method[method][benchmark]
                writer.writerow([
                    method, benchmark, f"{s.mean:.6f}", f"{s.std:.6f}",
                    f"{s.min_val:.6f}", f"{s.max_val:.6f}",
                    f"{s.ci_lower:.6f}", f"{s.ci_upper:.6f}",
                    s.n, ",".join(map(str, s.seeds))
                ])

    return output.getvalue()


def save_report(
    report: ComparisonReport,
    output_dir: str,
    formats: List[str] = None,
):
    """Save report in multiple formats."""
    if formats is None:
        formats = ["json", "markdown", "csv"]

    os.makedirs(output_dir, exist_ok=True)

    if "json" in formats:
        path = os.path.join(output_dir, "comparison_report.json")
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved JSON report: {path}")

    if "markdown" in formats:
        path = os.path.join(output_dir, "COMPARISON_REPORT.md")
        with open(path, "w") as f:
            f.write(generate_markdown_report(report))
        logger.info(f"Saved Markdown report: {path}")

    if "csv" in formats:
        path = os.path.join(output_dir, "comparison_report.csv")
        with open(path, "w") as f:
            f.write(generate_csv_report(report))
        logger.info(f"Saved CSV report: {path}")


# =============================================================================
# Additional Analysis
# =============================================================================

def analyze_by_difficulty(entries: List[ResultEntry]) -> Dict:
    """Analyze results by problem difficulty (if available)."""
    # This would require loading full result files
    # Placeholder for future implementation
    return {}


def analyze_by_step(entries: List[ResultEntry]) -> Dict:
    """Analyze how accuracy changes with training steps."""
    by_method = defaultdict(list)

    for entry in entries:
        by_method[entry.peft_method].append({
            "step": entry.step,
            "accuracy": entry.accuracy,
            "benchmark": entry.benchmark,
            "seed": entry.seed,
        })

    return dict(by_method)


def compute_statistical_tests(
    results_by_method: Dict[str, Dict[str, AggregatedStats]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise statistical tests between methods.

    Returns p-values for each method pair on each benchmark.
    """
    if not SCIPY_AVAILABLE:
        logger.warning(
            "*** USING FALLBACK: scipy not available, "
            "SKIPPING STATISTICAL TESTS ***"
        )
        return {}

    pvalues = {}
    methods = list(results_by_method.keys())

    for benchmark in set(
        b for m_results in results_by_method.values()
        for b in m_results.keys()
    ):
        pvalues[benchmark] = {}

        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1:]:
                if (benchmark in results_by_method.get(method1, {}) and
                        benchmark in results_by_method.get(method2, {})):

                    accs1 = results_by_method[method1][benchmark].accuracies
                    accs2 = results_by_method[method2][benchmark].accuracies

                    if len(accs1) >= 2 and len(accs2) >= 2:
                        try:
                            _, pvalue = scipy_stats.ttest_ind(accs1, accs2)
                            pvalues[benchmark][f"{method1}_vs_{method2}"] = float(pvalue)
                        except Exception:
                            pass

    return pvalues


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate PeRL evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing evaluation results",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="aggregated_results",
        help="Output directory for reports",
    )

    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Filter by PEFT methods (comma-separated)",
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Filter by benchmarks (comma-separated)",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["all", "json", "markdown", "csv"],
        help="Output format",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )

    parser.add_argument(
        "--statistical_tests",
        action="store_true",
        help="Include pairwise statistical tests",
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

    # Load results
    entries = load_all_results(args.results_dir)

    if not entries:
        logger.error("No results found!")
        sys.exit(1)

    # Apply filters
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    benchmarks = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None

    if methods or benchmarks:
        entries = filter_results(entries, methods=methods, benchmarks=benchmarks)
        logger.info(f"After filtering: {len(entries)} results")

    # Create comparison report
    report = create_comparison_report(entries, confidence_level=args.confidence)

    # Print to console
    print_comparison_table(report)

    # Save reports
    formats = ["json", "markdown", "csv"] if args.format == "all" else [args.format]
    save_report(report, args.output, formats=formats)

    # Statistical tests if requested
    if args.statistical_tests:
        pvalues = compute_statistical_tests(report.results_by_method)
        pvalues_path = os.path.join(args.output, "statistical_tests.json")
        with open(pvalues_path, "w") as f:
            json.dump(pvalues, f, indent=2)
        logger.info(f"Saved statistical tests: {pvalues_path}")

        print("\nStatistical Tests (p-values):")
        for benchmark, tests in pvalues.items():
            print(f"\n{benchmark}:")
            for test_name, pvalue in tests.items():
                sig = "*" if pvalue < 0.05 else ""
                print(f"  {test_name}: {pvalue:.4f}{sig}")

    print(f"\nReports saved to: {args.output}")


if __name__ == "__main__":
    main()
