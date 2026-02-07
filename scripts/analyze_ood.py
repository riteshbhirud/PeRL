#!/usr/bin/env python3
"""
OOD (Out-of-Distribution) Analysis for PeRL Evaluation Results.

Analyzes model performance across different dimensions:
- Difficulty levels (Easy/Medium/Hard)
- Problem categories (Algebra, Geometry, Number Theory, etc.)
- Reasoning chain length (Short/Medium/Long)

Usage:
    # Full OOD analysis
    python scripts/analyze_ood.py \
        --results_dir results/evaluations \
        --output_dir results/ood_analysis

    # Analyze specific dimension
    python scripts/analyze_ood.py \
        --results_dir results/evaluations \
        --dimension chain_length \
        --output_dir results/ood_analysis

    # Focus on specific methods
    python scripts/analyze_ood.py \
        --results_dir results/evaluations \
        --methods "lora,dora,pissa" \
        --output_dir results/ood_analysis

    # Generate plots
    python scripts/analyze_ood.py \
        --results_dir results/evaluations \
        --plot \
        --output_dir results/ood_analysis
"""

import argparse
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

# Optional imports
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_stats = None
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("perl.analyze_ood")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ProblemAnalysis:
    """Analysis data for a single problem."""
    problem_id: str
    correct: bool
    peft_method: str
    seed: int
    benchmark: str
    difficulty: Optional[str]
    category: Optional[str]
    reasoning_steps: int
    reasoning_tokens: int
    chain_length_bucket: str  # short, medium, long


@dataclass
class DimensionStats:
    """Statistics for a single dimension value."""
    dimension: str
    value: str
    peft_method: str
    total: int
    correct: int
    accuracy: float
    std: float  # Across seeds if available
    seeds: Dict[int, float] = field(default_factory=dict)


@dataclass
class OODReport:
    """Complete OOD analysis report."""
    generated_at: str
    total_problems: int
    methods: List[str]
    benchmarks: List[str]
    difficulty_analysis: Dict[str, List[DimensionStats]]
    category_analysis: Dict[str, List[DimensionStats]]
    chain_length_analysis: Dict[str, List[DimensionStats]]
    key_findings: List[str]
    statistical_tests: Dict[str, Any]


# =============================================================================
# Chain Length Bucketing
# =============================================================================

def bucket_chain_length(steps: int) -> str:
    """
    Bucket reasoning steps into categories.

    Categories:
    - short: 1-3 steps
    - medium: 4-8 steps
    - long: 9+ steps
    """
    if steps <= 3:
        return "short"
    elif steps <= 8:
        return "medium"
    else:
        return "long"


def normalize_difficulty(difficulty: Optional[str]) -> Optional[str]:
    """Normalize difficulty strings to standard categories."""
    if not difficulty:
        return None

    d = str(difficulty).lower().strip()

    # MATH dataset levels
    if "level 1" in d or "level 2" in d:
        return "easy"
    elif "level 3" in d:
        return "medium"
    elif "level 4" in d or "level 5" in d:
        return "hard"

    # Direct mappings
    if d in ["easy", "simple", "basic"]:
        return "easy"
    elif d in ["medium", "moderate", "intermediate"]:
        return "medium"
    elif d in ["hard", "difficult", "advanced", "challenging"]:
        return "hard"

    # If it looks like a number
    try:
        level = int(d.replace("level", "").strip())
        if level <= 2:
            return "easy"
        elif level == 3:
            return "medium"
        else:
            return "hard"
    except ValueError:
        pass

    return d  # Return as-is if can't normalize


def normalize_category(category: Optional[str]) -> Optional[str]:
    """Normalize category strings."""
    if not category:
        return None

    c = str(category).lower().strip()

    # Common mappings
    mappings = {
        "algebra": "algebra",
        "geometry": "geometry",
        "number theory": "number_theory",
        "number_theory": "number_theory",
        "numbertheory": "number_theory",
        "counting & probability": "counting_probability",
        "counting_probability": "counting_probability",
        "counting and probability": "counting_probability",
        "probability": "counting_probability",
        "prealgebra": "prealgebra",
        "pre-algebra": "prealgebra",
        "precalculus": "precalculus",
        "pre-calculus": "precalculus",
        "intermediate algebra": "intermediate_algebra",
        "intermediate_algebra": "intermediate_algebra",
    }

    return mappings.get(c, c)


# =============================================================================
# Data Loading
# =============================================================================

def load_problem_level_results(results_dir: str) -> List[ProblemAnalysis]:
    """
    Load all problem-level results from evaluation directory.

    Returns list of ProblemAnalysis objects for all problems across all evaluations.
    """
    from perl.evaluation import load_results

    problems = []
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []

    logger.info(f"Loading problem-level results from: {results_dir}")

    for json_file in results_path.rglob("*.json"):
        # Skip non-result files
        if json_file.name in ["evaluation_status.json", "aggregated_results.json"]:
            continue
        if "aggregated" in json_file.name or "comparison" in json_file.name:
            continue
        if "ood_" in json_file.name:
            continue

        try:
            result = load_results(json_file)
            meta = result.metadata

            for prob in result.results:
                chain_bucket = bucket_chain_length(prob.reasoning_steps)

                analysis = ProblemAnalysis(
                    problem_id=prob.problem_id,
                    correct=prob.correct,
                    peft_method=meta.peft_method,
                    seed=meta.seed or 0,
                    benchmark=meta.benchmark,
                    difficulty=normalize_difficulty(prob.difficulty),
                    category=normalize_category(prob.category),
                    reasoning_steps=prob.reasoning_steps,
                    reasoning_tokens=prob.reasoning_tokens,
                    chain_length_bucket=chain_bucket,
                )
                problems.append(analysis)

        except Exception as e:
            logger.debug(f"Failed to load {json_file}: {e}")

    logger.info(f"Loaded {len(problems)} problem-level results")

    return problems


def filter_problems(
    problems: List[ProblemAnalysis],
    methods: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
    min_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> List[ProblemAnalysis]:
    """Filter problems by various criteria."""
    filtered = problems

    if methods:
        methods_lower = [m.lower() for m in methods]
        filtered = [p for p in filtered if p.peft_method.lower() in methods_lower]

    if benchmarks:
        benchmarks_lower = [b.lower() for b in benchmarks]
        filtered = [p for p in filtered if p.benchmark.lower() in benchmarks_lower]

    if min_steps is not None:
        filtered = [p for p in filtered if p.reasoning_steps >= min_steps]

    if max_steps is not None:
        filtered = [p for p in filtered if p.reasoning_steps <= max_steps]

    return filtered


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_by_dimension(
    problems: List[ProblemAnalysis],
    dimension: str,
) -> Dict[str, List[DimensionStats]]:
    """
    Analyze accuracy by a specific dimension.

    Args:
        problems: List of problem analyses
        dimension: One of 'difficulty', 'category', 'chain_length'

    Returns:
        Dict mapping method -> list of DimensionStats
    """
    # Group by method, then by dimension value, then by seed
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for prob in problems:
        if dimension == "difficulty":
            value = prob.difficulty
        elif dimension == "category":
            value = prob.category
        elif dimension == "chain_length":
            value = prob.chain_length_bucket
        else:
            raise ValueError(f"Unknown dimension: {dimension}")

        if value is None:
            continue

        grouped[prob.peft_method][value][prob.seed].append(prob.correct)

    # Compute statistics
    results = {}

    for method, dim_values in grouped.items():
        results[method] = []

        for dim_value, seeds_data in dim_values.items():
            # Compute accuracy per seed
            seed_accuracies = {}
            total_correct = 0
            total_count = 0

            for seed, correctness_list in seeds_data.items():
                if correctness_list:
                    acc = sum(correctness_list) / len(correctness_list)
                    seed_accuracies[seed] = acc
                    total_correct += sum(correctness_list)
                    total_count += len(correctness_list)

            # Overall stats
            overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

            # Std across seeds
            if len(seed_accuracies) > 1:
                std = np.std(list(seed_accuracies.values()), ddof=1)
            else:
                std = 0.0

            stats = DimensionStats(
                dimension=dimension,
                value=dim_value,
                peft_method=method,
                total=total_count,
                correct=total_correct,
                accuracy=float(overall_accuracy),
                std=float(std),
                seeds=seed_accuracies,
            )
            results[method].append(stats)

    return results


def compute_chain_length_detailed(
    problems: List[ProblemAnalysis],
) -> Dict[str, Dict[int, float]]:
    """
    Compute accuracy for each specific chain length (not bucketed).

    Returns:
        Dict: method -> {steps: accuracy}
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for prob in problems:
        grouped[prob.peft_method][prob.reasoning_steps].append(prob.correct)

    results = {}
    for method, steps_data in grouped.items():
        results[method] = {}
        for steps, correctness in steps_data.items():
            if correctness:
                results[method][steps] = sum(correctness) / len(correctness)

    return results


# =============================================================================
# Statistical Testing
# =============================================================================

def paired_comparison(
    problems: List[ProblemAnalysis],
    method1: str,
    method2: str,
    filter_fn=None,
) -> Dict[str, Any]:
    """
    Compare two methods on the same problems.

    Uses paired t-test or Wilcoxon signed-rank test.
    """
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not available - statistical tests skipped"}

    # Get problems solved by both methods (same problem_id, benchmark, seed)
    method1_results = {}
    method2_results = {}

    for prob in problems:
        if filter_fn and not filter_fn(prob):
            continue

        key = (prob.problem_id, prob.benchmark, prob.seed)

        if prob.peft_method.lower() == method1.lower():
            method1_results[key] = 1 if prob.correct else 0
        elif prob.peft_method.lower() == method2.lower():
            method2_results[key] = 1 if prob.correct else 0

    # Find common problems
    common_keys = set(method1_results.keys()) & set(method2_results.keys())

    if len(common_keys) < 10:
        return {"error": f"Not enough common problems ({len(common_keys)})"}

    scores1 = [method1_results[k] for k in common_keys]
    scores2 = [method2_results[k] for k in common_keys]

    acc1 = np.mean(scores1)
    acc2 = np.mean(scores2)
    diff = acc2 - acc1  # positive means method2 is better

    # Paired t-test
    try:
        t_stat, p_value = scipy_stats.ttest_rel(scores2, scores1)
    except Exception:
        t_stat, p_value = None, None

    # Wilcoxon signed-rank (for non-normal data)
    try:
        wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(
            scores2, scores1, alternative='two-sided'
        )
    except Exception:
        wilcoxon_stat, wilcoxon_p = None, None

    return {
        "method1": method1,
        "method2": method2,
        "n_problems": len(common_keys),
        "accuracy_method1": float(acc1),
        "accuracy_method2": float(acc2),
        "difference": float(diff),
        "difference_pct": float(diff * 100),
        "t_statistic": float(t_stat) if t_stat is not None else None,
        "p_value_ttest": float(p_value) if p_value is not None else None,
        "wilcoxon_statistic": float(wilcoxon_stat) if wilcoxon_stat is not None else None,
        "p_value_wilcoxon": float(wilcoxon_p) if wilcoxon_p is not None else None,
        "significant_005": p_value < 0.05 if p_value is not None else None,
        "significant_001": p_value < 0.01 if p_value is not None else None,
    }


def run_all_pairwise_tests(
    problems: List[ProblemAnalysis],
    methods: List[str],
    dimension: str = None,
    dimension_value: str = None,
) -> Dict[str, Any]:
    """
    Run pairwise statistical tests between all method pairs.
    """
    if not SCIPY_AVAILABLE:
        logger.warning(
            "*** USING FALLBACK: scipy not available, "
            "SKIPPING ALL PAIRWISE STATISTICAL TESTS ***"
        )
        return {"_fallback": "scipy not available - no statistical tests performed"}

    results = {}

    # Create filter function
    if dimension and dimension_value:
        if dimension == "chain_length":
            filter_fn = lambda p: p.chain_length_bucket == dimension_value
        elif dimension == "difficulty":
            filter_fn = lambda p: p.difficulty == dimension_value
        elif dimension == "category":
            filter_fn = lambda p: p.category == dimension_value
        else:
            filter_fn = None
    else:
        filter_fn = None

    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            key = f"{m1}_vs_{m2}"
            results[key] = paired_comparison(problems, m1, m2, filter_fn)

    return results


# =============================================================================
# Key Findings Detection
# =============================================================================

def detect_key_findings(
    difficulty_analysis: Dict[str, List[DimensionStats]],
    category_analysis: Dict[str, List[DimensionStats]],
    chain_length_analysis: Dict[str, List[DimensionStats]],
    statistical_tests: Dict[str, Any],
) -> List[str]:
    """
    Automatically detect key findings from the analysis.
    """
    findings = []

    # Find best method for long-chain problems
    long_chain_stats = {}
    for method, stats_list in chain_length_analysis.items():
        for stats in stats_list:
            if stats.value == "long":
                long_chain_stats[method] = stats.accuracy

    if long_chain_stats:
        best_method = max(long_chain_stats, key=long_chain_stats.get)
        best_acc = long_chain_stats[best_method]

        # Compare to others
        for method, acc in long_chain_stats.items():
            if method != best_method and acc > 0:
                diff = best_acc - acc
                if diff > 0.05:  # 5% difference
                    findings.append(
                        f"{best_method} outperforms {method} on long-chain problems "
                        f"by {diff*100:.1f}% ({best_acc*100:.1f}% vs {acc*100:.1f}%)"
                    )

    # Find methods that struggle with hard problems
    hard_stats = {}
    for method, stats_list in difficulty_analysis.items():
        for stats in stats_list:
            if stats.value == "hard":
                hard_stats[method] = stats.accuracy

    if hard_stats:
        # Find worst performer
        worst_method = min(hard_stats, key=hard_stats.get)
        worst_acc = hard_stats[worst_method]

        if worst_acc < 0.1:  # Less than 10% on hard problems
            findings.append(
                f"{worst_method} struggles significantly on hard problems "
                f"(only {worst_acc*100:.1f}% accuracy)"
            )

    # Add significant statistical findings
    for test_name, test_result in statistical_tests.items():
        if isinstance(test_result, dict) and test_result.get("significant_001"):
            m1 = test_result.get("method1", "")
            m2 = test_result.get("method2", "")
            diff = test_result.get("difference_pct", 0)
            p = test_result.get("p_value_ttest", 1)

            if abs(diff) >= 5:  # At least 5% difference
                better = m2 if diff > 0 else m1
                worse = m1 if diff > 0 else m2
                findings.append(
                    f"{better} significantly outperforms {worse} "
                    f"(+{abs(diff):.1f}%, p={p:.4f})"
                )

    return findings


# =============================================================================
# Report Generation
# =============================================================================

def generate_ood_report(
    problems: List[ProblemAnalysis],
    output_dir: str,
) -> OODReport:
    """Generate complete OOD analysis report."""
    methods = sorted(set(p.peft_method for p in problems))
    benchmarks = sorted(set(p.benchmark for p in problems))

    logger.info("Analyzing by difficulty...")
    difficulty_analysis = analyze_by_dimension(problems, "difficulty")

    logger.info("Analyzing by category...")
    category_analysis = analyze_by_dimension(problems, "category")

    logger.info("Analyzing by chain length...")
    chain_length_analysis = analyze_by_dimension(problems, "chain_length")

    logger.info("Running statistical tests...")
    statistical_tests = {}

    # Overall tests
    statistical_tests["overall"] = run_all_pairwise_tests(problems, methods)

    # Tests on long-chain only
    statistical_tests["long_chain_only"] = run_all_pairwise_tests(
        problems, methods, "chain_length", "long"
    )

    # Tests on hard problems only
    statistical_tests["hard_only"] = run_all_pairwise_tests(
        problems, methods, "difficulty", "hard"
    )

    logger.info("Detecting key findings...")
    key_findings = detect_key_findings(
        difficulty_analysis, category_analysis, chain_length_analysis,
        statistical_tests.get("long_chain_only", {})
    )

    report = OODReport(
        generated_at=datetime.now().isoformat(),
        total_problems=len(problems),
        methods=methods,
        benchmarks=benchmarks,
        difficulty_analysis=difficulty_analysis,
        category_analysis=category_analysis,
        chain_length_analysis=chain_length_analysis,
        key_findings=key_findings,
        statistical_tests=statistical_tests,
    )

    return report


def save_report_json(report: OODReport, output_path: str):
    """Save report as JSON."""
    data = {
        "generated_at": report.generated_at,
        "total_problems": report.total_problems,
        "methods": report.methods,
        "benchmarks": report.benchmarks,
        "difficulty_analysis": {
            method: [
                {
                    "value": s.value,
                    "total": s.total,
                    "correct": s.correct,
                    "accuracy": s.accuracy,
                    "std": s.std,
                    "seeds": s.seeds,
                }
                for s in stats_list
            ]
            for method, stats_list in report.difficulty_analysis.items()
        },
        "category_analysis": {
            method: [
                {
                    "value": s.value,
                    "total": s.total,
                    "correct": s.correct,
                    "accuracy": s.accuracy,
                    "std": s.std,
                }
                for s in stats_list
            ]
            for method, stats_list in report.category_analysis.items()
        },
        "chain_length_analysis": {
            method: [
                {
                    "value": s.value,
                    "total": s.total,
                    "correct": s.correct,
                    "accuracy": s.accuracy,
                    "std": s.std,
                }
                for s in stats_list
            ]
            for method, stats_list in report.chain_length_analysis.items()
        },
        "key_findings": report.key_findings,
        "statistical_tests": report.statistical_tests,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved JSON report: {output_path}")


def generate_markdown_report(report: OODReport) -> str:
    """Generate markdown report."""
    lines = [
        "# OOD (Out-of-Distribution) Analysis Report",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Summary",
        "",
        f"- Total problems analyzed: {report.total_problems:,}",
        f"- PEFT methods: {', '.join(report.methods)}",
        f"- Benchmarks: {', '.join(report.benchmarks)}",
        "",
    ]

    # Key findings
    if report.key_findings:
        lines.extend([
            "## Key Findings",
            "",
        ])
        for i, finding in enumerate(report.key_findings, 1):
            lines.append(f"{i}. **{finding}**")
        lines.append("")

    # Difficulty Analysis
    lines.extend([
        "## Difficulty Analysis",
        "",
    ])

    # Build difficulty table
    difficulties = set()
    for stats_list in report.difficulty_analysis.values():
        for s in stats_list:
            difficulties.add(s.value)
    difficulties = sorted(difficulties, key=lambda x: {"easy": 0, "medium": 1, "hard": 2}.get(x, 3))

    if difficulties:
        header = "| Method |"
        separator = "|--------|"
        for d in difficulties:
            header += f" {d.capitalize()} |"
            separator += "--------|"
        lines.extend([header, separator])

        for method in sorted(report.methods):
            row = f"| {method} |"
            stats_dict = {s.value: s for s in report.difficulty_analysis.get(method, [])}
            for d in difficulties:
                if d in stats_dict:
                    s = stats_dict[d]
                    val = f"{s.accuracy:.3f}"
                    if s.std > 0:
                        val += f" ± {s.std:.3f}"
                    row += f" {val} |"
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

    # Chain Length Analysis
    lines.extend([
        "## Chain Length Analysis",
        "",
        "Chain length buckets: short (1-3 steps), medium (4-8 steps), long (9+ steps)",
        "",
    ])

    buckets = ["short", "medium", "long"]
    header = "| Method |"
    separator = "|--------|"
    for b in buckets:
        header += f" {b.capitalize()} |"
        separator += "--------|"
    lines.extend([header, separator])

    for method in sorted(report.methods):
        row = f"| {method} |"
        stats_dict = {s.value: s for s in report.chain_length_analysis.get(method, [])}
        for b in buckets:
            if b in stats_dict:
                s = stats_dict[b]
                val = f"{s.accuracy:.3f}"
                if s.std > 0:
                    val += f" ± {s.std:.3f}"
                row += f" {val} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # Category Analysis
    if report.category_analysis:
        lines.extend([
            "## Category Analysis",
            "",
        ])

        categories = set()
        for stats_list in report.category_analysis.values():
            for s in stats_list:
                categories.add(s.value)
        categories = sorted(categories)

        if categories:
            header = "| Method |"
            separator = "|--------|"
            for c in categories:
                header += f" {c[:12]} |"
                separator += "--------|"
            lines.extend([header, separator])

            for method in sorted(report.methods):
                row = f"| {method} |"
                stats_dict = {s.value: s for s in report.category_analysis.get(method, [])}
                for c in categories:
                    if c in stats_dict:
                        s = stats_dict[c]
                        row += f" {s.accuracy:.3f} |"
                    else:
                        row += " - |"
                lines.append(row)
            lines.append("")

    # Statistical Tests
    if report.statistical_tests:
        lines.extend([
            "## Statistical Comparisons",
            "",
        ])

        # Long-chain comparisons
        long_chain_tests = report.statistical_tests.get("long_chain_only", {})
        if long_chain_tests:
            lines.extend([
                "### Long-Chain Problems (9+ steps)",
                "",
                "| Comparison | Diff (%) | p-value | Significant |",
                "|------------|----------|---------|-------------|",
            ])

            for test_name, result in long_chain_tests.items():
                if isinstance(result, dict) and "error" not in result:
                    diff = result.get("difference_pct", 0)
                    p = result.get("p_value_ttest")
                    sig = ""
                    if result.get("significant_001"):
                        sig = "***"
                    elif result.get("significant_005"):
                        sig = "**"

                    p_str = f"{p:.4f}" if p else "N/A"
                    lines.append(f"| {test_name} | {diff:+.1f}% | {p_str} | {sig} |")

            lines.extend([
                "",
                "Significance: ** p < 0.05, *** p < 0.01",
                "",
            ])

    return "\n".join(lines)


def save_report_markdown(report: OODReport, output_path: str):
    """Save report as markdown."""
    content = generate_markdown_report(report)
    with open(output_path, "w") as f:
        f.write(content)
    logger.info(f"Saved Markdown report: {output_path}")


def print_report(report: OODReport):
    """Print report to console."""
    print("\n" + "=" * 80)
    print("OOD ANALYSIS REPORT")
    print("=" * 80)
    print(f"Generated: {report.generated_at}")
    print(f"Total problems: {report.total_problems:,}")
    print(f"Methods: {', '.join(report.methods)}")

    # Key findings
    if report.key_findings:
        print("\n" + "-" * 80)
        print("KEY FINDINGS")
        print("-" * 80)
        for i, finding in enumerate(report.key_findings, 1):
            print(f"  {i}. {finding}")

    # Chain length table
    print("\n" + "-" * 80)
    print("CHAIN LENGTH ANALYSIS")
    print("-" * 80)
    print(f"{'Method':<12} | {'Short':>10} | {'Medium':>10} | {'Long':>10}")
    print("-" * 50)

    for method in sorted(report.methods):
        stats_dict = {s.value: s for s in report.chain_length_analysis.get(method, [])}
        short = stats_dict.get("short")
        medium = stats_dict.get("medium")
        long = stats_dict.get("long")

        short_str = f"{short.accuracy:.3f}" if short else "-"
        medium_str = f"{medium.accuracy:.3f}" if medium else "-"
        long_str = f"{long.accuracy:.3f}" if long else "-"

        print(f"{method:<12} | {short_str:>10} | {medium_str:>10} | {long_str:>10}")

    print("=" * 80)


# =============================================================================
# Plotting
# =============================================================================

def generate_plots(
    report: OODReport,
    problems: List[ProblemAnalysis],
    output_dir: str,
):
    """Generate visualization plots."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping plots")
        return

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Accuracy vs Chain Length (bucketed)
    plot_chain_length_comparison(report, plots_dir / "chain_length_comparison.png")

    # Plot 2: Detailed accuracy by steps
    plot_accuracy_by_steps(problems, plots_dir / "accuracy_by_steps.png")

    # Plot 3: Difficulty comparison
    plot_difficulty_comparison(report, plots_dir / "difficulty_comparison.png")

    # Plot 4: Category heatmap
    plot_category_heatmap(report, plots_dir / "category_heatmap.png")

    logger.info(f"Saved plots to: {plots_dir}")


def plot_chain_length_comparison(report: OODReport, output_path: Path):
    """Plot accuracy by chain length bucket for each method."""
    if not MATPLOTLIB_AVAILABLE:
        return

    buckets = ["short", "medium", "long"]
    x = np.arange(len(buckets))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(sorted(report.methods)):
        stats_dict = {s.value: s for s in report.chain_length_analysis.get(method, [])}
        accuracies = [stats_dict.get(b, DimensionStats("", b, method, 0, 0, 0, 0)).accuracy
                      for b in buckets]
        errors = [stats_dict.get(b, DimensionStats("", b, method, 0, 0, 0, 0)).std
                  for b in buckets]

        ax.bar(x + i * width, accuracies, width, label=method, yerr=errors, capsize=3)

    ax.set_xlabel("Reasoning Chain Length")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Reasoning Chain Length")
    ax.set_xticks(x + width * (len(report.methods) - 1) / 2)
    ax.set_xticklabels(["Short\n(1-3 steps)", "Medium\n(4-8 steps)", "Long\n(9+ steps)"])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_accuracy_by_steps(problems: List[ProblemAnalysis], output_path: Path):
    """Plot detailed accuracy by number of reasoning steps."""
    if not MATPLOTLIB_AVAILABLE:
        return

    detailed = compute_chain_length_detailed(problems)

    fig, ax = plt.subplots(figsize=(12, 6))

    for method, steps_data in detailed.items():
        steps = sorted(steps_data.keys())
        accs = [steps_data[s] for s in steps]

        # Smooth with rolling average for visualization
        if len(steps) > 5:
            ax.plot(steps, accs, 'o-', label=method, alpha=0.7, markersize=4)
        else:
            ax.plot(steps, accs, 'o-', label=method, markersize=6)

    ax.set_xlabel("Number of Reasoning Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Reasoning Steps (Detailed)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add vertical lines for bucket boundaries
    ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5, label='Short/Medium')
    ax.axvline(x=8.5, color='gray', linestyle='--', alpha=0.5, label='Medium/Long')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_difficulty_comparison(report: OODReport, output_path: Path):
    """Plot accuracy by difficulty level."""
    if not MATPLOTLIB_AVAILABLE:
        return

    difficulties = ["easy", "medium", "hard"]
    x = np.arange(len(difficulties))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(sorted(report.methods)):
        stats_dict = {s.value: s for s in report.difficulty_analysis.get(method, [])}
        accuracies = [stats_dict.get(d, DimensionStats("", d, method, 0, 0, 0, 0)).accuracy
                      for d in difficulties]
        errors = [stats_dict.get(d, DimensionStats("", d, method, 0, 0, 0, 0)).std
                  for d in difficulties]

        ax.bar(x + i * width, accuracies, width, label=method, yerr=errors, capsize=3)

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Problem Difficulty")
    ax.set_xticks(x + width * (len(report.methods) - 1) / 2)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_category_heatmap(report: OODReport, output_path: Path):
    """Plot accuracy heatmap by method and category."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Get all categories
    categories = set()
    for stats_list in report.category_analysis.values():
        for s in stats_list:
            categories.add(s.value)
    categories = sorted(categories)

    if not categories:
        return

    methods = sorted(report.methods)

    # Build accuracy matrix
    data = np.zeros((len(methods), len(categories)))

    for i, method in enumerate(methods):
        stats_dict = {s.value: s for s in report.category_analysis.get(method, [])}
        for j, cat in enumerate(categories):
            if cat in stats_dict:
                data[i, j] = stats_dict[cat].accuracy

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(methods)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(categories)):
            if data[i, j] > 0:
                text = ax.text(j, i, f"{data[i, j]:.2f}",
                               ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Accuracy by Method and Category")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OOD analysis for PeRL evaluation results",
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
        "--output_dir", "-o",
        type=str,
        default="results/ood_analysis",
        help="Output directory for reports and plots",
    )

    parser.add_argument(
        "--dimension",
        type=str,
        default=None,
        choices=["difficulty", "category", "chain_length", "all"],
        help="Specific dimension to analyze (default: all)",
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
        "--plot",
        action="store_true",
        help="Generate visualization plots",
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

    # Load problems
    problems = load_problem_level_results(args.results_dir)

    if not problems:
        logger.error("No problems found!")
        sys.exit(1)

    # Apply filters
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        problems = filter_problems(problems, methods=methods)

    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
        problems = filter_problems(problems, benchmarks=benchmarks)

    logger.info(f"Analyzing {len(problems)} problems")

    # Generate report
    report = generate_ood_report(problems, args.output_dir)

    # Print to console
    print_report(report)

    # Save reports
    os.makedirs(args.output_dir, exist_ok=True)

    save_report_json(report, os.path.join(args.output_dir, "ood_report.json"))
    save_report_markdown(report, os.path.join(args.output_dir, "OOD_REPORT.md"))

    # Generate plots
    if args.plot:
        generate_plots(report, problems, args.output_dir)

    print(f"\nReports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
