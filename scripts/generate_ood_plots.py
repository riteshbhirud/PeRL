#!/usr/bin/env python3
"""
Standalone OOD Analysis Visualization Generator.

Generates publication-quality plots from OOD analysis results:
A) Accuracy vs Chain Length - Line plot comparing methods
B) Accuracy by Category Heatmap - Methods x Categories matrix
C) Difficulty Distribution - Grouped bar chart by method

Usage:
    python scripts/generate_ood_plots.py \
        --analysis-dir results/ood_analysis/ \
        --output-dir results/ood_analysis/plots/

    # Or with raw evaluation results:
    python scripts/generate_ood_plots.py \
        --results-dir results/aggregated/ \
        --output-dir results/ood_analysis/plots/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np


# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Try to import seaborn for enhanced styling
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None


# Color palette for methods
METHOD_COLORS = {
    "lora": "#1f77b4",      # Blue
    "dora": "#ff7f0e",      # Orange
    "adalora": "#2ca02c",   # Green
    "pissa": "#d62728",     # Red
    "milora": "#9467bd",    # Purple
    "vera": "#8c564b",      # Brown
    "lora_plus": "#e377c2", # Pink
    "miss": "#7f7f7f",      # Gray
    "full": "#bcbd22",      # Olive
    "lorafa": "#17becf",    # Cyan
}

# Method display names
METHOD_NAMES = {
    "lora": "LoRA",
    "dora": "DoRA",
    "adalora": "AdaLoRA",
    "pissa": "PiSSA",
    "milora": "MiLoRA",
    "vera": "VeRA",
    "lora_plus": "LoRA+",
    "miss": "MiSS",
    "full": "Full FT",
    "lorafa": "LoRA-FA",
}

# Chain length bucket labels
CHAIN_BUCKETS = {
    "short": "Short (1-3)",
    "medium": "Medium (4-8)",
    "long": "Long (9+)",
}


def setup_plot_style():
    """Set up matplotlib style for publication-quality plots."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    if SEABORN_AVAILABLE:
        sns.set_palette("husl")
        sns.set_style("whitegrid")


def load_analysis_results(analysis_dir: Path) -> Dict[str, Any]:
    """Load pre-computed analysis results."""
    results = {}

    # Load chain length analysis
    chain_file = analysis_dir / "chain_length_analysis.json"
    if chain_file.exists():
        with open(chain_file) as f:
            results["chain_length"] = json.load(f)

    # Load category analysis
    category_file = analysis_dir / "category_analysis.json"
    if category_file.exists():
        with open(category_file) as f:
            results["category"] = json.load(f)

    # Load difficulty analysis
    difficulty_file = analysis_dir / "difficulty_analysis.json"
    if difficulty_file.exists():
        with open(difficulty_file) as f:
            results["difficulty"] = json.load(f)

    # Load combined analysis
    combined_file = analysis_dir / "ood_analysis.json"
    if combined_file.exists():
        with open(combined_file) as f:
            results["combined"] = json.load(f)

    return results


def load_aggregated_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load aggregated evaluation results by method."""
    results = {}

    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        results[method_name] = {"benchmarks": {}}

        # Look for aggregated results
        for benchmark_dir in method_dir.iterdir():
            if not benchmark_dir.is_dir():
                continue

            agg_file = benchmark_dir / "aggregated.json"
            if agg_file.exists():
                with open(agg_file) as f:
                    results[method_name]["benchmarks"][benchmark_dir.name] = json.load(f)

    return results


def extract_chain_length_data(
    analysis: Dict[str, Any],
    methods: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract chain length accuracy data for plotting."""
    buckets = ["short", "medium", "long"]
    accuracies = defaultdict(list)
    errors = defaultdict(list)

    chain_data = analysis.get("chain_length", {})
    if not chain_data:
        # Try combined analysis
        chain_data = analysis.get("combined", {}).get("chain_length", {})

    if not chain_data:
        return buckets, {}, {}

    # Get methods from data
    available_methods = set()
    for bucket_data in chain_data.values():
        if isinstance(bucket_data, dict):
            available_methods.update(bucket_data.keys())

    if methods:
        available_methods = available_methods.intersection(methods)

    for method in sorted(available_methods):
        for bucket in buckets:
            bucket_data = chain_data.get(bucket, {})
            if isinstance(bucket_data, dict) and method in bucket_data:
                method_data = bucket_data[method]
                if isinstance(method_data, dict):
                    accuracies[method].append(method_data.get("accuracy", 0) * 100)
                    errors[method].append(method_data.get("std", 0) * 100)
                else:
                    accuracies[method].append(float(method_data) * 100)
                    errors[method].append(0)
            else:
                accuracies[method].append(0)
                errors[method].append(0)

    return buckets, dict(accuracies), dict(errors)


def extract_category_data(
    analysis: Dict[str, Any],
    methods: Optional[List[str]] = None
) -> Tuple[List[str], List[str], np.ndarray]:
    """Extract category accuracy data for heatmap."""
    category_data = analysis.get("category", {})
    if not category_data:
        category_data = analysis.get("combined", {}).get("category", {})

    if not category_data:
        return [], [], np.array([])

    # Get categories and methods
    categories = sorted(category_data.keys())

    available_methods = set()
    for cat_data in category_data.values():
        if isinstance(cat_data, dict):
            available_methods.update(cat_data.keys())

    if methods:
        available_methods = available_methods.intersection(methods)

    method_list = sorted(available_methods)

    # Build accuracy matrix
    matrix = np.zeros((len(method_list), len(categories)))

    for j, category in enumerate(categories):
        cat_data = category_data.get(category, {})
        for i, method in enumerate(method_list):
            if method in cat_data:
                method_data = cat_data[method]
                if isinstance(method_data, dict):
                    matrix[i, j] = method_data.get("accuracy", 0) * 100
                else:
                    matrix[i, j] = float(method_data) * 100

    return method_list, categories, matrix


def extract_difficulty_data(
    analysis: Dict[str, Any],
    methods: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Extract difficulty accuracy data for bar chart."""
    difficulty_data = analysis.get("difficulty", {})
    if not difficulty_data:
        difficulty_data = analysis.get("combined", {}).get("difficulty", {})

    if not difficulty_data:
        return [], {}

    # Get difficulties and methods
    difficulties = ["easy", "medium", "hard"]

    available_methods = set()
    for diff_data in difficulty_data.values():
        if isinstance(diff_data, dict):
            available_methods.update(diff_data.keys())

    if methods:
        available_methods = available_methods.intersection(methods)

    method_list = sorted(available_methods)

    # Build data structure
    result = {}
    for method in method_list:
        result[method] = {}
        for difficulty in difficulties:
            diff_data = difficulty_data.get(difficulty, {})
            if method in diff_data:
                method_data = diff_data[method]
                if isinstance(method_data, dict):
                    result[method][difficulty] = method_data.get("accuracy", 0) * 100
                else:
                    result[method][difficulty] = float(method_data) * 100
            else:
                result[method][difficulty] = 0

    return difficulties, result


def plot_chain_length_comparison(
    buckets: List[str],
    accuracies: Dict[str, List[float]],
    errors: Dict[str, List[float]],
    output_path: Path,
    title: str = "Accuracy vs Reasoning Chain Length"
):
    """
    Generate accuracy vs chain length line plot.

    X-axis: Number of reasoning steps (bucketed)
    Y-axis: Accuracy (%)
    Different lines for different methods
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping chain length plot")
        return

    if not accuracies:
        print("Warning: No chain length data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(buckets))
    bucket_labels = [CHAIN_BUCKETS.get(b, b) for b in buckets]

    for method, accs in accuracies.items():
        color = METHOD_COLORS.get(method, None)
        label = METHOD_NAMES.get(method, method)
        errs = errors.get(method, [0] * len(accs))

        ax.errorbar(
            x, accs, yerr=errs,
            marker='o', markersize=8,
            linewidth=2, capsize=5,
            label=label, color=color
        )

    ax.set_xlabel("Reasoning Chain Length")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim(0, 100)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved chain length plot to {output_path}")


def plot_category_heatmap(
    methods: List[str],
    categories: List[str],
    matrix: np.ndarray,
    output_path: Path,
    title: str = "Accuracy by Problem Category"
):
    """
    Generate category accuracy heatmap.

    Rows: Methods
    Columns: Categories
    Color: Accuracy (%)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping category heatmap")
        return

    if len(methods) == 0 or len(categories) == 0:
        print("Warning: No category data available")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(methods) * 0.6)))

    # Create heatmap
    if SEABORN_AVAILABLE:
        method_labels = [METHOD_NAMES.get(m, m) for m in methods]
        sns.heatmap(
            matrix,
            annot=True, fmt='.1f',
            xticklabels=categories,
            yticklabels=method_labels,
            cmap='RdYlGn',
            vmin=0, vmax=100,
            cbar_kws={'label': 'Accuracy (%)'},
            ax=ax
        )
    else:
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)')

        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(categories)):
                text = ax.text(
                    j, i, f'{matrix[i, j]:.1f}',
                    ha='center', va='center',
                    color='white' if matrix[i, j] < 50 else 'black',
                    fontsize=9
                )

        # Set ticks
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_yticklabels([METHOD_NAMES.get(m, m) for m in methods])

    ax.set_xlabel("Problem Category")
    ax.set_ylabel("PEFT Method")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved category heatmap to {output_path}")


def plot_difficulty_distribution(
    difficulties: List[str],
    data: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Accuracy by Problem Difficulty"
):
    """
    Generate difficulty distribution bar chart.

    X-axis: Difficulty levels (Easy, Medium, Hard)
    Y-axis: Accuracy (%)
    Grouped bars by method
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping difficulty plot")
        return

    if not data:
        print("Warning: No difficulty data available")
        return

    methods = list(data.keys())
    n_methods = len(methods)
    n_difficulties = len(difficulties)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate bar positions
    bar_width = 0.8 / n_methods
    x = np.arange(n_difficulties)

    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        values = [data[method].get(d, 0) for d in difficulties]
        color = METHOD_COLORS.get(method, None)
        label = METHOD_NAMES.get(method, method)

        ax.bar(
            x + offset, values,
            bar_width * 0.9,
            label=label,
            color=color
        )

    ax.set_xlabel("Problem Difficulty")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_ylim(0, 100)
    ax.legend(loc='best', framealpha=0.9, ncol=min(3, n_methods))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved difficulty distribution plot to {output_path}")


def plot_method_comparison_summary(
    analysis: Dict[str, Any],
    output_path: Path,
    title: str = "PEFT Method Comparison Summary"
):
    """Generate a summary comparison plot across all dimensions."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping summary plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Chain Length
    buckets, accuracies, errors = extract_chain_length_data(analysis)
    if accuracies:
        ax = axes[0]
        x = np.arange(len(buckets))
        bucket_labels = [CHAIN_BUCKETS.get(b, b) for b in buckets]

        for method, accs in accuracies.items():
            color = METHOD_COLORS.get(method, None)
            label = METHOD_NAMES.get(method, method)
            ax.plot(x, accs, marker='o', markersize=6, linewidth=2, label=label, color=color)

        ax.set_xlabel("Chain Length")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("By Chain Length")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=15)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    # Plot 2: Difficulty
    difficulties, diff_data = extract_difficulty_data(analysis)
    if diff_data:
        ax = axes[1]
        methods = list(diff_data.keys())
        n_methods = len(methods)
        bar_width = 0.8 / n_methods
        x = np.arange(len(difficulties))

        for i, method in enumerate(methods):
            offset = (i - n_methods / 2 + 0.5) * bar_width
            values = [diff_data[method].get(d, 0) for d in difficulties]
            color = METHOD_COLORS.get(method, None)
            ax.bar(x + offset, values, bar_width * 0.9, color=color)

        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("By Difficulty")
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Overall comparison
    ax = axes[2]
    combined = analysis.get("combined", {})
    if "overall" in combined:
        overall = combined["overall"]
        methods = sorted(overall.keys())
        accuracies = []
        for method in methods:
            method_data = overall[method]
            if isinstance(method_data, dict):
                accuracies.append(method_data.get("accuracy", 0) * 100)
            else:
                accuracies.append(float(method_data) * 100)

        colors = [METHOD_COLORS.get(m, '#333333') for m in methods]
        labels = [METHOD_NAMES.get(m, m) for m in methods]

        bars = ax.bar(range(len(methods)), accuracies, color=colors)
        ax.set_xlabel("PEFT Method")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Overall Accuracy")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

    # Add legend for methods (only once)
    handles = []
    labels = []
    for method in sorted(METHOD_COLORS.keys()):
        if method in accuracies if 'accuracies' in dir() else True:
            handles.append(Patch(facecolor=METHOD_COLORS[method], label=METHOD_NAMES.get(method, method)))
            labels.append(METHOD_NAMES.get(method, method))

    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(5, len(handles)),
                   bbox_to_anchor=(0.5, 1.05), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved summary plot to {output_path}")


def plot_learning_curves_by_chain(
    checkpoints_data: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Learning Curves by Chain Length"
):
    """Generate learning curves separated by chain length."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping learning curves")
        return

    # This requires checkpoint-level data organized by step
    # Each entry should have: method, step, chain_bucket -> accuracy

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bucket_names = ["short", "medium", "long"]

    for ax, bucket in zip(axes, bucket_names):
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{CHAIN_BUCKETS.get(bucket, bucket)} Chains")
        ax.grid(True, alpha=0.3)

        # Plot data for this bucket
        for method, data in checkpoints_data.items():
            steps = sorted(data.get("steps", {}).keys())
            accs = []
            for step in steps:
                step_data = data["steps"][step]
                if bucket in step_data:
                    accs.append(step_data[bucket].get("accuracy", 0) * 100)
                else:
                    accs.append(None)

            # Filter None values
            valid_steps = [s for s, a in zip(steps, accs) if a is not None]
            valid_accs = [a for a in accs if a is not None]

            if valid_steps:
                color = METHOD_COLORS.get(method, None)
                label = METHOD_NAMES.get(method, method)
                ax.plot(valid_steps, valid_accs, marker='o', markersize=4,
                       linewidth=2, label=label, color=color)

        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved learning curves plot to {output_path}")


def generate_all_plots(
    analysis: Dict[str, Any],
    output_dir: Path,
    methods: Optional[List[str]] = None
):
    """Generate all OOD analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}...")

    # A) Accuracy vs Chain Length
    buckets, accuracies, errors = extract_chain_length_data(analysis, methods)
    if accuracies:
        plot_chain_length_comparison(
            buckets, accuracies, errors,
            output_dir / "accuracy_vs_chain_length.png"
        )

    # B) Category Heatmap
    method_list, categories, matrix = extract_category_data(analysis, methods)
    if len(method_list) > 0 and len(categories) > 0:
        plot_category_heatmap(
            method_list, categories, matrix,
            output_dir / "accuracy_by_category_heatmap.png"
        )

    # C) Difficulty Distribution
    difficulties, diff_data = extract_difficulty_data(analysis, methods)
    if diff_data:
        plot_difficulty_distribution(
            difficulties, diff_data,
            output_dir / "accuracy_by_difficulty.png"
        )

    # Summary plot
    plot_method_comparison_summary(
        analysis,
        output_dir / "method_comparison_summary.png"
    )

    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate OOD analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate plots from analysis results
    python scripts/generate_ood_plots.py \\
        --analysis-dir results/ood_analysis/ \\
        --output-dir results/ood_analysis/plots/

    # Generate plots with specific methods only
    python scripts/generate_ood_plots.py \\
        --analysis-dir results/ood_analysis/ \\
        --methods lora dora pissa milora \\
        --output-dir results/plots/
        """
    )

    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Directory containing OOD analysis JSON files"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing aggregated evaluation results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ood_analysis/plots"),
        help="Directory for output plots (default: results/ood_analysis/plots)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        help="Specific methods to include (default: all)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for plots (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output (default: 300)"
    )

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting")
        print("Install with: pip install matplotlib")
        sys.exit(1)

    if not args.analysis_dir and not args.results_dir:
        print("Error: Must specify either --analysis-dir or --results-dir")
        sys.exit(1)

    # Set up plot style
    setup_plot_style()
    if args.dpi:
        plt.rcParams['savefig.dpi'] = args.dpi

    # Load data
    if args.analysis_dir:
        if not args.analysis_dir.exists():
            print(f"Error: Analysis directory not found: {args.analysis_dir}")
            sys.exit(1)

        print(f"Loading analysis results from {args.analysis_dir}...")
        analysis = load_analysis_results(args.analysis_dir)

        if not analysis:
            print("Error: No analysis files found")
            sys.exit(1)
    else:
        if not args.results_dir.exists():
            print(f"Error: Results directory not found: {args.results_dir}")
            sys.exit(1)

        print(f"Loading aggregated results from {args.results_dir}...")
        results = load_aggregated_results(args.results_dir)

        # Convert to analysis format
        # This would need additional processing depending on the result format
        print("Note: Direct result loading requires additional processing")
        print("Consider running analyze_ood.py first to generate analysis files")
        sys.exit(1)

    # Generate plots
    generate_all_plots(analysis, args.output_dir, args.methods)

    print("\nPlot generation complete!")


if __name__ == "__main__":
    main()
