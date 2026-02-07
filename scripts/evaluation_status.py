#!/usr/bin/env python3
"""
Evaluation Status Checker for PeRL.

Check progress of batch evaluations, identify failures, and estimate completion.

Usage:
    # Quick status check
    python scripts/evaluation_status.py --results_dir results/evaluations

    # Detailed status with failures
    python scripts/evaluation_status.py --results_dir results/evaluations --verbose

    # Watch mode (refresh every N seconds)
    python scripts/evaluation_status.py --results_dir results/evaluations --watch

    # JSON output for scripting
    python scripts/evaluation_status.py --results_dir results/evaluations --json

    # Check specific experiments
    python scripts/evaluation_status.py --results_dir results/evaluations --filter "lora,dora"
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvaluationEntry:
    """Information about a single evaluation."""
    experiment: str
    benchmark: str
    status: str
    accuracy: Optional[float]
    result_path: str
    error: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class StatusSummary:
    """Summary of evaluation status."""
    total_expected: int
    completed: int
    failed: int
    pending: int
    accuracy_by_method: Dict[str, Dict[str, float]]
    entries: List[EvaluationEntry]
    elapsed_time: Optional[float] = None
    estimated_remaining: Optional[float] = None


def scan_results_directory(results_dir: str) -> List[EvaluationEntry]:
    """Scan results directory for completed evaluations."""
    from perl.evaluation import load_results

    entries = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return entries

    for json_file in results_path.rglob("*.json"):
        # Skip status files
        if json_file.name in ["evaluation_status.json", "aggregated_results.json"]:
            continue
        if "aggregated" in json_file.name or "comparison" in json_file.name:
            continue

        try:
            result = load_results(json_file)
            m = result.metadata

            # Extract experiment name from path
            experiment = json_file.parent.name
            if experiment.startswith("step_"):
                experiment = json_file.parent.parent.name

            entries.append(EvaluationEntry(
                experiment=experiment,
                benchmark=m.benchmark,
                status="completed",
                accuracy=result.statistics.accuracy,
                result_path=str(json_file),
                timestamp=m.evaluation_date,
            ))

        except Exception as e:
            entries.append(EvaluationEntry(
                experiment=json_file.parent.name,
                benchmark=json_file.stem,
                status="failed",
                accuracy=None,
                result_path=str(json_file),
                error=str(e),
            ))

    return entries


def load_status_file(results_dir: str) -> Optional[dict]:
    """Load the evaluation_status.json if it exists."""
    status_path = Path(results_dir) / "evaluation_status.json"
    if status_path.exists():
        with open(status_path) as f:
            return json.load(f)
    return None


def compute_summary(
    entries: List[EvaluationEntry],
    status_data: Optional[dict] = None,
) -> StatusSummary:
    """Compute summary statistics from evaluation entries."""
    completed = [e for e in entries if e.status == "completed"]
    failed = [e for e in entries if e.status == "failed"]

    # Get total expected from status file or estimate
    total_expected = len(entries)
    if status_data:
        total_expected = status_data.get("total_tasks", total_expected)

    pending = max(0, total_expected - len(completed) - len(failed))

    # Compute accuracy by method
    accuracy_by_method = defaultdict(dict)
    for entry in completed:
        if entry.accuracy is not None:
            # Extract method from experiment name
            method = entry.experiment.split("_")[0]
            if entry.benchmark not in accuracy_by_method[method]:
                accuracy_by_method[method][entry.benchmark] = []
            accuracy_by_method[method][entry.benchmark].append(entry.accuracy)

    # Average accuracies
    for method in accuracy_by_method:
        for benchmark in accuracy_by_method[method]:
            accs = accuracy_by_method[method][benchmark]
            accuracy_by_method[method][benchmark] = sum(accs) / len(accs)

    # Estimate timing
    elapsed_time = None
    estimated_remaining = None

    if status_data:
        start_time = status_data.get("start_time")
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                elapsed_time = (datetime.now() - start_dt).total_seconds()

                if len(completed) > 0:
                    avg_time = elapsed_time / len(completed)
                    estimated_remaining = avg_time * pending
            except Exception:
                pass

    return StatusSummary(
        total_expected=total_expected,
        completed=len(completed),
        failed=len(failed),
        pending=pending,
        accuracy_by_method=dict(accuracy_by_method),
        entries=entries,
        elapsed_time=elapsed_time,
        estimated_remaining=estimated_remaining,
    )


def print_status(
    summary: StatusSummary,
    verbose: bool = False,
    filter_methods: Optional[List[str]] = None,
):
    """Print status summary to console."""
    print("\n" + "=" * 70)
    print("EVALUATION STATUS")
    print("=" * 70)

    # Progress bar
    total = summary.total_expected
    done = summary.completed + summary.failed
    pct = 100 * done / total if total > 0 else 0

    bar_width = 40
    filled = int(bar_width * done / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"\nProgress: [{bar}] {pct:.1f}%")
    print(f"\n{'Total expected:':<20} {total}")
    print(f"{'Completed:':<20} {summary.completed} ✓")
    print(f"{'Failed:':<20} {summary.failed} ✗")
    print(f"{'Pending:':<20} {summary.pending} ○")

    # Timing
    if summary.elapsed_time:
        elapsed_min = summary.elapsed_time / 60
        print(f"\n{'Elapsed time:':<20} {elapsed_min:.1f} min")

        if summary.estimated_remaining:
            remaining_min = summary.estimated_remaining / 60
            eta = datetime.now().timestamp() + summary.estimated_remaining
            eta_str = datetime.fromtimestamp(eta).strftime("%H:%M")
            print(f"{'Estimated remaining:':<20} {remaining_min:.1f} min (ETA: {eta_str})")

    # Accuracy by method
    if summary.accuracy_by_method:
        print("\n" + "-" * 70)
        print("ACCURACY BY METHOD")
        print("-" * 70)

        methods = summary.accuracy_by_method.keys()
        if filter_methods:
            methods = [m for m in methods if m.lower() in [f.lower() for f in filter_methods]]

        for method in sorted(methods):
            print(f"\n{method}:")
            for benchmark, acc in sorted(summary.accuracy_by_method[method].items()):
                print(f"  {benchmark:<20} {acc:.4f}")

    # Failures
    failed_entries = [e for e in summary.entries if e.status == "failed"]
    if failed_entries:
        print("\n" + "-" * 70)
        print(f"FAILED EVALUATIONS ({len(failed_entries)})")
        print("-" * 70)

        for entry in failed_entries[:10]:  # Show first 10
            print(f"\n  ✗ {entry.experiment} on {entry.benchmark}")
            if entry.error and verbose:
                print(f"    Error: {entry.error[:100]}")

        if len(failed_entries) > 10:
            print(f"\n  ... and {len(failed_entries) - 10} more")

    # Verbose: show all completed
    if verbose:
        completed_entries = [e for e in summary.entries if e.status == "completed"]
        if completed_entries:
            print("\n" + "-" * 70)
            print(f"COMPLETED EVALUATIONS ({len(completed_entries)})")
            print("-" * 70)

            for entry in sorted(completed_entries, key=lambda e: e.experiment):
                acc_str = f"{entry.accuracy:.4f}" if entry.accuracy else "N/A"
                print(f"  ✓ {entry.experiment:<30} {entry.benchmark:<15} {acc_str}")

    print("\n" + "=" * 70)


def print_json_status(summary: StatusSummary):
    """Print status as JSON."""
    data = {
        "total_expected": summary.total_expected,
        "completed": summary.completed,
        "failed": summary.failed,
        "pending": summary.pending,
        "progress_pct": 100 * (summary.completed + summary.failed) / summary.total_expected if summary.total_expected > 0 else 0,
        "elapsed_seconds": summary.elapsed_time,
        "estimated_remaining_seconds": summary.estimated_remaining,
        "accuracy_by_method": summary.accuracy_by_method,
        "failed_evaluations": [
            {"experiment": e.experiment, "benchmark": e.benchmark, "error": e.error}
            for e in summary.entries if e.status == "failed"
        ],
    }
    print(json.dumps(data, indent=2))


def watch_status(
    results_dir: str,
    interval: int = 30,
    filter_methods: Optional[List[str]] = None,
):
    """Watch status and refresh periodically."""
    print(f"Watching evaluation status (refresh every {interval}s, Ctrl+C to stop)")

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')

            # Load and display
            entries = scan_results_directory(results_dir)
            status_data = load_status_file(results_dir)
            summary = compute_summary(entries, status_data)
            print_status(summary, verbose=False, filter_methods=filter_methods)

            print(f"\nLast updated: {datetime.now().strftime('%H:%M:%S')}")
            print(f"Refreshing in {interval}s... (Ctrl+C to stop)")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check evaluation status for PeRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results_dir", "-r",
        type=str,
        required=True,
        help="Directory containing evaluation results",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information",
    )

    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch mode: refresh periodically",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Watch interval in seconds (default: 30)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter by methods (comma-separated)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    filter_methods = None
    if args.filter:
        filter_methods = [m.strip() for m in args.filter.split(",")]

    if args.watch:
        watch_status(args.results_dir, args.interval, filter_methods)
        return

    # Load data
    entries = scan_results_directory(args.results_dir)
    status_data = load_status_file(args.results_dir)
    summary = compute_summary(entries, status_data)

    if args.json:
        print_json_status(summary)
    else:
        print_status(summary, verbose=args.verbose, filter_methods=filter_methods)


if __name__ == "__main__":
    main()
