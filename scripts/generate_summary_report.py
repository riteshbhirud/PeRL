#!/usr/bin/env python3
"""
Summary Report Generator for PeRL Experiments.

Generates a comprehensive markdown summary of experiment results.

Usage:
    python scripts/generate_summary_report.py --output_dir output/core_1.5B
    python scripts/generate_summary_report.py --output_dir output/core_1.5B --output SUMMARY.md
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Result information for a single experiment."""
    name: str
    status: str
    output_dir: str
    duration_minutes: Optional[float] = None
    size_gb: Optional[float] = None
    checkpoint_count: int = 0
    final_loss: Optional[float] = None
    final_step: Optional[int] = None
    peft_type: Optional[str] = None
    rank: Optional[int] = None
    seed: Optional[int] = None
    learning_rate: Optional[float] = None
    error_type: Optional[str] = None
    wandb_url: Optional[str] = None


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
    return total_size / (1024 ** 3)


def parse_experiment_name(name: str) -> Dict[str, Any]:
    """Parse experiment name for PEFT type, rank, and seed."""
    result = {'peft_type': None, 'rank': None, 'seed': None}

    # Common patterns: lora_r16_s42, dora_seed42, pissa_r32_seed123
    import re

    # Extract PEFT type
    peft_types = ['lora', 'dora', 'adalora', 'pissa', 'milora', 'vera', 'lora_plus', 'miss']
    for peft_type in peft_types:
        if peft_type in name.lower():
            result['peft_type'] = peft_type
            break

    # Extract rank
    rank_match = re.search(r'_r(\d+)', name)
    if rank_match:
        result['rank'] = int(rank_match.group(1))

    # Extract seed
    seed_match = re.search(r'_s(\d+)|seed(\d+)', name)
    if seed_match:
        result['seed'] = int(seed_match.group(1) or seed_match.group(2))

    return result


def scan_experiment(exp_dir: str, exp_name: str) -> ExperimentResult:
    """Scan a single experiment for result information."""
    result = ExperimentResult(
        name=exp_name,
        status='not_started',
        output_dir=exp_dir,
    )

    # Parse name for metadata
    parsed = parse_experiment_name(exp_name)
    result.peft_type = parsed['peft_type']
    result.rank = parsed['rank']
    result.seed = parsed['seed']

    if not os.path.exists(exp_dir):
        return result

    result.size_gb = get_directory_size(exp_dir)

    # Count checkpoints
    checkpoint_dirs = glob.glob(os.path.join(exp_dir, 'checkpoint-*'))
    result.checkpoint_count = len(checkpoint_dirs)

    # Check for completion markers
    final_markers = [
        os.path.join(exp_dir, 'adapter_model.safetensors'),
        os.path.join(exp_dir, 'adapter_model.bin'),
        os.path.join(exp_dir, 'pytorch_model.bin'),
        os.path.join(exp_dir, 'model.safetensors'),
    ]
    has_final = any(os.path.exists(m) for m in final_markers)

    # Read experiment metadata
    metadata_path = os.path.join(exp_dir, 'experiment_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            result.status = metadata.get('status', result.status)
            if 'duration_seconds' in metadata:
                result.duration_minutes = metadata['duration_seconds'] / 60
            result.peft_type = metadata.get('peft_type', result.peft_type)
            result.rank = metadata.get('rank', result.rank)
            result.seed = metadata.get('seed', result.seed)
            result.learning_rate = metadata.get('learning_rate', result.learning_rate)
            result.error_type = metadata.get('error_type')
            result.wandb_url = metadata.get('wandb_url')
        except:
            pass

    # Parse log for final metrics
    log_paths = [
        os.path.join(exp_dir, 'training.log'),
        os.path.join(exp_dir, 'experiment.log'),
    ]

    for log_path in log_paths:
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', errors='ignore') as f:
                    lines = f.readlines()

                import re
                for line in reversed(lines[-200:]):
                    if result.final_loss is None:
                        loss_match = re.search(r"['\"]loss['\"]:\s*([\d.]+)", line)
                        if loss_match:
                            result.final_loss = float(loss_match.group(1))

                    if result.final_step is None:
                        step_match = re.search(r"['\"]global_step['\"]:\s*(\d+)", line)
                        if step_match:
                            result.final_step = int(step_match.group(1))

                    if result.final_loss and result.final_step:
                        break
            except:
                pass
            break

    # Determine final status
    if result.status == 'not_started':
        if has_final:
            result.status = 'completed'
        elif result.error_type or (result.checkpoint_count > 0 and not has_final):
            # Check if still running
            if result.size_gb and result.size_gb > 0.001:
                mod_time = os.path.getmtime(exp_dir)
                if (datetime.now() - datetime.fromtimestamp(mod_time)).total_seconds() < 600:
                    result.status = 'running'
                else:
                    result.status = 'failed'

    return result


def scan_all_experiments(output_dir: str) -> List[ExperimentResult]:
    """Scan all experiments in output directory."""
    results = []

    if not os.path.exists(output_dir):
        return results

    # Get experiment list
    experiment_names = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item not in ['logs', 'wandb', '__pycache__']:
            experiment_names.append(item)

    for exp_name in sorted(experiment_names):
        exp_dir = os.path.join(output_dir, exp_name)
        result = scan_experiment(exp_dir, exp_name)
        results.append(result)

    return results


def format_duration(minutes: Optional[float]) -> str:
    """Format duration in human-readable form."""
    if minutes is None:
        return "-"
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    return f"{hours:.1f} hr"


def format_size(gb: Optional[float]) -> str:
    """Format size in human-readable form."""
    if gb is None or gb < 0.001:
        return "-"
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.1f} GB"


def format_loss(loss: Optional[float]) -> str:
    """Format loss value."""
    if loss is None:
        return "-"
    return f"{loss:.4f}"


def generate_markdown_report(results: List[ExperimentResult], output_dir: str) -> str:
    """Generate markdown summary report."""
    now = datetime.now()

    # Calculate statistics
    completed = [r for r in results if r.status == 'completed']
    running = [r for r in results if r.status == 'running']
    failed = [r for r in results if r.status == 'failed']
    not_started = [r for r in results if r.status == 'not_started']

    total_size = sum(r.size_gb or 0 for r in results)
    total_duration = sum(r.duration_minutes or 0 for r in completed)
    avg_duration = total_duration / len(completed) if completed else 0

    # Start building markdown
    lines = []
    lines.append(f"# Experiment Summary: {os.path.basename(output_dir)}")
    lines.append("")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Status:** {len(completed)}/{len(results)} complete ({len(completed)/len(results)*100:.1f}%)" if results else "No experiments found")
    lines.append("")

    # Progress section
    lines.append("## Progress")
    lines.append("")
    if results:
        progress = len(completed) / len(results)
        filled = int(20 * progress)
        bar = '[' + '#' * filled + '-' * (20 - filled) + ']'
        lines.append(f"```")
        lines.append(f"{bar} {progress * 100:.1f}%")
        lines.append(f"```")
        lines.append("")
        lines.append(f"| Status | Count |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Completed | {len(completed)} |")
        lines.append(f"| Running | {len(running)} |")
        lines.append(f"| Failed | {len(failed)} |")
        lines.append(f"| Not Started | {len(not_started)} |")
        lines.append(f"| **Total** | **{len(results)}** |")
    lines.append("")

    # Completed experiments
    if completed:
        lines.append("## Completed Experiments")
        lines.append("")
        lines.append("| Experiment | Duration | Final Loss | Step | Checkpoints | Size |")
        lines.append("|------------|----------|------------|------|-------------|------|")
        for r in sorted(completed, key=lambda x: x.name):
            lines.append(f"| {r.name} | {format_duration(r.duration_minutes)} | {format_loss(r.final_loss)} | {r.final_step or '-'} | {r.checkpoint_count} | {format_size(r.size_gb)} |")
        lines.append("")

        # Best results
        if any(r.final_loss for r in completed):
            best = min((r for r in completed if r.final_loss), key=lambda x: x.final_loss)
            lines.append(f"**Best final loss:** {best.name} ({best.final_loss:.4f})")
            lines.append("")

    # Running experiments
    if running:
        lines.append("## Running Experiments")
        lines.append("")
        for r in running:
            step_info = f" (step {r.final_step})" if r.final_step else ""
            lines.append(f"- **{r.name}**{step_info}")
        lines.append("")

    # Failed experiments
    if failed:
        lines.append("## Failed Experiments")
        lines.append("")
        lines.append("| Experiment | Error | Checkpoints |")
        lines.append("|------------|-------|-------------|")
        for r in sorted(failed, key=lambda x: x.name):
            error = r.error_type or "Unknown"
            lines.append(f"| {r.name} | {error} | {r.checkpoint_count} |")
        lines.append("")
        lines.append(f"Run `python scripts/diagnose_failures.py --output_dir {output_dir}` for detailed diagnosis.")
        lines.append("")

    # Not started
    if not_started:
        lines.append("## Pending Experiments")
        lines.append("")
        lines.append(f"{len(not_started)} experiments not yet started:")
        lines.append("")
        for r in sorted(not_started, key=lambda x: x.name)[:10]:
            lines.append(f"- {r.name}")
        if len(not_started) > 10:
            lines.append(f"- ... and {len(not_started) - 10} more")
        lines.append("")

    # Statistics section
    lines.append("## Statistics")
    lines.append("")
    lines.append("### Time")
    lines.append("")
    if completed:
        lines.append(f"- Total training time: {format_duration(total_duration)}")
        lines.append(f"- Average per experiment: {format_duration(avg_duration)}")
        if not_started:
            remaining = len(not_started) * avg_duration
            lines.append(f"- Estimated remaining: {format_duration(remaining)}")
    else:
        lines.append("- No completed experiments yet")
    lines.append("")

    lines.append("### Storage")
    lines.append("")
    lines.append(f"- Current usage: {format_size(total_size)}")
    if completed and results:
        avg_size = total_size / len([r for r in results if r.size_gb and r.size_gb > 0.001])
        estimated_total = avg_size * len(results)
        lines.append(f"- Average per experiment: {format_size(avg_size)}")
        lines.append(f"- Estimated total: ~{format_size(estimated_total)}")
    lines.append("")

    # Results by PEFT type
    peft_types = {}
    for r in completed:
        pt = r.peft_type or 'unknown'
        if pt not in peft_types:
            peft_types[pt] = []
        peft_types[pt].append(r)

    if peft_types:
        lines.append("## Results by PEFT Method")
        lines.append("")
        lines.append("| Method | Experiments | Avg Loss | Avg Duration |")
        lines.append("|--------|-------------|----------|--------------|")
        for pt in sorted(peft_types.keys()):
            exps = peft_types[pt]
            losses = [r.final_loss for r in exps if r.final_loss]
            durations = [r.duration_minutes for r in exps if r.duration_minutes]
            avg_loss = sum(losses) / len(losses) if losses else None
            avg_dur = sum(durations) / len(durations) if durations else None
            lines.append(f"| {pt} | {len(exps)} | {format_loss(avg_loss)} | {format_duration(avg_dur)} |")
        lines.append("")

    # WandB links
    wandb_urls = [r for r in completed if r.wandb_url]
    if wandb_urls:
        lines.append("## WandB Links")
        lines.append("")
        for r in wandb_urls[:5]:
            lines.append(f"- [{r.name}]({r.wandb_url})")
        if len(wandb_urls) > 5:
            lines.append(f"- ... and {len(wandb_urls) - 5} more")
        lines.append("")

    # Issues section
    lines.append("## Issues")
    lines.append("")
    if failed:
        lines.append(f"- {len(failed)} experiment(s) failed - see above for details")
    else:
        lines.append("None detected")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `scripts/generate_summary_report.py`*")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary report for PeRL experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report
  python scripts/generate_summary_report.py --output_dir output/core_1.5B

  # Save to specific file
  python scripts/generate_summary_report.py --output_dir output/core_1.5B --output report.md

  # Print to stdout only
  python scripts/generate_summary_report.py --output_dir output/core_1.5B --stdout
"""
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory containing experiments"
    )
    parser.add_argument(
        "--output", "-f",
        type=str,
        default="SUMMARY.md",
        help="Output filename (default: SUMMARY.md in output_dir)"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout only, don't save file"
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

    # Scan experiments
    results = scan_all_experiments(args.output_dir)

    if not results:
        print(f"No experiments found in: {args.output_dir}")
        sys.exit(1)

    # Generate report
    report = generate_markdown_report(results, args.output_dir)

    if args.stdout:
        print(report)
    else:
        # Determine output path
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            output_path = os.path.join(args.output_dir, args.output)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"Summary report generated: {output_path}")

        # Also print to console
        print("\n" + "=" * 60)
        print(report)


if __name__ == "__main__":
    main()
