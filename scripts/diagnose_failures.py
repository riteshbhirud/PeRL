#!/usr/bin/env python3
"""
Failure Diagnosis Tool for PeRL Experiments.

Analyzes failed experiments to identify error types and suggest fixes.

Usage:
    python scripts/diagnose_failures.py --output_dir output/core_1.5B
    python scripts/diagnose_failures.py --output_dir output/core_1.5B --lines 50
"""

import os
import sys
import json
import argparse
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FailureDiagnosis:
    """Diagnosis information for a failed experiment."""
    name: str
    output_dir: str
    error_type: str
    error_message: str
    suggested_fix: str
    last_log_lines: List[str]
    checkpoint_count: int
    last_step: Optional[int]
    traceback: Optional[str]


# Common error patterns and their fixes
ERROR_PATTERNS = [
    {
        'patterns': [
            r'cuda out of memory',
            r'outofmemoryerror',
            r'cuda error: out of memory',
            r'torch.cuda.outofmemoryerror',
        ],
        'error_type': 'CUDA Out of Memory',
        'suggested_fix': """
1. Reduce batch size: --config.training.per_device_train_batch_size 2
2. Increase gradient accumulation: --config.training.gradient_accumulation_steps 16
3. Use gradient checkpointing: --config.training.gradient_checkpointing true
4. Reduce model precision if not already: --config.model.dtype bfloat16
5. Reduce sequence length: --config.training.max_completion_length 512
"""
    },
    {
        'patterns': [
            r'killed',
            r'sigkill',
            r'signal 9',
        ],
        'error_type': 'Process Killed (likely OOM)',
        'suggested_fix': """
1. This usually indicates system OOM (not GPU OOM)
2. Reduce batch size significantly
3. Check system memory usage with: free -h
4. Consider using DeepSpeed ZeRO-2 or ZeRO-3 for memory optimization
5. Reduce number of workers if using DataLoader workers
"""
    },
    {
        'patterns': [
            r'filenotfounderror',
            r'no such file or directory',
        ],
        'error_type': 'File Not Found',
        'suggested_fix': """
1. Check that the dataset path is correct
2. Verify model checkpoint exists at the specified path
3. Ensure output directory is writable
4. Check for typos in file paths
"""
    },
    {
        'patterns': [
            r'connectionerror',
            r'connection refused',
            r'urlopen error',
            r'httperror',
            r'timeout.*error',
        ],
        'error_type': 'Network/Connection Error',
        'suggested_fix': """
1. Check internet connection
2. Try running with HF_HUB_OFFLINE=1 if model is cached
3. Use a local copy of the dataset
4. Check if HuggingFace Hub is accessible
5. Try setting a longer timeout
"""
    },
    {
        'patterns': [
            r'valueerror.*expected',
            r'valueerror.*shape',
            r'runtimeerror.*size mismatch',
        ],
        'error_type': 'Shape/Value Mismatch',
        'suggested_fix': """
1. Check that model and tokenizer are compatible
2. Verify input sequence lengths
3. Ensure PEFT config matches model architecture
4. Check if target_modules are valid for this model
"""
    },
    {
        'patterns': [
            r'assertionerror',
        ],
        'error_type': 'Assertion Error',
        'suggested_fix': """
1. Check configuration values are in expected ranges
2. Verify dataset format matches expected structure
3. Review any custom reward functions
"""
    },
    {
        'patterns': [
            r'keyerror',
        ],
        'error_type': 'Key Error',
        'suggested_fix': """
1. Check that dataset has required columns
2. Verify config keys match expected names
3. Ensure model config is compatible
"""
    },
    {
        'patterns': [
            r'importerror',
            r'modulenotfounderror',
        ],
        'error_type': 'Import/Module Error',
        'suggested_fix': """
1. Check that all dependencies are installed
2. Run: pip install -r requirements.txt
3. Verify correct Python environment is activated
4. For flash-attention: pip install flash-attn --no-build-isolation
"""
    },
    {
        'patterns': [
            r'typeerror',
        ],
        'error_type': 'Type Error',
        'suggested_fix': """
1. Check argument types in configuration
2. Verify API compatibility with library versions
3. Review any custom functions for type mismatches
"""
    },
    {
        'patterns': [
            r'gradient.*nan',
            r'loss.*nan',
            r'inf.*loss',
        ],
        'error_type': 'Numerical Instability (NaN/Inf)',
        'suggested_fix': """
1. Reduce learning rate: --config.training.learning_rate 1e-6
2. Enable gradient clipping: --config.training.max_grad_norm 1.0
3. Use bfloat16 instead of float16 for better numerical stability
4. Check reward function for extreme values
"""
    },
    {
        'patterns': [
            r'no space left on device',
            r'disk quota exceeded',
        ],
        'error_type': 'Disk Space Error',
        'suggested_fix': """
1. Free up disk space: rm -rf output/*/checkpoint-*  (keep only needed checkpoints)
2. Reduce checkpoint frequency: --config.training.save_steps 500
3. Move output to a different disk with more space
4. Delete old experiments: rm -rf output/old_experiment
"""
    },
]


def find_error_pattern(text: str) -> Tuple[str, str]:
    """Find matching error pattern and return error type and suggested fix."""
    text_lower = text.lower()

    for pattern_group in ERROR_PATTERNS:
        for pattern in pattern_group['patterns']:
            if re.search(pattern, text_lower):
                return pattern_group['error_type'], pattern_group['suggested_fix']

    return 'Unknown Error', """
1. Check the log file for specific error messages
2. Search for the error message online
3. Check library versions match requirements
4. Try running with --config.common.debug true for more verbose output
"""


def extract_traceback(lines: List[str]) -> Optional[str]:
    """Extract Python traceback from log lines."""
    traceback_lines = []
    in_traceback = False

    for line in lines:
        if 'Traceback (most recent call last)' in line:
            in_traceback = True
            traceback_lines = [line]
        elif in_traceback:
            traceback_lines.append(line)
            # End of traceback is usually the error line (not indented after File/line info)
            if line.strip() and not line.startswith(' ') and ':' in line and 'File' not in line:
                break

    if traceback_lines:
        return ''.join(traceback_lines)
    return None


def diagnose_experiment(exp_dir: str, exp_name: str, num_lines: int = 30) -> Optional[FailureDiagnosis]:
    """Diagnose a potentially failed experiment."""
    diagnosis = FailureDiagnosis(
        name=exp_name,
        output_dir=exp_dir,
        error_type='Unknown',
        error_message='No error found',
        suggested_fix='',
        last_log_lines=[],
        checkpoint_count=0,
        last_step=None,
        traceback=None,
    )

    if not os.path.exists(exp_dir):
        return None

    # Count checkpoints
    checkpoint_dirs = glob.glob(os.path.join(exp_dir, 'checkpoint-*'))
    diagnosis.checkpoint_count = len(checkpoint_dirs)

    # Check for completion markers
    final_markers = [
        os.path.join(exp_dir, 'adapter_model.safetensors'),
        os.path.join(exp_dir, 'adapter_model.bin'),
        os.path.join(exp_dir, 'pytorch_model.bin'),
        os.path.join(exp_dir, 'model.safetensors'),
    ]
    if any(os.path.exists(m) for m in final_markers):
        return None  # Not failed - completed successfully

    # Find log file
    log_paths = [
        os.path.join(exp_dir, 'training.log'),
        os.path.join(exp_dir, 'experiment.log'),
        os.path.join(os.path.dirname(exp_dir), 'logs', f'{exp_name}.log'),
    ]

    log_content = ""
    log_path_used = None
    for log_path in log_paths:
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', errors='ignore') as f:
                    log_content = f.read()
                    log_path_used = log_path
                break
            except:
                pass

    if not log_content:
        # Check experiment metadata for failure info
        metadata_path = os.path.join(exp_dir, 'experiment_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if metadata.get('status') == 'failed':
                    diagnosis.error_type = metadata.get('error_type', 'Unknown')
                    diagnosis.error_message = metadata.get('error', 'See metadata file')
                    return diagnosis
                elif metadata.get('status') == 'completed':
                    return None  # Not failed
            except:
                pass

        # Check if still running (recently modified files)
        mod_time = os.path.getmtime(exp_dir)
        if (datetime.now() - datetime.fromtimestamp(mod_time)).total_seconds() < 600:
            return None  # Probably still running

        # No log, no metadata, not recent - probably not started or crashed early
        if diagnosis.checkpoint_count == 0:
            return None  # Not started

        diagnosis.error_type = 'Unknown Crash'
        diagnosis.error_message = 'No log file found, experiment may have crashed early'
        diagnosis.suggested_fix = """
1. Check system logs: dmesg | tail -50
2. Check for GPU errors: nvidia-smi
3. Try running the experiment again with more verbose logging
"""
        return diagnosis

    # Parse log content
    lines = log_content.split('\n')
    diagnosis.last_log_lines = lines[-num_lines:] if len(lines) > num_lines else lines

    # Extract traceback
    diagnosis.traceback = extract_traceback(lines)

    # Find error pattern
    search_text = log_content[-50000:]  # Last 50k chars for error search
    if diagnosis.traceback:
        search_text = diagnosis.traceback + search_text

    diagnosis.error_type, diagnosis.suggested_fix = find_error_pattern(search_text)

    # Extract specific error message
    error_lines = []
    for line in reversed(lines):
        line_lower = line.lower()
        if 'error' in line_lower or 'exception' in line_lower:
            error_lines.insert(0, line.strip())
            if len(error_lines) >= 3:
                break

    diagnosis.error_message = '\n'.join(error_lines) if error_lines else 'See log file for details'

    # Find last training step
    for line in reversed(lines):
        step_match = re.search(r'step\s*[=:]?\s*(\d+)', line.lower())
        if step_match:
            diagnosis.last_step = int(step_match.group(1))
            break
        global_step_match = re.search(r"['\"]global_step['\"]:\s*(\d+)", line)
        if global_step_match:
            diagnosis.last_step = int(global_step_match.group(1))
            break

    return diagnosis


def scan_for_failures(output_dir: str, num_lines: int = 30) -> List[FailureDiagnosis]:
    """Scan output directory for failed experiments."""
    failures = []

    if not os.path.exists(output_dir):
        return failures

    # Get experiment list
    experiment_names = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item not in ['logs', 'wandb', '__pycache__']:
            experiment_names.append(item)

    for exp_name in sorted(experiment_names):
        exp_dir = os.path.join(output_dir, exp_name)
        diagnosis = diagnose_experiment(exp_dir, exp_name, num_lines)
        if diagnosis:
            failures.append(diagnosis)

    return failures


def print_diagnosis(failures: List[FailureDiagnosis], verbose: bool = False):
    """Print failure diagnoses."""
    if not failures:
        print("\033[92m✓ No failures detected!\033[0m")
        print("\nAll experiments either completed successfully, are still running, or haven't started yet.")
        return

    print("=" * 60)
    print(f"Scanning for failed experiments...")
    print("=" * 60)
    print(f"\n\033[91mFound {len(failures)} failure(s):\033[0m\n")

    for i, diagnosis in enumerate(failures, 1):
        print("-" * 60)
        print(f"\n\033[91m{i}. {diagnosis.name}\033[0m")
        print(f"   Output: {diagnosis.output_dir}")
        print(f"   Error:  \033[93m{diagnosis.error_type}\033[0m")

        if diagnosis.last_step:
            print(f"   Last step: {diagnosis.last_step}")
        if diagnosis.checkpoint_count > 0:
            print(f"   Checkpoints saved: {diagnosis.checkpoint_count}")

        # Show error message
        if diagnosis.error_message and diagnosis.error_message != 'No error found':
            print(f"\n   Error details:")
            for line in diagnosis.error_message.split('\n')[:5]:
                print(f"     {line}")

        # Show traceback if available and verbose
        if verbose and diagnosis.traceback:
            print(f"\n   Traceback:")
            for line in diagnosis.traceback.split('\n')[:20]:
                print(f"     {line}")

        # Show last log lines
        if diagnosis.last_log_lines:
            print(f"\n   Last {len(diagnosis.last_log_lines)} lines of log:")
            for line in diagnosis.last_log_lines[-20:]:  # Show last 20 of the stored lines
                # Truncate long lines
                if len(line) > 100:
                    line = line[:97] + '...'
                print(f"     {line}")

        # Show suggested fix
        print(f"\n   \033[92mSuggested fix:\033[0m")
        for line in diagnosis.suggested_fix.strip().split('\n'):
            print(f"   {line}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Group by error type
    error_types = {}
    for f in failures:
        error_types[f.error_type] = error_types.get(f.error_type, 0) + 1

    print("\nFailures by type:")
    for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")

    # Suggest bulk actions
    if 'CUDA Out of Memory' in error_types or 'Process Killed (likely OOM)' in error_types:
        print("\n\033[93mBulk fix suggestion:\033[0m")
        print("  Many experiments failed due to memory issues.")
        print("  Consider running with reduced batch size:")
        print("  --config.training.per_device_train_batch_size 2")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose failed PeRL experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnose failures in output directory
  python scripts/diagnose_failures.py --output_dir output/core_1.5B

  # Show more log lines
  python scripts/diagnose_failures.py --output_dir output/core_1.5B --lines 50

  # Verbose mode (show tracebacks)
  python scripts/diagnose_failures.py --output_dir output/core_1.5B --verbose

  # Output as JSON
  python scripts/diagnose_failures.py --output_dir output/core_1.5B --json
"""
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory containing experiments"
    )
    parser.add_argument(
        "--lines", "-n",
        type=int,
        default=30,
        help="Number of log lines to show (default: 30)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full tracebacks"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
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

    failures = scan_for_failures(args.output_dir, args.lines)

    if args.json:
        output = []
        for f in failures:
            output.append({
                'name': f.name,
                'output_dir': f.output_dir,
                'error_type': f.error_type,
                'error_message': f.error_message,
                'suggested_fix': f.suggested_fix,
                'checkpoint_count': f.checkpoint_count,
                'last_step': f.last_step,
            })
        print(json.dumps(output, indent=2))
    else:
        print_diagnosis(failures, args.verbose)

    # Exit with error code if failures found
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
