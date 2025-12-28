#!/usr/bin/env python3
"""
Script to read evaluation results from subfolders and generate a markdown table.
Each subfolder represents a benchmark and contains a result.json file.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def read_result_json(file_path: Path) -> Dict:
    """Read and parse a result.json file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_benchmark_data(result_data: Dict) -> Tuple[float, float, float, int, int]:
    """
    Extract accuracy, std, max, number of problems, and rollout_n from result.json.
    
    Returns:
        (accuracy, std, max, num_problems, rollout_n)
    """
    acc = result_data['summary']['avg']
    std = result_data['summary']['std']
    max_val = result_data['summary']['max']
    num_problems = len(result_data['raw'])
    rollout_n = result_data['rollout_n']
    return acc, std, max_val, num_problems, rollout_n


def process_folder(folder_path: str) -> List[Dict]:
    """
    Process a folder containing benchmark subfolders.
    
    Args:
        folder_path: Path to the folder containing benchmark subfolders
        
    Returns:
        List of dictionaries with benchmark data
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist", file=sys.stderr)
        return []
    
    benchmarks = []
    
    # Iterate through subfolders
    for subfolder in sorted(folder.iterdir()):
        if not subfolder.is_dir():
            continue
        
        benchmark_name = subfolder.name
        result_json_path = subfolder / 'result.json'
        
        if not result_json_path.exists():
            print(f"Warning: {result_json_path} not found, skipping {benchmark_name}", file=sys.stderr)
            continue
        
        try:
            result_data = read_result_json(result_json_path)
            acc, std, max_val, num_problems, rollout_n = extract_benchmark_data(result_data)
            
            benchmarks.append({
                'name': benchmark_name,
                'acc': acc,
                'std': std,
                'max': max_val,
                'num_problems': num_problems,
                'rollout_n': rollout_n,
                'total_samples': num_problems * rollout_n
            })
        except Exception as e:
            print(f"Error processing {benchmark_name}: {e}", file=sys.stderr)
            continue
    
    return benchmarks


def calculate_overall_acc(benchmarks: List[Dict]) -> float:
    """
    Calculate overall accuracy as weighted average.
    
    Overall acc = sum(acc_i * (num_problems_i * rollout_n_i)) / sum(num_problems_i * rollout_n_i)
    """
    total_weighted_sum = 0.0
    total_samples = 0
    
    for bench in benchmarks:
        weight = bench['total_samples']
        total_weighted_sum += bench['acc'] * weight
        total_samples += weight
    
    if total_samples == 0:
        return 0.0
    
    return total_weighted_sum / total_samples


def generate_markdown_table(benchmarks: List[Dict], overall_acc: float, folder_name: str = None) -> str:
    """Generate a markdown table with benchmark results."""
    lines = []
    
    # Add folder name as header if provided
    if folder_name:
        lines.append(f"## {folder_name}")
        lines.append("")
    
    # Table header
    lines.append("| Benchmark | Accuracy | Std | Max | # Problems | N | Total Samples |")
    lines.append("|-----------|----------|-----|-----|------------|---|---------------|")
    
    # Table rows for each benchmark
    for bench in benchmarks:
        lines.append(
            f"| {bench['name']} | "
            f"{bench['acc']:.4f} | "
            f"{bench['std']:.4f} | "
            f"{bench['max']:.4f} | "
            f"{bench['num_problems']} | "
            f"{bench['rollout_n']} | "
            f"{bench['total_samples']} |"
        )
    
    # Overall accuracy row
    total_samples = sum(b['total_samples'] for b in benchmarks)
    lines.append("")
    lines.append(f"**Overall Accuracy**: {overall_acc:.4f} (based on {total_samples} total samples)")
    lines.append("")
    
    return "\n".join(lines)


def generate_combined_markdown(all_folder_results: List[Tuple[str, List[Dict], float]], overall_acc: float) -> str:
    """Generate a combined markdown output for multiple folders."""
    lines = []
    
    # Process each folder
    for folder_name, benchmarks, folder_acc in all_folder_results:
        lines.append(generate_markdown_table(benchmarks, folder_acc, folder_name))
    
    # Add overall summary
    lines.append("## Overall Summary (All Folders)")
    lines.append("")
    
    # Collect all benchmarks from all folders
    all_benchmarks = []
    for _, benchmarks, _ in all_folder_results:
        all_benchmarks.extend(benchmarks)
    
    # Calculate total samples across all folders
    total_samples = sum(b['total_samples'] for b in all_benchmarks)
    lines.append(f"**Overall Accuracy (All Folders)**: {overall_acc:.4f} (based on {total_samples} total samples)")
    
    return "\n".join(lines)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python view_eval.py <folder_path1> [folder_path2] ...", file=sys.stderr)
        sys.exit(1)
    
    folder_paths = sys.argv[1:]
    all_folder_results = []
    all_benchmarks = []
    
    # Process each folder
    for folder_path in folder_paths:
        folder = Path(folder_path)
        folder_name = folder.name if folder.exists() else folder_path
        
        benchmarks = process_folder(folder_path)
        
        if not benchmarks:
            print(f"Warning: No benchmarks found in {folder_path}, skipping", file=sys.stderr)
            continue
        
        folder_acc = calculate_overall_acc(benchmarks)
        all_folder_results.append((folder_name, benchmarks, folder_acc))
        all_benchmarks.extend(benchmarks)
    
    if not all_folder_results:
        print("No benchmarks found in any folder!", file=sys.stderr)
        sys.exit(1)
    
    # Calculate overall accuracy across all folders
    overall_acc = calculate_overall_acc(all_benchmarks)
    
    # Generate combined markdown output
    if len(all_folder_results) == 1:
        # Single folder: use simple format
        folder_name, benchmarks, _ = all_folder_results[0]
        markdown_table = generate_markdown_table(benchmarks, overall_acc)
    else:
        # Multiple folders: use combined format
        markdown_table = generate_combined_markdown(all_folder_results, overall_acc)
    
    print(markdown_table)


if __name__ == '__main__':
    main()

