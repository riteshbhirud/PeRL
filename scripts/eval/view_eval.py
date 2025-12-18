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


def extract_benchmark_data(result_data: Dict) -> Tuple[float, float, int, int]:
    """
    Extract accuracy, std, number of problems, and rollout_n from result.json.
    
    Returns:
        (accuracy, std, num_problems, rollout_n)
    """
    acc = result_data['summary']['avg']
    std = result_data['summary']['std']
    num_problems = len(result_data['raw'])
    rollout_n = result_data['rollout_n']
    return acc, std, num_problems, rollout_n


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
        sys.exit(1)
    
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
            acc, std, num_problems, rollout_n = extract_benchmark_data(result_data)
            
            benchmarks.append({
                'name': benchmark_name,
                'acc': acc,
                'std': std,
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


def generate_markdown_table(benchmarks: List[Dict], overall_acc: float) -> str:
    """Generate a markdown table with benchmark results."""
    lines = []
    
    # Table header
    lines.append("| Benchmark | Accuracy | Std | # Problems | N | Total Samples |")
    lines.append("|-----------|----------|-----|------------|---|---------------|")
    
    # Table rows for each benchmark
    for bench in benchmarks:
        lines.append(
            f"| {bench['name']} | "
            f"{bench['acc']:.4f} | "
            f"{bench['std']:.4f} | "
            f"{bench['num_problems']} | "
            f"{bench['rollout_n']} | "
            f"{bench['total_samples']} |"
        )
    
    # Overall accuracy row
    total_samples = sum(b['total_samples'] for b in benchmarks)
    lines.append("")
    lines.append(f"**Overall Accuracy**: {overall_acc:.4f} (based on {total_samples} total samples)")
    
    return "\n".join(lines)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python view_eval.py <folder_path>", file=sys.stderr)
        sys.exit(1)
    
    folder_path = sys.argv[1]
    benchmarks = process_folder(folder_path)
    
    if not benchmarks:
        print("No benchmarks found!", file=sys.stderr)
        sys.exit(1)
    
    overall_acc = calculate_overall_acc(benchmarks)
    markdown_table = generate_markdown_table(benchmarks, overall_acc)
    
    print(markdown_table)


if __name__ == '__main__':
    main()

