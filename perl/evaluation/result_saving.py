"""
Result Saving and Loading for PeRL Evaluation.

Provides standardized JSON format for problem-level evaluation results.
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("perl.evaluation.results")


@dataclass
class ProblemResult:
    """Result for a single problem."""
    problem_id: str
    question: str
    ground_truth: str
    model_answer: Optional[str]
    model_response: Optional[str]  # Full model response (optional)
    correct: bool
    score: float
    reasoning_steps: int
    reasoning_tokens: int
    difficulty: Optional[str]
    category: Optional[str]
    generation_time: float
    extraction_method: str
    verification_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationMetadata:
    """Metadata about the evaluation run."""
    checkpoint: str
    checkpoint_step: Optional[int]
    benchmark: str
    model: str
    peft_method: str
    rank: int
    alpha: int
    seed: Optional[int]
    evaluation_date: str
    total_problems: int
    correct_count: int
    accuracy: float
    generation_config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationStatistics:
    """Aggregate statistics from evaluation."""
    accuracy: float
    correct_count: int
    total_count: int
    avg_reasoning_steps: float
    avg_reasoning_tokens: float
    avg_generation_time: float
    total_generation_time: float
    accuracy_by_difficulty: Dict[str, float] = field(default_factory=dict)
    accuracy_by_category: Dict[str, float] = field(default_factory=dict)
    extraction_method_counts: Dict[str, int] = field(default_factory=dict)
    verification_method_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result with metadata, results, and statistics."""
    metadata: EvaluationMetadata
    results: List[ProblemResult]
    statistics: EvaluationStatistics

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "statistics": self.statistics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        """Create from dictionary."""
        metadata = EvaluationMetadata(**data["metadata"])
        results = [ProblemResult(**r) for r in data["results"]]
        statistics = EvaluationStatistics(**data["statistics"])
        return cls(metadata=metadata, results=results, statistics=statistics)


def compute_statistics(results: List[ProblemResult]) -> EvaluationStatistics:
    """
    Compute aggregate statistics from problem results.

    Args:
        results: List of ProblemResult objects

    Returns:
        EvaluationStatistics with computed metrics
    """
    if not results:
        return EvaluationStatistics(
            accuracy=0.0,
            correct_count=0,
            total_count=0,
            avg_reasoning_steps=0.0,
            avg_reasoning_tokens=0.0,
            avg_generation_time=0.0,
            total_generation_time=0.0,
        )

    total = len(results)
    correct = sum(1 for r in results if r.correct)

    # Compute averages
    avg_steps = sum(r.reasoning_steps for r in results) / total
    avg_tokens = sum(r.reasoning_tokens for r in results) / total
    total_time = sum(r.generation_time for r in results)
    avg_time = total_time / total

    # Accuracy by difficulty
    by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r.difficulty:
            by_difficulty[r.difficulty]["total"] += 1
            if r.correct:
                by_difficulty[r.difficulty]["correct"] += 1

    accuracy_by_difficulty = {
        d: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for d, stats in by_difficulty.items()
    }

    # Accuracy by category
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r.category:
            by_category[r.category]["total"] += 1
            if r.correct:
                by_category[r.category]["correct"] += 1

    accuracy_by_category = {
        c: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for c, stats in by_category.items()
    }

    # Method counts
    extraction_counts = defaultdict(int)
    verification_counts = defaultdict(int)
    for r in results:
        extraction_counts[r.extraction_method] += 1
        verification_counts[r.verification_method] += 1

    return EvaluationStatistics(
        accuracy=correct / total,
        correct_count=correct,
        total_count=total,
        avg_reasoning_steps=avg_steps,
        avg_reasoning_tokens=avg_tokens,
        avg_generation_time=avg_time,
        total_generation_time=total_time,
        accuracy_by_difficulty=dict(accuracy_by_difficulty),
        accuracy_by_category=dict(accuracy_by_category),
        extraction_method_counts=dict(extraction_counts),
        verification_method_counts=dict(verification_counts),
    )


def save_results(
    evaluation_result: EvaluationResult,
    output_path: Union[str, Path],
    save_generations: bool = True,
    indent: int = 2,
) -> str:
    """
    Save evaluation results to JSON file.

    Args:
        evaluation_result: Complete evaluation result
        output_path: Path to save JSON file
        save_generations: Whether to include full model responses
        indent: JSON indentation level

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = evaluation_result.to_dict()

    # Optionally remove full responses to reduce file size
    if not save_generations:
        for result in data["results"]:
            result["model_response"] = None

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)

    logger.info(f"Saved results to: {output_path}")

    return str(output_path)


def load_results(path: Union[str, Path]) -> EvaluationResult:
    """
    Load evaluation results from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        EvaluationResult object
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return EvaluationResult.from_dict(data)


def generate_output_path(
    checkpoint_path: str,
    benchmark_name: str,
    output_dir: str,
    suffix: str = "",
) -> Path:
    """
    Generate standardized output path for results.

    Args:
        checkpoint_path: Path to the checkpoint
        benchmark_name: Name of the benchmark
        output_dir: Output directory
        suffix: Optional suffix to add

    Returns:
        Path object for output file
    """
    checkpoint_name = Path(checkpoint_path).name

    # Clean benchmark name for filename
    benchmark_clean = benchmark_name.replace("/", "_").replace(" ", "_").lower()

    # Build filename
    filename = f"{checkpoint_name}_{benchmark_clean}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".json"

    return Path(output_dir) / filename


def save_summary_csv(
    results: List[EvaluationResult],
    output_path: Union[str, Path],
):
    """
    Save summary of multiple evaluations to CSV.

    Args:
        results: List of EvaluationResult objects
        output_path: Path to save CSV file
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint",
        "step",
        "benchmark",
        "peft_method",
        "rank",
        "seed",
        "accuracy",
        "correct",
        "total",
        "avg_reasoning_steps",
        "avg_generation_time",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "checkpoint": result.metadata.checkpoint,
                "step": result.metadata.checkpoint_step,
                "benchmark": result.metadata.benchmark,
                "peft_method": result.metadata.peft_method,
                "rank": result.metadata.rank,
                "seed": result.metadata.seed,
                "accuracy": f"{result.statistics.accuracy:.4f}",
                "correct": result.statistics.correct_count,
                "total": result.statistics.total_count,
                "avg_reasoning_steps": f"{result.statistics.avg_reasoning_steps:.1f}",
                "avg_generation_time": f"{result.statistics.avg_generation_time:.2f}",
            })

    logger.info(f"Saved summary CSV to: {output_path}")


def merge_results(
    result_files: List[str],
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Merge multiple result files into a single summary.

    Args:
        result_files: List of paths to result JSON files
        output_path: Optional path to save merged results

    Returns:
        List of merged result dictionaries
    """
    merged = []

    for path in result_files:
        try:
            result = load_results(path)
            merged.append({
                "file": str(path),
                "metadata": result.metadata.to_dict(),
                "statistics": result.statistics.to_dict(),
            })
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info(f"Saved merged results to: {output_path}")

    return merged


def print_summary(result: EvaluationResult):
    """Print a human-readable summary of evaluation results."""
    m = result.metadata
    s = result.statistics

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nCheckpoint: {m.checkpoint}")
    print(f"Benchmark:  {m.benchmark}")
    print(f"PEFT:       {m.peft_method} (rank={m.rank})")
    if m.seed is not None:
        print(f"Seed:       {m.seed}")
    if m.checkpoint_step is not None:
        print(f"Step:       {m.checkpoint_step}")

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Accuracy':<25} {s.accuracy:>14.2%}")
    print(f"{'Correct':<25} {s.correct_count:>15}")
    print(f"{'Total':<25} {s.total_count:>15}")
    print(f"{'Avg Reasoning Steps':<25} {s.avg_reasoning_steps:>15.1f}")
    print(f"{'Avg Reasoning Tokens':<25} {s.avg_reasoning_tokens:>15.1f}")
    print(f"{'Avg Generation Time':<25} {s.avg_generation_time:>14.2f}s")
    print(f"{'Total Generation Time':<25} {s.total_generation_time:>14.2f}s")

    if s.accuracy_by_difficulty:
        print(f"\n{'Accuracy by Difficulty':}")
        for diff, acc in sorted(s.accuracy_by_difficulty.items()):
            print(f"  {diff:<20} {acc:>10.2%}")

    if s.accuracy_by_category:
        print(f"\n{'Accuracy by Category':}")
        for cat, acc in sorted(s.accuracy_by_category.items()):
            print(f"  {cat:<20} {acc:>10.2%}")

    print("=" * 60 + "\n")
