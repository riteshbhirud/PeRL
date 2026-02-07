"""
Benchmark Dataset Loaders for PeRL Evaluation.

Supports loading various math benchmarks with standardized Problem format.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger("perl.evaluation.benchmarks")


@dataclass
class Problem:
    """Standardized problem format for evaluation."""
    id: str
    question: str
    answer: str
    difficulty: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark dataset."""
    name: str
    dataset_path: str
    split: str = "test"
    question_field: str = "problem"
    answer_field: str = "answer"
    id_field: Optional[str] = None
    difficulty_field: Optional[str] = None
    category_field: Optional[str] = None
    subset: Optional[str] = None
    max_problems: Optional[int] = None
    system_prompt: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Default system prompt for math evaluation
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that solves math problems step by step.
Think through the problem carefully, showing your work.
Put your final answer in \\boxed{} format at the end."""


# Supported benchmark configurations
SUPPORTED_BENCHMARKS: Dict[str, BenchmarkConfig] = {
    # AIME Benchmarks
    "aime2024": BenchmarkConfig(
        name="AIME 2024",
        dataset_path="AI-MO/aimo-validation-aime",
        split="train",
        question_field="problem",
        answer_field="answer",
        id_field="url",
    ),
    "aime2025": BenchmarkConfig(
        name="AIME 2025",
        dataset_path="yentinglin/aime_2025",
        split="train",
        question_field="problem",
        answer_field="answer",
    ),
    "aime": BenchmarkConfig(
        name="AIME (AI-MO)",
        dataset_path="AI-MO/aimo-validation-aime",
        split="train",
        question_field="problem",
        answer_field="answer",
    ),

    # MATH Benchmarks
    "math500": BenchmarkConfig(
        name="MATH-500",
        dataset_path="HuggingFaceH4/MATH-500",
        split="test",
        question_field="problem",
        answer_field="answer",
        category_field="type",
        difficulty_field="level",
    ),
    "math": BenchmarkConfig(
        name="MATH",
        dataset_path="hendrycks/competition_math",
        split="test",
        question_field="problem",
        answer_field="solution",
        category_field="type",
        difficulty_field="level",
    ),
    "math_train": BenchmarkConfig(
        name="MATH Training",
        dataset_path="hendrycks/competition_math",
        split="train",
        question_field="problem",
        answer_field="solution",
        category_field="type",
        difficulty_field="level",
    ),

    # Competition Math
    "amc2023": BenchmarkConfig(
        name="AMC 2023",
        dataset_path="zwhe99/amc23",
        split="test",
        question_field="problem",
        answer_field="answer",
    ),
    "hmmt2025": BenchmarkConfig(
        name="HMMT 2025",
        dataset_path="FlagEval/HMMT_2025",
        split="train",
        question_field="problem",
        answer_field="answer",
    ),

    # Other Benchmarks
    "minerva": BenchmarkConfig(
        name="Minerva Math",
        dataset_path="math-ai/minervamath",
        split="test",
        question_field="problem",
        answer_field="answer",
    ),
    "gsm8k": BenchmarkConfig(
        name="GSM8K",
        dataset_path="gsm8k",
        subset="main",
        split="test",
        question_field="question",
        answer_field="answer",
    ),
    "olympiad": BenchmarkConfig(
        name="OlympiadBench",
        dataset_path="knoveleng/OlympiadBench",
        split="test",
        question_field="problem",
        answer_field="answer",
    ),
}


def get_benchmark_config(benchmark_name: str) -> BenchmarkConfig:
    """
    Get benchmark configuration by name.

    Args:
        benchmark_name: Name of the benchmark (case-insensitive)

    Returns:
        BenchmarkConfig for the benchmark
    """
    name_lower = benchmark_name.lower().replace("-", "").replace("_", "")

    # Direct match
    if benchmark_name.lower() in SUPPORTED_BENCHMARKS:
        return SUPPORTED_BENCHMARKS[benchmark_name.lower()]

    # Try normalized match
    for key, config in SUPPORTED_BENCHMARKS.items():
        if key.replace("_", "").replace("-", "") == name_lower:
            return config

    # Might be a HuggingFace dataset path
    if "/" in benchmark_name:
        logger.info(f"Using {benchmark_name} as HuggingFace dataset path")
        return BenchmarkConfig(
            name=benchmark_name.split("/")[-1],
            dataset_path=benchmark_name,
            split="test",
            question_field="problem",
            answer_field="answer",
        )

    raise ValueError(
        f"Unknown benchmark: {benchmark_name}. "
        f"Available: {', '.join(SUPPORTED_BENCHMARKS.keys())}"
    )


def load_benchmark(
    benchmark: Union[str, BenchmarkConfig],
    max_problems: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Problem]:
    """
    Load a benchmark dataset.

    Args:
        benchmark: Benchmark name or config
        max_problems: Maximum number of problems to load
        shuffle: Whether to shuffle problems
        seed: Random seed for shuffling

    Returns:
        List of Problem objects
    """
    if isinstance(benchmark, str):
        # Check if it's a local file
        if os.path.exists(benchmark):
            return load_local_dataset(benchmark, max_problems)
        config = get_benchmark_config(benchmark)
    else:
        config = benchmark

    logger.info(f"Loading benchmark: {config.name}")
    logger.info(f"  Dataset: {config.dataset_path}")
    logger.info(f"  Split: {config.split}")

    # Load from HuggingFace
    problems = load_huggingface_dataset(config, max_problems, shuffle, seed)

    logger.info(f"Loaded {len(problems)} problems")

    return problems


def load_huggingface_dataset(
    config: BenchmarkConfig,
    max_problems: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Problem]:
    """Load dataset from HuggingFace Hub."""
    from datasets import load_dataset

    # Load dataset
    load_kwargs = {"split": config.split}
    if config.subset:
        load_kwargs["name"] = config.subset

    try:
        dataset = load_dataset(config.dataset_path, **load_kwargs)
    except Exception as e:
        logger.error(f"Failed to load dataset {config.dataset_path}: {e}")
        raise

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    # Limit if requested
    if max_problems and max_problems < len(dataset):
        dataset = dataset.select(range(max_problems))

    # Convert to Problem objects
    problems = []
    for idx, item in enumerate(dataset):
        # Extract question
        question = item.get(config.question_field, "")
        if not question:
            logger.warning(f"Empty question at index {idx}, skipping")
            continue

        # Extract answer
        answer = str(item.get(config.answer_field, ""))

        # For MATH dataset, extract answer from solution
        if config.dataset_path == "hendrycks/competition_math":
            answer = extract_answer_from_math_solution(answer)

        # Extract ID
        if config.id_field and config.id_field in item:
            problem_id = str(item[config.id_field])
        else:
            problem_id = f"{config.name.lower().replace(' ', '_')}_{idx}"

        # Extract difficulty
        difficulty = None
        if config.difficulty_field and config.difficulty_field in item:
            difficulty = str(item[config.difficulty_field])

        # Extract category
        category = None
        if config.category_field and config.category_field in item:
            category = str(item[config.category_field])

        # Build metadata
        metadata = {}
        for key, value in item.items():
            if key not in [config.question_field, config.answer_field]:
                try:
                    metadata[key] = value
                except Exception:
                    pass

        problem = Problem(
            id=problem_id,
            question=question,
            answer=answer,
            difficulty=difficulty,
            category=category,
            source=config.name,
            metadata=metadata,
        )
        problems.append(problem)

    return problems


def extract_answer_from_math_solution(solution: str) -> str:
    """
    Extract the final answer from a MATH dataset solution.

    The MATH dataset has full solutions with the answer in \\boxed{}.
    """
    # Look for \boxed{...}
    match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
    if match:
        return match.group(1)

    # Fallback: return the last line
    lines = solution.strip().split('\n')
    return lines[-1] if lines else solution


def load_local_dataset(
    path: str,
    max_problems: Optional[int] = None,
) -> List[Problem]:
    """
    Load dataset from local JSON or JSONL file.

    Expected format:
    - JSON: List of objects with "question" and "answer" fields
    - JSONL: One object per line with "question" and "answer" fields
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    logger.info(f"Loading local dataset: {path}")

    problems = []

    if path.suffix == ".jsonl":
        with open(path) as f:
            for idx, line in enumerate(f):
                if max_problems and idx >= max_problems:
                    break
                try:
                    item = json.loads(line)
                    problems.append(_dict_to_problem(item, idx, path.stem))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {idx}")
                    continue

    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            for idx, item in enumerate(data):
                if max_problems and idx >= max_problems:
                    break
                problems.append(_dict_to_problem(item, idx, path.stem))
        else:
            problems.append(_dict_to_problem(data, 0, path.stem))

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Loaded {len(problems)} problems from {path}")

    return problems


def _dict_to_problem(item: dict, idx: int, source: str) -> Problem:
    """Convert a dictionary to a Problem object."""
    # Try different field names for question
    question = (
        item.get("question") or
        item.get("problem") or
        item.get("prompt") or
        item.get("input") or
        ""
    )

    # Try different field names for answer
    answer = str(
        item.get("answer") or
        item.get("solution") or
        item.get("target") or
        item.get("output") or
        ""
    )

    return Problem(
        id=str(item.get("id", f"{source}_{idx}")),
        question=question,
        answer=answer,
        difficulty=item.get("difficulty"),
        category=item.get("category") or item.get("type"),
        source=source,
        metadata={k: v for k, v in item.items()
                  if k not in ["question", "problem", "answer", "solution", "id"]},
    )


def list_supported_benchmarks() -> Dict[str, str]:
    """List all supported benchmarks with descriptions."""
    return {
        name: config.name
        for name, config in SUPPORTED_BENCHMARKS.items()
    }


def format_problem_prompt(
    problem: Problem,
    system_prompt: Optional[str] = None,
    few_shot_examples: Optional[List[Problem]] = None,
) -> str:
    """
    Format a problem into a prompt for the model.

    Args:
        problem: The problem to format
        system_prompt: Optional system prompt to prepend
        few_shot_examples: Optional few-shot examples to include

    Returns:
        Formatted prompt string
    """
    parts = []

    # Add system prompt
    if system_prompt:
        parts.append(system_prompt)
    else:
        parts.append(DEFAULT_SYSTEM_PROMPT)

    parts.append("")

    # Add few-shot examples
    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(f"Problem: {ex.question}")
            parts.append(f"Solution: The answer is \\boxed{{{ex.answer}}}")
            parts.append("")

    # Add the actual problem
    parts.append(f"Problem: {problem.question}")
    parts.append("Solution:")

    return "\n".join(parts)
