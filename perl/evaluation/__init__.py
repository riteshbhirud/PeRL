"""
PeRL Problem-Level Evaluation Module.

Provides comprehensive evaluation with detailed per-problem results.
Supports multiple benchmarks, PEFT checkpoint loading, and answer verification.

Usage:
    from perl.evaluation import (
        load_checkpoint,
        load_benchmark,
        extract_answer,
        check_correctness,
        save_results,
        EvaluationResult,
        ProblemResult,
    )
"""

from perl.evaluation.checkpoint_loader import (
    load_checkpoint,
    CheckpointInfo,
    get_checkpoint_metadata,
)
from perl.evaluation.benchmark_loaders import (
    load_benchmark,
    BenchmarkConfig,
    Problem,
    SUPPORTED_BENCHMARKS,
)
from perl.evaluation.answer_extraction import (
    extract_answer,
    extract_reasoning_steps,
    AnswerExtractor,
)
from perl.evaluation.correctness_checking import (
    check_correctness,
    CorrectnessChecker,
)
from perl.evaluation.result_saving import (
    save_results,
    load_results,
    print_summary,
    EvaluationResult,
    ProblemResult,
    EvaluationMetadata,
    EvaluationStatistics,
)

__all__ = [
    # Checkpoint loading
    "load_checkpoint",
    "CheckpointInfo",
    "get_checkpoint_metadata",
    # Benchmark loading
    "load_benchmark",
    "BenchmarkConfig",
    "Problem",
    "SUPPORTED_BENCHMARKS",
    # Answer extraction
    "extract_answer",
    "extract_reasoning_steps",
    "AnswerExtractor",
    # Correctness checking
    "check_correctness",
    "CorrectnessChecker",
    # Result saving
    "save_results",
    "load_results",
    "print_summary",
    "EvaluationResult",
    "ProblemResult",
    "EvaluationMetadata",
    "EvaluationStatistics",
]
