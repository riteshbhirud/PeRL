"""
PeRL Evaluation Module.

This module provides utilities for evaluating trained models on math benchmarks.
"""

from perl.eval.utils import StageContext, setup_logging, merge_model_if_needed
from perl.eval.grader import grade_answer_perl
from perl.eval.vllm import (
    extract_vllm_args,
    start_vllm_processes,
    stop_vllm_processes,
    wait_for_vllm_ready,
    generate_with_vllm_async,
)

__all__ = [
    "StageContext",
    "setup_logging",
    "merge_model_if_needed",
    "grade_answer_perl",
    "extract_vllm_args",
    "start_vllm_processes",
    "stop_vllm_processes",
    "wait_for_vllm_ready",
    "generate_with_vllm_async",
]
