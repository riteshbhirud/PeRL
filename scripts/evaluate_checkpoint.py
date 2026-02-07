#!/usr/bin/env python3
"""
Problem-Level Checkpoint Evaluation for PeRL.

Evaluates a PEFT checkpoint on math benchmarks with detailed per-problem results.

Usage:
    # Evaluate on single benchmark
    python scripts/evaluate_checkpoint.py \
        --checkpoint outputs/lora_r16_s42/checkpoint-1024 \
        --benchmark aime2024 \
        --output_dir results/evaluations

    # Evaluate on multiple benchmarks
    python scripts/evaluate_checkpoint.py \
        --checkpoint outputs/lora_r16_s42/checkpoint-1024 \
        --benchmarks "aime2024,math500" \
        --output_dir results/evaluations

    # Quick test on subset
    python scripts/evaluate_checkpoint.py \
        --checkpoint outputs/lora_r16_s42/checkpoint-1024 \
        --benchmark aime2024 \
        --max_problems 10

    # With custom generation settings
    python scripts/evaluate_checkpoint.py \
        --checkpoint outputs/lora_r16_s42/checkpoint-1024 \
        --benchmark math500 \
        --temperature 0.0 \
        --max_tokens 4096
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perl.evaluation import (
    load_checkpoint,
    load_benchmark,
    extract_answer,
    extract_reasoning_steps,
    check_correctness,
    save_results,
    print_summary,
    CheckpointInfo,
    Problem,
    ProblemResult,
    EvaluationResult,
    EvaluationMetadata,
    EvaluationStatistics,
    SUPPORTED_BENCHMARKS,
)
from perl.evaluation.answer_extraction import (
    AnswerExtractor,
    count_reasoning_tokens,
)
from perl.evaluation.correctness_checking import CorrectnessChecker
from perl.evaluation.result_saving import compute_statistics, generate_output_path

logger = logging.getLogger("perl.evaluation")


# =============================================================================
# Progress Tracking
# =============================================================================

class ProgressTracker:
    """Simple progress tracker with optional tqdm support."""

    def __init__(
        self,
        total: int,
        description: str = "Evaluating",
        use_tqdm: bool = True,
    ):
        self.total = total
        self.description = description
        self.current = 0
        self.correct = 0
        self.start_time = time.time()
        self.times = []

        self.use_tqdm = use_tqdm
        self.pbar = None

        if use_tqdm:
            try:
                from tqdm import tqdm
                self.pbar = tqdm(
                    total=total,
                    desc=description,
                    unit="problem",
                    dynamic_ncols=True,
                )
            except ImportError:
                self.use_tqdm = False

    def update(self, is_correct: bool, gen_time: float):
        """Update progress with a result."""
        self.current += 1
        self.times.append(gen_time)
        if is_correct:
            self.correct += 1

        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({
                "acc": f"{self.correct}/{self.current} ({100*self.correct/self.current:.1f}%)",
                "avg_t": f"{sum(self.times)/len(self.times):.1f}s",
            })
        elif self.current % 10 == 0 or self.current == self.total:
            self._print_progress()

    def _print_progress(self):
        """Print progress without tqdm."""
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current if self.current > 0 else 0
        remaining = avg_time * (self.total - self.current)

        pct = 100 * self.current / self.total
        acc = 100 * self.correct / self.current if self.current > 0 else 0

        bar_width = 30
        filled = int(bar_width * self.current / self.total)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"\r[{bar}] {self.current}/{self.total} ({pct:.0f}%) | "
              f"Acc: {self.correct}/{self.current} ({acc:.1f}%) | "
              f"Avg: {avg_time:.1f}s | ETA: {remaining:.0f}s",
              end="", flush=True)

    def close(self):
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()
        elif self.current > 0:
            print()  # New line after progress


# =============================================================================
# Generation
# =============================================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
) -> str:
    """
    Generate a response from the model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        do_sample: Whether to use sampling

    Returns:
        Generated response text
    """
    import torch

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Set up generation config
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    # Decode response (only the generated part)
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def format_prompt(
    problem: Problem,
    system_prompt: Optional[str] = None,
) -> str:
    """Format a problem into a prompt."""
    if system_prompt is None:
        system_prompt = """You are a helpful assistant that solves math problems step by step.
Think through the problem carefully, showing your work.
Put your final answer in \\boxed{} format at the end."""

    return f"{system_prompt}\n\nProblem: {problem.question}\n\nSolution:"


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_checkpoint(
    checkpoint_path: str,
    benchmark_name: str,
    output_dir: str,
    base_model: Optional[str] = None,
    max_problems: Optional[int] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    save_generations: bool = True,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    use_tqdm: bool = True,
    system_prompt: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate a checkpoint on a benchmark.

    Args:
        checkpoint_path: Path to the PEFT checkpoint
        benchmark_name: Name of the benchmark to evaluate
        output_dir: Directory to save results
        base_model: Override base model path
        max_problems: Maximum number of problems to evaluate
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum tokens to generate
        save_generations: Whether to save full model responses
        device_map: Device mapping strategy
        torch_dtype: Model dtype
        use_tqdm: Whether to use tqdm for progress
        system_prompt: Custom system prompt

    Returns:
        EvaluationResult with all problem results and statistics
    """
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Benchmark: {benchmark_name}")

    # Load checkpoint
    model, tokenizer, checkpoint_info = load_checkpoint(
        checkpoint_path=checkpoint_path,
        base_model_path=base_model,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    # Load benchmark
    problems = load_benchmark(
        benchmark=benchmark_name,
        max_problems=max_problems,
    )

    if not problems:
        raise ValueError(f"No problems loaded from {benchmark_name}")

    logger.info(f"Loaded {len(problems)} problems")

    # Initialize components
    extractor = AnswerExtractor()
    checker = CorrectnessChecker()

    # Track results
    problem_results = []

    # Progress tracking
    progress = ProgressTracker(
        total=len(problems),
        description=f"Evaluating {benchmark_name}",
        use_tqdm=use_tqdm,
    )

    # Evaluate each problem
    for problem in problems:
        start_time = time.time()

        try:
            # Format prompt
            prompt = format_prompt(problem, system_prompt)

            # Generate response
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            gen_time = time.time() - start_time

            # Extract answer
            extraction = extractor.extract(response)
            model_answer = extraction.answer

            # Check correctness
            correctness = checker.check(model_answer, problem.answer)

            # Count reasoning
            step_count, _ = extract_reasoning_steps(response)
            token_count = count_reasoning_tokens(response)

            # Create result
            result = ProblemResult(
                problem_id=problem.id,
                question=problem.question,
                ground_truth=problem.answer,
                model_answer=model_answer,
                model_response=response if save_generations else None,
                correct=correctness.correct,
                score=correctness.score,
                reasoning_steps=step_count,
                reasoning_tokens=token_count,
                difficulty=problem.difficulty,
                category=problem.category,
                generation_time=gen_time,
                extraction_method=extraction.method,
                verification_method=correctness.method,
                metadata=problem.metadata,
            )

        except Exception as e:
            logger.error(f"Error evaluating problem {problem.id}: {e}")
            gen_time = time.time() - start_time

            result = ProblemResult(
                problem_id=problem.id,
                question=problem.question,
                ground_truth=problem.answer,
                model_answer=None,
                model_response=None,
                correct=False,
                score=0.0,
                reasoning_steps=0,
                reasoning_tokens=0,
                difficulty=problem.difficulty,
                category=problem.category,
                generation_time=gen_time,
                extraction_method="error",
                verification_method="error",
                metadata={"error": str(e)},
            )

        problem_results.append(result)
        progress.update(result.correct, gen_time)

    progress.close()

    # Compute statistics
    statistics = compute_statistics(problem_results)

    # Create metadata
    metadata = EvaluationMetadata(
        checkpoint=checkpoint_path,
        checkpoint_step=checkpoint_info.step,
        benchmark=benchmark_name,
        model=checkpoint_info.base_model,
        peft_method=checkpoint_info.peft_type,
        rank=checkpoint_info.rank,
        alpha=checkpoint_info.alpha,
        seed=checkpoint_info.seed,
        evaluation_date=datetime.now().isoformat(),
        total_problems=len(problems),
        correct_count=statistics.correct_count,
        accuracy=statistics.accuracy,
        generation_config={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    )

    # Create result object
    evaluation_result = EvaluationResult(
        metadata=metadata,
        results=problem_results,
        statistics=statistics,
    )

    # Save results
    output_path = generate_output_path(
        checkpoint_path=checkpoint_path,
        benchmark_name=benchmark_name,
        output_dir=output_dir,
    )
    save_results(evaluation_result, output_path, save_generations=save_generations)

    # Print summary
    print_summary(evaluation_result)

    return evaluation_result


def evaluate_multiple_benchmarks(
    checkpoint_path: str,
    benchmarks: List[str],
    output_dir: str,
    **kwargs
) -> List[EvaluationResult]:
    """
    Evaluate a checkpoint on multiple benchmarks.

    Args:
        checkpoint_path: Path to the PEFT checkpoint
        benchmarks: List of benchmark names
        output_dir: Directory to save results
        **kwargs: Additional arguments passed to evaluate_checkpoint

    Returns:
        List of EvaluationResult objects
    """
    results = []

    for i, benchmark in enumerate(benchmarks):
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark {i+1}/{len(benchmarks)}: {benchmark}")
        logger.info(f"{'='*60}\n")

        try:
            result = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                benchmark_name=benchmark,
                output_dir=output_dir,
                **kwargs
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate on {benchmark}: {e}")
            continue

    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a PEFT checkpoint on math benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Checkpoint argument (required unless --list_benchmarks)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the PEFT checkpoint directory",
    )

    # Benchmark arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Single benchmark to evaluate (e.g., 'aime2024', 'math500')",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results/evaluations",
        help="Directory to save results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        default=True,
        help="Save full model responses (default: True)",
    )
    parser.add_argument(
        "--no_save_generations",
        action="store_true",
        help="Don't save full model responses",
    )

    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Override base model path",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )

    # Generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate",
    )

    # Control arguments
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate",
    )
    parser.add_argument(
        "--no_tqdm",
        action="store_true",
        help="Disable tqdm progress bar",
    )

    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Utility
    parser.add_argument(
        "--list_benchmarks",
        action="store_true",
        help="List available benchmarks and exit",
    )

    return parser.parse_args()


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # List benchmarks if requested
    if args.list_benchmarks:
        print("\nSupported Benchmarks:")
        print("-" * 40)
        for name, config in SUPPORTED_BENCHMARKS.items():
            print(f"  {name:<15} - {config.name}")
        print("\nYou can also use HuggingFace dataset paths directly.")
        return

    # Validate arguments
    if not args.checkpoint:
        print("Error: Must specify --checkpoint")
        sys.exit(1)

    if not args.benchmark and not args.benchmarks:
        print("Error: Must specify --benchmark or --benchmarks")
        sys.exit(1)

    # Parse benchmarks
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    else:
        benchmarks = [args.benchmark]

    # Handle save_generations
    save_generations = args.save_generations and not args.no_save_generations

    # Run evaluation
    if len(benchmarks) == 1:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            benchmark_name=benchmarks[0],
            output_dir=args.output_dir,
            base_model=args.base_model,
            max_problems=args.max_problems,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            save_generations=save_generations,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            use_tqdm=not args.no_tqdm,
        )
    else:
        evaluate_multiple_benchmarks(
            checkpoint_path=args.checkpoint,
            benchmarks=benchmarks,
            output_dir=args.output_dir,
            base_model=args.base_model,
            max_problems=args.max_problems,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            save_generations=save_generations,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            use_tqdm=not args.no_tqdm,
        )


if __name__ == "__main__":
    main()
