"""
Correctness Checking for PeRL Evaluation.

Verifies if model answers match ground truth using multiple strategies:
- Exact string matching
- Numerical comparison with tolerance
- Symbolic/mathematical equivalence via math-verify
"""

import logging
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

logger = logging.getLogger("perl.evaluation.correctness")


@dataclass
class CorrectnessResult:
    """Result of correctness check."""
    correct: bool
    method: str  # Which method determined correctness
    normalized_model: str  # Normalized model answer
    normalized_truth: str  # Normalized ground truth
    score: float  # 0.0 or 1.0 (could be partial in future)
    details: Optional[str] = None  # Additional info


class CorrectnessChecker:
    """
    Configurable correctness checker for different answer types.
    """

    def __init__(
        self,
        use_math_verify: bool = True,
        numerical_tolerance: float = 1e-6,
        timeout_seconds: float = 5.0,
    ):
        """
        Initialize the checker.

        Args:
            use_math_verify: Whether to use math-verify library
            numerical_tolerance: Tolerance for numerical comparison
            timeout_seconds: Timeout for symbolic verification
        """
        self.use_math_verify = use_math_verify
        self.numerical_tolerance = numerical_tolerance
        self.timeout_seconds = timeout_seconds

        # Check if math-verify is available
        self._math_verify_available = False
        if use_math_verify:
            try:
                from math_verify import verify
                self._math_verify_available = True
            except ImportError:
                logger.warning(
                    "math-verify not available, falling back to string matching"
                )

    def check(
        self,
        model_answer: Optional[str],
        ground_truth: str,
    ) -> CorrectnessResult:
        """
        Check if model answer matches ground truth.

        Args:
            model_answer: Extracted model answer (may be None)
            ground_truth: Expected answer

        Returns:
            CorrectnessResult with verdict and details
        """
        # Handle None/empty model answer
        if not model_answer or not model_answer.strip():
            return CorrectnessResult(
                correct=False,
                method="no_answer",
                normalized_model="",
                normalized_truth=self._normalize(ground_truth),
                score=0.0,
                details="No answer extracted from model response",
            )

        # Normalize both answers
        norm_model = self._normalize(model_answer)
        norm_truth = self._normalize(ground_truth)

        # Strategy 1: Exact match after normalization
        if norm_model == norm_truth:
            return CorrectnessResult(
                correct=True,
                method="exact_match",
                normalized_model=norm_model,
                normalized_truth=norm_truth,
                score=1.0,
            )

        # Strategy 2: Numerical comparison
        num_result = self._check_numerical(norm_model, norm_truth)
        if num_result is not None:
            return num_result

        # Strategy 3: Math-verify for symbolic comparison
        if self._math_verify_available:
            mv_result = self._check_math_verify(model_answer, ground_truth)
            if mv_result is not None:
                return mv_result

        # Strategy 4: Fuzzy string matching
        fuzzy_result = self._check_fuzzy(norm_model, norm_truth)
        if fuzzy_result:
            return CorrectnessResult(
                correct=True,
                method="fuzzy_match",
                normalized_model=norm_model,
                normalized_truth=norm_truth,
                score=1.0,
                details="Matched via fuzzy comparison",
            )

        # No match found
        return CorrectnessResult(
            correct=False,
            method="no_match",
            normalized_model=norm_model,
            normalized_truth=norm_truth,
            score=0.0,
            details=f"Model: '{norm_model}' != Truth: '{norm_truth}'",
        )

    def _normalize(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if not answer:
            return ""

        s = str(answer).strip()

        # Remove $ delimiters
        s = re.sub(r'^\$+|\$+$', '', s)

        # Remove \boxed wrapper
        match = re.match(r'^\\boxed\{(.+)\}$', s)
        if match:
            s = match.group(1)

        # Remove whitespace
        s = " ".join(s.split())

        # Lowercase for comparison
        s = s.lower()

        # Remove common LaTeX commands that don't affect value
        s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)

        # Normalize fractions
        s = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', s)
        s = re.sub(r'\\dfrac\{(\d+)\}\{(\d+)\}', r'\1/\2', s)
        s = re.sub(r'\\tfrac\{(\d+)\}\{(\d+)\}', r'\1/\2', s)

        return s

    def _check_numerical(
        self,
        model: str,
        truth: str,
    ) -> Optional[CorrectnessResult]:
        """Check numerical equivalence."""
        try:
            # Try parsing as floats
            model_val = self._parse_number(model)
            truth_val = self._parse_number(truth)

            if model_val is None or truth_val is None:
                return None

            # Check equality with tolerance
            if abs(model_val - truth_val) <= self.numerical_tolerance:
                return CorrectnessResult(
                    correct=True,
                    method="numerical_match",
                    normalized_model=str(model_val),
                    normalized_truth=str(truth_val),
                    score=1.0,
                    details=f"Numerical match: {model_val} ≈ {truth_val}",
                )
            else:
                return CorrectnessResult(
                    correct=False,
                    method="numerical_mismatch",
                    normalized_model=str(model_val),
                    normalized_truth=str(truth_val),
                    score=0.0,
                    details=f"Numerical mismatch: {model_val} != {truth_val}",
                )

        except Exception:
            return None

    def _parse_number(self, s: str) -> Optional[float]:
        """Parse a string as a number, handling fractions."""
        s = s.strip()

        # Remove commas
        s = s.replace(",", "")

        # Handle fractions
        if "/" in s:
            try:
                frac = Fraction(s)
                return float(frac)
            except Exception:
                pass

        # Try direct parsing
        try:
            return float(s)
        except ValueError:
            return None

    def _check_math_verify(
        self,
        model: str,
        truth: str,
    ) -> Optional[CorrectnessResult]:
        """Check using math-verify library."""
        try:
            from perl.rm.math_verifier import compute_score

            score = compute_score(
                prediction=model,
                ground_truth=truth,
                timeout_score=0.0,
            )

            if score >= 0.5:  # math-verify returns 0 or 1
                return CorrectnessResult(
                    correct=True,
                    method="math_verify",
                    normalized_model=model,
                    normalized_truth=truth,
                    score=score,
                    details="Verified via math-verify",
                )
            else:
                return CorrectnessResult(
                    correct=False,
                    method="math_verify",
                    normalized_model=model,
                    normalized_truth=truth,
                    score=score,
                    details="Failed math-verify",
                )

        except Exception as e:
            logger.debug(f"math-verify failed: {e}")
            return None

    def _check_fuzzy(self, model: str, truth: str) -> bool:
        """Check for fuzzy string match."""
        # Remove all whitespace and punctuation
        model_clean = re.sub(r'[^a-z0-9]', '', model.lower())
        truth_clean = re.sub(r'[^a-z0-9]', '', truth.lower())

        if model_clean == truth_clean and model_clean:
            return True

        # Check if one contains the other (for cases like "42" vs "the answer is 42")
        if truth_clean and truth_clean in model_clean:
            # But the model shouldn't be too much longer
            if len(model_clean) <= len(truth_clean) * 2:
                return True

        return False


def check_correctness(
    model_answer: Optional[str],
    ground_truth: str,
    use_math_verify: bool = True,
    numerical_tolerance: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Convenience function to check correctness.

    Args:
        model_answer: Extracted model answer
        ground_truth: Expected answer
        use_math_verify: Whether to use math-verify
        numerical_tolerance: Tolerance for numerical comparison

    Returns:
        Tuple of (is_correct, score)
    """
    checker = CorrectnessChecker(
        use_math_verify=use_math_verify,
        numerical_tolerance=numerical_tolerance,
    )
    result = checker.check(model_answer, ground_truth)
    return result.correct, result.score


def batch_check_correctness(
    model_answers: list,
    ground_truths: list,
    use_math_verify: bool = True,
) -> list:
    """
    Check correctness for a batch of answers.

    Args:
        model_answers: List of extracted model answers
        ground_truths: List of expected answers
        use_math_verify: Whether to use math-verify

    Returns:
        List of CorrectnessResult objects
    """
    if len(model_answers) != len(ground_truths):
        raise ValueError("Mismatched lengths: model_answers vs ground_truths")

    checker = CorrectnessChecker(use_math_verify=use_math_verify)
    results = []

    for model, truth in zip(model_answers, ground_truths):
        result = checker.check(model, truth)
        results.append(result)

    return results


def compute_accuracy(results: list) -> dict:
    """
    Compute accuracy statistics from a list of CorrectnessResult.

    Args:
        results: List of CorrectnessResult objects

    Returns:
        Dictionary with accuracy statistics
    """
    if not results:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "by_method": {},
        }

    total = len(results)
    correct = sum(1 for r in results if r.correct)

    # Count by verification method
    by_method = {}
    for r in results:
        method = r.method
        if method not in by_method:
            by_method[method] = {"correct": 0, "total": 0}
        by_method[method]["total"] += 1
        if r.correct:
            by_method[method]["correct"] += 1

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "by_method": by_method,
    }
