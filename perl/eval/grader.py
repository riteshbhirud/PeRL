"""
Answer grading module for PeRL evaluation.

Uses math-verify library for mathematical answer verification.
"""

import re
import logging
from typing import Tuple, Optional

# Import from existing math_verifier module
from perl.rm.math_verifier import extract_boxed_answer, compute_score

logger = logging.getLogger("perl.eval.grader")


def grade_answer_perl(
    response: str,
    ground_truth: str,
    timeout_score: float = 0.0
) -> Tuple[float, float]:
    """
    Grade a model response against ground truth.

    This function:
    1. Extracts the answer from the response (looking for \\boxed{} or Answer: format)
    2. Computes mathematical equivalence with ground truth
    3. Checks format compliance

    Args:
        response: Model generated response
        ground_truth: Expected answer (can be LaTeX)
        timeout_score: Score to return if verification times out

    Returns:
        Tuple of (accuracy_score, format_score)
        - accuracy_score: 1.0 if answer matches, 0.0 otherwise
        - format_score: 1.0 if response has proper format, 0.0 otherwise
    """
    accuracy_score = 0.0
    format_score = 0.0

    # Try to extract answer from response
    extracted_answer = extract_answer(response)

    if extracted_answer is not None:
        format_score = 1.0

        # Compute accuracy using math-verify
        try:
            accuracy_score = compute_score(
                prediction=extracted_answer,
                ground_truth=ground_truth,
                timeout_score=timeout_score
            )
        except Exception as e:
            logger.debug(f"Grading failed: {e}")
            accuracy_score = 0.0
    else:
        # No answer extracted, check if ground truth appears in response
        format_score = 0.0

        # Fallback: try to match ground truth directly in response
        try:
            accuracy_score = compute_score(
                prediction=response,
                ground_truth=ground_truth,
                timeout_score=timeout_score
            )
        except Exception:
            accuracy_score = 0.0

    return accuracy_score, format_score


def extract_answer(response: str) -> Optional[str]:
    """
    Extract the final answer from a model response.

    Tries multiple extraction patterns in order:
    1. \\boxed{...} or \\fbox{...} (LaTeX boxed format)
    2. Answer: ... (text format)
    3. <answer>...</answer> (XML format)
    4. Therefore, the final answer is: ... (prose format)

    Args:
        response: Model generated response

    Returns:
        Extracted answer string, or None if no answer found
    """
    if not response:
        return None

    # Method 1: LaTeX boxed format
    boxed_answer = extract_boxed_answer(response)
    if boxed_answer:
        return boxed_answer.strip()

    # Method 2: "Answer: ..." format
    answer_match = re.search(
        r'(?:^|\n)\s*(?:answer|final answer|the answer is)[:\s]*(.+?)(?:\n|$)',
        response,
        re.IGNORECASE | re.MULTILINE
    )
    if answer_match:
        answer = answer_match.group(1).strip()
        # Clean up common formatting
        answer = re.sub(r'^\$+|\$+$', '', answer)  # Remove $ delimiters
        answer = re.sub(r'^[.:]?\s*', '', answer)  # Remove leading punctuation
        if answer:
            return answer

    # Method 3: XML <answer>...</answer> format
    xml_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if xml_match:
        answer = xml_match.group(1).strip()
        # Try to extract boxed from within answer tags
        boxed_in_answer = extract_boxed_answer(answer)
        if boxed_in_answer:
            return boxed_in_answer.strip()
        if answer:
            return answer

    # Method 4: "Therefore, the final answer is: ..." format
    therefore_match = re.search(
        r'therefore[,\s]+(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\n|$)',
        response,
        re.IGNORECASE
    )
    if therefore_match:
        answer = therefore_match.group(1).strip()
        # Check for boxed within this section
        boxed_in_therefore = extract_boxed_answer(answer)
        if boxed_in_therefore:
            return boxed_in_therefore.strip()
        # Clean up
        answer = re.sub(r'^\$+|\$+$', '', answer)
        answer = re.sub(r'\.$', '', answer)  # Remove trailing period
        if answer:
            return answer

    return None


def check_format_compliance(response: str) -> float:
    """
    Check if the response follows expected format conventions.

    Returns a score between 0.0 and 1.0 based on format quality.

    Args:
        response: Model generated response

    Returns:
        Format compliance score (0.0 to 1.0)
    """
    score = 0.0

    if not response:
        return 0.0

    # Check for thinking/reasoning section
    has_think_tag = '</think>' in response or '<think>' in response
    has_answer_tag = '</answer>' in response or '<answer>' in response
    has_boxed = '\\boxed' in response or '\\fbox' in response

    # Award points for structure
    if has_think_tag:
        score += 0.3
    if has_answer_tag:
        score += 0.3
    if has_boxed:
        score += 0.4

    # If no special formatting, check for "Answer:" format
    if score == 0.0:
        if re.search(r'(?:^|\n)\s*answer[:\s]', response, re.IGNORECASE):
            score = 0.5

    return min(score, 1.0)


def count_reasoning_steps(response: str) -> int:
    """
    Count the number of reasoning steps in a response.

    Useful for analyzing long-chain reasoning.

    Args:
        response: Model generated response

    Returns:
        Estimated number of reasoning steps
    """
    if not response:
        return 0

    # Count step indicators
    step_patterns = [
        r'(?:^|\n)\s*step\s*\d+[:\.]',  # "Step 1:"
        r'(?:^|\n)\s*\d+[.)]\s+',  # "1) " or "1. "
        r'(?:^|\n)\s*first(?:ly)?[,\s]',  # "First,"
        r'(?:^|\n)\s*second(?:ly)?[,\s]',  # "Second,"
        r'(?:^|\n)\s*third(?:ly)?[,\s]',  # "Third,"
        r'(?:^|\n)\s*then[,\s]',  # "Then,"
        r'(?:^|\n)\s*next[,\s]',  # "Next,"
        r'(?:^|\n)\s*finally[,\s]',  # "Finally,"
        r'(?:^|\n)\s*therefore[,\s]',  # "Therefore,"
    ]

    step_count = 0
    for pattern in step_patterns:
        step_count += len(re.findall(pattern, response, re.IGNORECASE))

    # Fallback: count paragraphs/newlines as approximate steps
    if step_count == 0:
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        step_count = len(paragraphs)

    return step_count


def get_reasoning_length(response: str) -> dict:
    """
    Analyze the length of reasoning in a response.

    Args:
        response: Model generated response

    Returns:
        Dictionary with length statistics
    """
    if not response:
        return {"total_chars": 0, "total_words": 0, "reasoning_chars": 0}

    # Total length
    total_chars = len(response)
    total_words = len(response.split())

    # Extract reasoning section (between <think> tags if present)
    reasoning_chars = 0
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        reasoning_chars = len(think_match.group(1))
    else:
        # Assume everything before "Answer:" is reasoning
        answer_split = re.split(r'(?:^|\n)\s*answer[:\s]', response, flags=re.IGNORECASE)
        if len(answer_split) > 1:
            reasoning_chars = len(answer_split[0])
        else:
            reasoning_chars = total_chars

    return {
        "total_chars": total_chars,
        "total_words": total_words,
        "reasoning_chars": reasoning_chars,
        "reasoning_ratio": reasoning_chars / max(total_chars, 1),
    }
