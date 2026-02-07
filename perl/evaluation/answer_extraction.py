"""
Answer Extraction for PeRL Evaluation.

Extracts final answers from model-generated responses.
Handles various answer formats: boxed, text, XML tags, etc.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger("perl.evaluation.extraction")


@dataclass
class ExtractionResult:
    """Result of answer extraction."""
    answer: Optional[str]
    raw_answer: Optional[str]  # Before normalization
    method: str  # Which extraction method succeeded
    confidence: float  # 0.0 to 1.0


class AnswerExtractor:
    """
    Configurable answer extractor for different problem types.

    Supports multiple extraction strategies in priority order.
    """

    def __init__(
        self,
        extraction_order: Optional[List[str]] = None,
        normalize_numbers: bool = True,
        normalize_latex: bool = True,
    ):
        """
        Initialize the extractor.

        Args:
            extraction_order: Order of extraction methods to try
            normalize_numbers: Whether to normalize numerical answers
            normalize_latex: Whether to normalize LaTeX expressions
        """
        self.extraction_order = extraction_order or [
            "boxed",
            "answer_tag",
            "final_answer_line",
            "therefore_line",
            "last_number",
        ]
        self.normalize_numbers = normalize_numbers
        self.normalize_latex = normalize_latex

    def extract(self, response: str) -> ExtractionResult:
        """
        Extract answer from a model response.

        Args:
            response: Model-generated response

        Returns:
            ExtractionResult with extracted answer
        """
        if not response or not response.strip():
            return ExtractionResult(
                answer=None,
                raw_answer=None,
                method="none",
                confidence=0.0,
            )

        for method in self.extraction_order:
            extractor = getattr(self, f"_extract_{method}", None)
            if extractor is None:
                logger.warning(f"Unknown extraction method: {method}")
                continue

            result = extractor(response)
            if result is not None:
                raw_answer = result
                normalized = self._normalize(result)
                return ExtractionResult(
                    answer=normalized,
                    raw_answer=raw_answer,
                    method=method,
                    confidence=self._compute_confidence(method),
                )

        return ExtractionResult(
            answer=None,
            raw_answer=None,
            method="none",
            confidence=0.0,
        )

    def _extract_boxed(self, response: str) -> Optional[str]:
        """Extract from \\boxed{...} or \\fbox{...}."""
        # Handle nested braces
        patterns = [
            r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            r'\\fbox\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # Return the last boxed answer (usually the final answer)
                return matches[-1].strip()

        return None

    def _extract_answer_tag(self, response: str) -> Optional[str]:
        """Extract from <answer>...</answer> tags."""
        match = re.search(
            r'<answer>(.*?)</answer>',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            content = match.group(1).strip()
            # Check for boxed within answer tags
            boxed = self._extract_boxed(content)
            if boxed:
                return boxed
            return content

        return None

    def _extract_final_answer_line(self, response: str) -> Optional[str]:
        """Extract from 'Answer:', 'Final answer:', etc."""
        patterns = [
            r'(?:^|\n)\s*(?:final\s+)?answer\s*[:\s]+(.+?)(?:\n|$)',
            r'(?:^|\n)\s*the\s+answer\s+is\s*[:\s]*(.+?)(?:\n|$)',
            r'(?:^|\n)\s*(?:so|thus)[,\s]+the\s+answer\s+is\s*[:\s]*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                # Check for boxed in the answer line
                boxed = self._extract_boxed(answer)
                if boxed:
                    return boxed
                # Clean up
                answer = self._clean_answer_text(answer)
                if answer:
                    return answer

        return None

    def _extract_therefore_line(self, response: str) -> Optional[str]:
        """Extract from 'Therefore...' conclusions."""
        patterns = [
            r'therefore[,\s]+(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\n|$)',
            r'hence[,\s]+(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\n|$)',
            r'thus[,\s]+(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                boxed = self._extract_boxed(answer)
                if boxed:
                    return boxed
                answer = self._clean_answer_text(answer)
                if answer:
                    return answer

        return None

    def _extract_last_number(self, response: str) -> Optional[str]:
        """Extract the last number or mathematical expression."""
        # Find all numbers (including decimals, fractions, negatives)
        number_pattern = r'[-+]?\d+(?:\.\d+)?(?:/\d+)?'
        matches = re.findall(number_pattern, response)

        if matches:
            return matches[-1]

        return None

    def _clean_answer_text(self, answer: str) -> str:
        """Clean up extracted answer text."""
        # Remove $ delimiters
        answer = re.sub(r'^\$+|\$+$', '', answer)

        # Remove trailing punctuation
        answer = re.sub(r'[.!?,;:]+$', '', answer)

        # Remove leading punctuation
        answer = re.sub(r'^[.!?,;:]+\s*', '', answer)

        # Remove "is" prefix
        answer = re.sub(r'^is\s+', '', answer, flags=re.IGNORECASE)

        return answer.strip()

    def _normalize(self, answer: str) -> str:
        """Normalize an extracted answer."""
        if not answer:
            return answer

        normalized = answer.strip()

        if self.normalize_numbers:
            normalized = self._normalize_number(normalized)

        if self.normalize_latex:
            normalized = self._normalize_latex_expr(normalized)

        return normalized

    def _normalize_number(self, answer: str) -> str:
        """Normalize numerical answers."""
        # Remove commas from numbers
        answer = answer.replace(",", "")

        # Convert fraction notation
        if "/" in answer:
            try:
                # Handle simple fractions like 3/4
                match = re.match(r'^(-?\d+)/(\d+)$', answer)
                if match:
                    num, den = int(match.group(1)), int(match.group(2))
                    # Keep as fraction if it's clean
                    from math import gcd
                    g = gcd(abs(num), den)
                    num, den = num // g, den // g
                    if den == 1:
                        return str(num)
                    return f"{num}/{den}"
            except Exception:
                pass

        # Try to parse as float and clean up
        try:
            val = float(answer)
            # If it's a whole number, return as int
            if val == int(val):
                return str(int(val))
            # Otherwise round to reasonable precision
            return f"{val:.10g}"
        except ValueError:
            pass

        return answer

    def _normalize_latex_expr(self, answer: str) -> str:
        """Normalize LaTeX expressions."""
        # Remove unnecessary whitespace
        answer = " ".join(answer.split())

        # Normalize \frac{a}{b} to a/b for simple fractions
        answer = re.sub(
            r'\\frac\{(\d+)\}\{(\d+)\}',
            r'\1/\2',
            answer
        )

        # Remove \text{} wrapper
        answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)

        # Normalize \sqrt
        answer = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', answer)

        return answer

    def _compute_confidence(self, method: str) -> float:
        """Compute confidence score based on extraction method."""
        confidence_map = {
            "boxed": 1.0,
            "answer_tag": 0.95,
            "final_answer_line": 0.8,
            "therefore_line": 0.7,
            "last_number": 0.3,
            "none": 0.0,
        }
        return confidence_map.get(method, 0.5)


def extract_answer(
    response: str,
    extraction_order: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Convenience function to extract answer from response.

    Args:
        response: Model-generated response
        extraction_order: Order of extraction methods

    Returns:
        Extracted answer or None
    """
    extractor = AnswerExtractor(extraction_order=extraction_order)
    result = extractor.extract(response)
    return result.answer


def extract_reasoning_steps(response: str) -> Tuple[int, List[str]]:
    """
    Count and extract reasoning steps from a response.

    Looks for explicit step markers (Step 1:, 1), First, etc.)
    and estimates steps from paragraph structure.

    Args:
        response: Model-generated response

    Returns:
        Tuple of (step_count, list_of_step_texts)
    """
    steps = []

    # Pattern 1: "Step N:" or "Step N."
    step_pattern = r'(?:^|\n)\s*step\s*(\d+)\s*[.:]\s*(.+?)(?=\n\s*step\s*\d+|\n\n|$)'
    step_matches = re.findall(step_pattern, response, re.IGNORECASE | re.DOTALL)
    if step_matches:
        for num, text in step_matches:
            steps.append(f"Step {num}: {text.strip()[:100]}...")
        return len(steps), steps

    # Pattern 2: Numbered lists "1) " or "1. "
    list_pattern = r'(?:^|\n)\s*(\d+)\s*[.)]\s+(.+?)(?=\n\s*\d+\s*[.)]|\n\n|$)'
    list_matches = re.findall(list_pattern, response, re.DOTALL)
    if list_matches:
        for num, text in list_matches:
            steps.append(f"{num}. {text.strip()[:100]}...")
        return len(steps), steps

    # Pattern 3: Transition words (First, Second, Then, Next, Finally)
    transition_words = [
        (r'(?:^|\n)\s*first(?:ly)?[,\s]', "First"),
        (r'(?:^|\n)\s*second(?:ly)?[,\s]', "Second"),
        (r'(?:^|\n)\s*third(?:ly)?[,\s]', "Third"),
        (r'(?:^|\n)\s*then[,\s]', "Then"),
        (r'(?:^|\n)\s*next[,\s]', "Next"),
        (r'(?:^|\n)\s*finally[,\s]', "Finally"),
        (r'(?:^|\n)\s*therefore[,\s]', "Therefore"),
        (r'(?:^|\n)\s*thus[,\s]', "Thus"),
        (r'(?:^|\n)\s*hence[,\s]', "Hence"),
    ]

    for pattern, label in transition_words:
        if re.search(pattern, response, re.IGNORECASE):
            steps.append(label)

    if steps:
        return len(steps), steps

    # Fallback: count paragraphs as approximate steps
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        return len(paragraphs), [f"Para {i+1}" for i in range(len(paragraphs))]

    # Single paragraph or unclear structure
    # Estimate based on sentence count
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    step_count = max(1, len(sentences) // 3)  # Rough estimate: 3 sentences per step

    return step_count, []


def count_reasoning_tokens(response: str) -> int:
    """
    Estimate the number of reasoning tokens in a response.

    Excludes the final answer from the count.
    """
    # Try to find where reasoning ends (before final answer)
    end_patterns = [
        r'(?:^|\n)\s*(?:final\s+)?answer\s*:',
        r'\\boxed\{',
        r'<answer>',
        r'therefore[,\s]+(?:the\s+)?(?:final\s+)?answer\s+is',
    ]

    reasoning_end = len(response)
    for pattern in end_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            reasoning_end = min(reasoning_end, match.start())

    reasoning_text = response[:reasoning_end]

    # Rough token count (words * 1.3)
    words = len(reasoning_text.split())
    return int(words * 1.3)


def extract_think_content(response: str) -> Optional[str]:
    """
    Extract content from <think>...</think> tags.

    Args:
        response: Model-generated response

    Returns:
        Content between think tags, or None
    """
    match = re.search(
        r'<think>(.*?)</think>',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    return None
