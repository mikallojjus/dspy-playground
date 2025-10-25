"""
Confidence calculator for claims based on quote quality.

Calculates weighted confidence scores using average relevance, max relevance,
and quote count with diminishing returns.

Usage:
    from src.scoring.confidence_calculator import ConfidenceCalculator

    calculator = ConfidenceCalculator()
    components = calculator.calculate(quotes)

    print(f"Confidence: {components.final_confidence:.3f}")
    print(f"  Avg relevance: {components.avg_relevance:.3f}")
    print(f"  Max relevance: {components.max_relevance:.3f}")
    print(f"  Quote count: {components.quote_count}")
"""

from dataclasses import dataclass
from typing import List

from src.extraction.quote_finder import Quote
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfidenceComponents:
    """
    Components of confidence score for debugging.

    Attributes:
        avg_relevance: Average of all quote relevance scores
        max_relevance: Highest quote relevance score
        quote_count: Number of quotes
        count_score: Normalized quote count score (0-1)
        final_confidence: Final weighted confidence (0-1)
    """
    avg_relevance: float
    max_relevance: float
    quote_count: int
    count_score: float
    final_confidence: float


class ConfidenceCalculator:
    """
    Calculate weighted confidence scores for claims based on quote quality.

    Formula: (avgRelevance × 0.6) + (maxRelevance × 0.2) + (countScore × 0.2)

    The formula balances:
    - Average relevance (60%): Primary indicator of claim quality
    - Max relevance (20%): Rewards exceptional supporting quotes
    - Quote count (20%): More evidence increases confidence (with diminishing returns)

    Count score formula: max(0, (count - 1) / 4)
    - 1 quote → 0.0 (no confidence from single quote)
    - 2 quotes → 0.25
    - 3 quotes → 0.5
    - 5+ quotes → 1.0

    Example:
        ```python
        calculator = ConfidenceCalculator()

        quotes = [
            Quote(..., relevance_score=0.92),
            Quote(..., relevance_score=0.89),
            Quote(..., relevance_score=0.87),
        ]

        components = calculator.calculate(quotes)
        # avg: 0.893, max: 0.92, count: 3
        # count_score: (3-1)/4 = 0.5
        # confidence: 0.893×0.6 + 0.92×0.2 + 0.5×0.2 = 0.820
        ```
    """

    def __init__(
        self,
        relevance_weight: float = 0.6,
        max_relevance_weight: float = 0.2,
        count_weight: float = 0.2,
        max_quotes_for_full_score: int = 5
    ):
        """
        Initialize the confidence calculator.

        Args:
            relevance_weight: Weight for average relevance (default: 0.6)
            max_relevance_weight: Weight for max relevance (default: 0.2)
            count_weight: Weight for quote count (default: 0.2)
            max_quotes_for_full_score: Quote count for full count score (default: 5)
        """
        self.relevance_weight = relevance_weight
        self.max_relevance_weight = max_relevance_weight
        self.count_weight = count_weight
        self.max_quotes_for_full_score = max_quotes_for_full_score

        logger.info(
            f"Initialized ConfidenceCalculator: "
            f"weights=({relevance_weight:.1f}, {max_relevance_weight:.1f}, {count_weight:.1f}), "
            f"max_quotes={max_quotes_for_full_score}"
        )

    def calculate(self, quotes: List[Quote]) -> ConfidenceComponents:
        """
        Calculate confidence from quotes.

        Args:
            quotes: List of quotes supporting the claim

        Returns:
            ConfidenceComponents with all score components

        Example:
            ```python
            calculator = ConfidenceCalculator()

            quotes = [Quote(..., relevance_score=0.85) for _ in range(8)]
            components = calculator.calculate(quotes)

            assert 0.0 <= components.final_confidence <= 1.0
            assert components.quote_count == 8
            assert components.count_score == 1.0  # (8-1)/4 = 1.75 capped at 1.0
            ```
        """
        if not quotes:
            return ConfidenceComponents(
                avg_relevance=0.0,
                max_relevance=0.0,
                quote_count=0,
                count_score=0.0,
                final_confidence=0.0
            )

        avg_relevance = sum(q.relevance_score for q in quotes) / len(quotes)
        max_relevance = max(q.relevance_score for q in quotes)

        # Penalize single-quote claims: (count - 1) / 4
        # 1 quote → 0.0, 2 quotes → 0.25, 3 quotes → 0.5, 5+ quotes → 1.0
        count_score = min(max(0.0, (len(quotes) - 1) / 4), 1.0)

        confidence = (
            avg_relevance * self.relevance_weight +
            max_relevance * self.max_relevance_weight +
            count_score * self.count_weight
        )

        confidence = max(0.0, min(1.0, confidence))

        logger.debug(
            f"Calculated confidence: {confidence:.3f} "
            f"(avg={avg_relevance:.3f}, max={max_relevance:.3f}, count={len(quotes)}, count_score={count_score:.3f})"
        )

        return ConfidenceComponents(
            avg_relevance=avg_relevance,
            max_relevance=max_relevance,
            quote_count=len(quotes),
            count_score=count_score,
            final_confidence=confidence
        )
