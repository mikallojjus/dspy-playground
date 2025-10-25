"""
Quote deduplicator using position-based and text-based methods.

Deduplicates quotes globally across all claims using transcript positions
as the primary method, with text similarity as fallback.

Usage:
    from src.deduplication.quote_deduplicator import QuoteDeduplicator

    deduplicator = QuoteDeduplicator()
    unique_quotes = deduplicator.deduplicate(all_quotes)

    print(f"Deduplicated: {len(all_quotes)} → {len(unique_quotes)}")
"""

import re
from typing import List

from src.extraction.quote_finder import Quote
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class QuoteDeduplicator:
    """
    Deduplicate quotes using position overlap and text similarity.

    Features:
    - Primary method: Position overlap detection (>50% = duplicate)
    - Fallback method: Text similarity using Jaccard (>80% = duplicate)
    - Merge strategy: Keep longest text, highest relevance, earliest position
    - Text normalization for reliable comparison

    Example:
        ```python
        deduplicator = QuoteDeduplicator()

        quotes = [
            Quote("Bitcoin hit $69k", relevance=0.9, start=1500, end=1520, ...),
            Quote("Bitcoin hit $69k in Nov", relevance=0.92, start=1500, end=1545, ...),
            Quote("BTC reached sixty-nine thousand", relevance=0.88, start=1520, end=1560, ...),
        ]

        unique = deduplicator.deduplicate(quotes)
        # Result: 1 merged quote (position overlap detected)
        ```
    """

    def __init__(
        self,
        position_overlap_threshold: float = 0.5,
        text_similarity_threshold: float = 0.8,
        position_proximity_chars: int = 20
    ):
        """
        Initialize the quote deduplicator.

        Args:
            position_overlap_threshold: Overlap fraction to consider duplicates (default: 0.5)
            text_similarity_threshold: Jaccard similarity threshold (default: 0.8)
            position_proximity_chars: Max distance for text similarity check (default: 20)
        """
        self.position_overlap_threshold = position_overlap_threshold
        self.text_similarity_threshold = text_similarity_threshold
        self.position_proximity_chars = position_proximity_chars

        logger.info(
            f"Initialized QuoteDeduplicator: "
            f"overlap_threshold={position_overlap_threshold}, "
            f"text_threshold={text_similarity_threshold}"
        )

    def deduplicate(self, quotes: List[Quote]) -> List[Quote]:
        """
        Deduplicate quotes globally.

        Algorithm:
        1. Sort by position
        2. For each quote, check overlap with existing unique quotes
        3. If overlap > 50% or text similarity > 80%, merge
        4. Return unique quotes sorted by relevance

        Args:
            quotes: List of quotes to deduplicate

        Returns:
            List of unique quotes sorted by relevance (highest first)

        Example:
            ```python
            deduplicator = QuoteDeduplicator()

            quotes = [...]  # 147 quotes
            unique = deduplicator.deduplicate(quotes)  # 89 unique quotes

            print(f"Reduction: {len(quotes) - len(unique)} duplicates removed")
            print(f"Percentage: {100 * (len(quotes) - len(unique)) / len(quotes):.1f}%")
            ```
        """
        if not quotes:
            return []

        logger.info(f"Deduplicating {len(quotes)} quotes...")

        sorted_quotes = sorted(quotes, key=lambda q: q.start_position)
        unique_quotes = []

        for quote in sorted_quotes:
            merged = False

            for i, existing in enumerate(unique_quotes):
                if self._are_duplicates(quote, existing):
                    unique_quotes[i] = self._merge_quotes(existing, quote)
                    merged = True
                    logger.debug(
                        f"Merged duplicate: pos {quote.start_position}-{quote.end_position} "
                        f"into {existing.start_position}-{existing.end_position}"
                    )
                    break

            if not merged:
                unique_quotes.append(quote)

        unique_quotes.sort(key=lambda q: q.relevance_score, reverse=True)

        logger.info(
            f"Deduplicated {len(quotes)} → {len(unique_quotes)} quotes "
            f"({len(quotes) - len(unique_quotes)} duplicates removed, "
            f"{100 * (len(quotes) - len(unique_quotes)) / len(quotes):.1f}% reduction)"
        )

        return unique_quotes

    def _are_duplicates(self, q1: Quote, q2: Quote) -> bool:
        """
        Check if two quotes are duplicates.

        Args:
            q1: First quote
            q2: Second quote

        Returns:
            True if quotes are duplicates
        """
        overlap = self._position_overlap(q1, q2)

        if overlap >= self.position_overlap_threshold:
            return True

        position_distance = abs(q1.start_position - q2.start_position)

        if position_distance < self.position_proximity_chars:
            similarity = self._text_similarity(q1.quote_text, q2.quote_text)
            if similarity >= self.text_similarity_threshold:
                return True

        return False

    def _position_overlap(self, q1: Quote, q2: Quote) -> float:
        """
        Calculate position overlap percentage (0.0-1.0).

        Args:
            q1: First quote
            q2: Second quote

        Returns:
            Overlap as fraction of smaller quote length

        Example:
            ```python
            q1 = Quote(..., start=1500, end=1535)  # 35 chars
            q2 = Quote(..., start=1500, end=1545)  # 45 chars
            # Overlap: 35 chars, min length: 35 chars
            # Result: 35/35 = 1.0 (100% overlap)
            ```
        """
        overlap = max(0, min(q1.end_position, q2.end_position) -
                         max(q1.start_position, q2.start_position))

        min_length = min(
            q1.end_position - q1.start_position,
            q2.end_position - q2.start_position
        )

        return overlap / min_length if min_length > 0 else 0.0

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity on normalized tokens.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity (0.0-1.0)

        Example:
            ```python
            text1 = "Bitcoin reached $69,000"
            text2 = "Bitcoin reached $69k"
            # Tokens1: {bitcoin, reached, 69000}
            # Tokens2: {bitcoin, reached, 69k}
            # Intersection: {bitcoin, reached} = 2
            # Union: {bitcoin, reached, 69000, 69k} = 4
            # Similarity: 2/4 = 0.5
            ```
        """
        tokens1 = set(self._normalize_text(text1).split())
        tokens2 = set(self._normalize_text(text2).split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Normalization:
        - Lowercase
        - Remove punctuation
        - Collapse whitespace

        Args:
            text: Text to normalize

        Returns:
            Normalized text

        Example:
            ```python
            text = "  Bitcoin reached $69,000!  "
            normalized = self._normalize_text(text)
            # Result: "bitcoin reached 69000"
            ```
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _merge_quotes(self, q1: Quote, q2: Quote) -> Quote:
        """
        Merge two duplicate quotes (keep best attributes).

        Merge strategy:
        - Text: Keep longer text (more complete)
        - Relevance: Keep higher score
        - Position: Use earliest start, latest end (covers both)
        - Speaker: Keep from first quote
        - Timestamp: Keep from first quote
        - Entailment: Keep from quote with higher relevance (most validated)

        Args:
            q1: First quote
            q2: Second quote

        Returns:
            Merged quote with best attributes from both

        Example:
            ```python
            q1 = Quote("Bitcoin reached $69k", relevance=0.90, start=1500, end=1535)
            q2 = Quote("Bitcoin reached $69k in November", relevance=0.92, start=1500, end=1560)
            merged = self._merge_quotes(q1, q2)
            # Result: text from q2 (longer), relevance from q2 (higher),
            #         position: 1500-1560 (covers both)
            ```
        """
        longer_text = q1.quote_text if len(q1.quote_text) >= len(q2.quote_text) else q2.quote_text
        higher_score = max(q1.relevance_score, q2.relevance_score)
        earlier_start = min(q1.start_position, q2.start_position)
        later_end = max(q1.end_position, q2.end_position)

        # Preserve entailment data from quote with higher relevance score
        # (most likely to have accurate validation)
        quote_with_higher_relevance = q1 if q1.relevance_score >= q2.relevance_score else q2

        return Quote(
            quote_text=longer_text,
            relevance_score=higher_score,
            start_position=earlier_start,
            end_position=later_end,
            speaker=q1.speaker,
            timestamp_seconds=q1.timestamp_seconds,
            entailment_score=quote_with_higher_relevance.entailment_score,
            entailment_relationship=quote_with_higher_relevance.entailment_relationship
        )
