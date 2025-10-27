"""
Data models for quotes and claims with quotes.

These dataclasses are used throughout the pipeline to represent
quotes and their relationship to claims.

Usage:
    from src.extraction.quote_finder import Quote, ClaimWithQuotes

    quote = Quote(
        quote_text="Bitcoin hit $69k in November 2021",
        relevance_score=0.95,
        start_position=1000,
        end_position=1035,
        speaker="Speaker 1",
        timestamp_seconds=120
    )

    claim_with_quotes = ClaimWithQuotes(
        claim_text="Bitcoin reached $69,000 in November 2021",
        source_chunk_id=0,
        quotes=[quote],
        confidence=0.95
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.scoring.confidence_calculator import ConfidenceComponents


@dataclass
class Quote:
    """
    A quote supporting a claim.

    Attributes:
        quote_text: The quote text
        relevance_score: How relevant this quote is to the claim (0.0-1.0)
        start_position: Character position in transcript
        end_position: Character position in transcript
        speaker: Speaker identifier
        timestamp_seconds: Timestamp in seconds
        entailment_score: Entailment confidence score (0.0-1.0), if validated
        entailment_relationship: SUPPORTS/RELATED/NEUTRAL/CONTRADICTS, if validated
    """

    quote_text: str
    relevance_score: float
    start_position: int
    end_position: int
    speaker: str
    timestamp_seconds: int
    entailment_score: Optional[float] = None
    entailment_relationship: Optional[str] = None


@dataclass
class ClaimWithQuotes:
    """
    A claim with its supporting quotes.

    Attributes:
        claim_text: The claim text
        source_chunk_id: Source chunk ID
        quotes: List of supporting quotes (max 10)
        confidence: Initial confidence (from claim extraction)
        confidence_components: Confidence score components (set by ConfidenceCalculator)
        metadata: Additional metadata (for merge tracking, etc.)
    """

    claim_text: str
    source_chunk_id: int
    quotes: List[Quote]
    confidence: float = 1.0
    confidence_components: Optional[ConfidenceComponents] = None
    metadata: dict = field(default_factory=dict)
