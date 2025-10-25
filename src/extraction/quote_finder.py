"""
Quote finder service for linking claims to supporting quotes.

Uses semantic search to find relevant quotes for each claim and filters
out questions.

Usage:
    from src.extraction.quote_finder import QuoteFinder

    finder = QuoteFinder(search_index)
    claims_with_quotes = await finder.find_quotes_for_claims(claims)

    for claim in claims_with_quotes:
        print(f"Claim: {claim.claim_text}")
        print(f"Quotes: {len(claim.quotes)}")
"""

from dataclasses import dataclass
from typing import List

from src.config.settings import settings
from src.extraction.claim_extractor import ExtractedClaim
from src.search.transcript_search_index import TranscriptSearchIndex, QuoteCandidate
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


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
    """
    quote_text: str
    relevance_score: float
    start_position: int
    end_position: int
    speaker: str
    timestamp_seconds: int


@dataclass
class ClaimWithQuotes:
    """
    A claim with its supporting quotes.

    Attributes:
        claim_text: The claim text
        source_chunk_id: Source chunk ID
        quotes: List of supporting quotes (max 10)
        confidence: Initial confidence (from claim extraction)
    """
    claim_text: str
    source_chunk_id: int
    quotes: List[Quote]
    confidence: float = 1.0


class QuoteFinder:
    """
    Service for finding supporting quotes for claims.

    Features:
    - Semantic search using transcript search index
    - Question filtering (removes rhetorical questions)
    - Top-K selection (configurable, default: 10)
    - Relevance scoring based on embedding similarity

    Example:
        ```python
        finder = QuoteFinder(search_index)

        claims_with_quotes = await finder.find_quotes_for_claims(claims)

        for cwq in claims_with_quotes:
            print(f"Claim: {cwq.claim_text}")
            print(f"  {len(cwq.quotes)} quotes:")
            for quote in cwq.quotes[:3]:
                print(f"    [{quote.relevance_score:.3f}] {quote.quote_text[:50]}...")
        ```
    """

    # Question words (case-insensitive)
    QUESTION_WORDS = {
        "who", "what", "when", "where", "why", "how",
        "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "will", "would", "should"
    }

    def __init__(
        self,
        search_index: TranscriptSearchIndex,
        max_quotes_per_claim: int = None,
        initial_candidates: int = 30
    ):
        """
        Initialize the quote finder.

        Args:
            search_index: Transcript search index for semantic search
            max_quotes_per_claim: Maximum quotes per claim (default from settings)
            initial_candidates: Number of candidates to fetch before filtering
        """
        self.search_index = search_index
        self.max_quotes = max_quotes_per_claim or settings.max_quotes_per_claim
        self.initial_candidates = initial_candidates

        logger.info(
            f"Initialized QuoteFinder: max_quotes={self.max_quotes}, "
            f"initial_candidates={self.initial_candidates}"
        )

    async def find_quotes_for_claims(
        self,
        claims: List[ExtractedClaim]
    ) -> List[ClaimWithQuotes]:
        """
        Find supporting quotes for all claims.

        Args:
            claims: List of extracted claims

        Returns:
            List of claims with their supporting quotes

        Example:
            ```python
            finder = QuoteFinder(search_index)
            claims_with_quotes = await finder.find_quotes_for_claims(claims)

            total_quotes = sum(len(c.quotes) for c in claims_with_quotes)
            avg_quotes = total_quotes / len(claims_with_quotes)
            print(f"Found {total_quotes} quotes for {len(claims_with_quotes)} claims")
            print(f"Average: {avg_quotes:.1f} quotes per claim")
            ```
        """
        if not claims:
            logger.warning("No claims provided for quote finding")
            return []

        logger.info(f"Finding quotes for {len(claims)} claims")

        claims_with_quotes: List[ClaimWithQuotes] = []

        for i, claim in enumerate(claims, 1):
            quotes = await self._find_quotes_for_claim(claim)

            claim_with_quotes = ClaimWithQuotes(
                claim_text=claim.claim_text,
                source_chunk_id=claim.source_chunk_id,
                quotes=quotes,
                confidence=claim.confidence
            )

            claims_with_quotes.append(claim_with_quotes)

            logger.debug(
                f"Claim {i}/{len(claims)}: found {len(quotes)} quotes "
                f"(relevance: {quotes[0].relevance_score:.3f}-{quotes[-1].relevance_score:.3f})"
                if quotes else f"Claim {i}/{len(claims)}: no quotes found"
            )

        total_quotes = sum(len(c.quotes) for c in claims_with_quotes)
        avg_quotes = total_quotes / len(claims_with_quotes) if claims_with_quotes else 0

        logger.info(
            f"Found {total_quotes} quotes for {len(claims)} claims "
            f"(avg {avg_quotes:.1f} quotes/claim)"
        )

        return claims_with_quotes

    async def _find_quotes_for_claim(
        self,
        claim: ExtractedClaim
    ) -> List[Quote]:
        """
        Find supporting quotes for a single claim.

        Args:
            claim: Extracted claim

        Returns:
            List of quotes (up to max_quotes)
        """
        # 1. Semantic search for candidates
        candidates = await self.search_index.find_quotes_for_claim(
            claim.claim_text,
            top_k=self.initial_candidates
        )

        if not candidates:
            logger.warning(f"No candidates found for claim: {claim.claim_text[:60]}...")
            return []

        # 2. Filter out questions
        filtered_candidates = [
            c for c in candidates
            if not self._is_question(c.quote_text)
        ]

        if len(filtered_candidates) < len(candidates):
            logger.debug(
                f"Filtered {len(candidates) - len(filtered_candidates)} questions "
                f"({len(filtered_candidates)} remaining)"
            )

        if not filtered_candidates:
            logger.warning(
                f"All candidates were questions for claim: {claim.claim_text[:60]}..."
            )
            return []

        # 3. Select top quotes (already sorted by similarity from search)
        top_candidates = filtered_candidates[:self.max_quotes]

        # 4. Convert to Quote objects
        quotes = [
            Quote(
                quote_text=c.quote_text,
                relevance_score=c.similarity_score,
                start_position=c.start_position,
                end_position=c.end_position,
                speaker=c.speaker,
                timestamp_seconds=c.timestamp_seconds
            )
            for c in top_candidates
        ]

        return quotes

    def _is_question(self, text: str) -> bool:
        """
        Check if text is a question.

        Uses simple heuristics:
        - Ends with question mark
        - Starts with question word

        Args:
            text: Text to check

        Returns:
            True if text appears to be a question

        Example:
            ```python
            finder = QuoteFinder(search_index)

            assert finder._is_question("What is Bitcoin?")
            assert finder._is_question("Is this true?")
            assert not finder._is_question("Bitcoin is great.")
            assert not finder._is_question("What happened was...")  # Statement
            ```
        """
        if not text:
            return False

        # Check for question mark
        if "?" in text:
            return True

        # Check if starts with question word
        first_word = text.split()[0].lower().strip(".,!?;:")

        if first_word in self.QUESTION_WORDS:
            # Additional check: "What happened was..." is not a question
            # Simple heuristic: if it doesn't end with "?", check for certain patterns
            if "what happened" in text.lower() or "what occurred" in text.lower():
                return False

            return True

        return False

    def get_claims_without_quotes(
        self,
        claims_with_quotes: List[ClaimWithQuotes]
    ) -> List[ClaimWithQuotes]:
        """
        Get claims that have no supporting quotes.

        Args:
            claims_with_quotes: List of claims with quotes

        Returns:
            Claims with empty quotes list

        Example:
            ```python
            finder = QuoteFinder(search_index)
            claims_with_quotes = await finder.find_quotes_for_claims(claims)

            no_quotes = finder.get_claims_without_quotes(claims_with_quotes)
            print(f"{len(no_quotes)} claims have no supporting quotes")
            ```
        """
        return [c for c in claims_with_quotes if not c.quotes]
