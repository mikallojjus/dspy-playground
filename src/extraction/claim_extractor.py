"""
Claim extraction service using optimized DSPy model.

Processes transcript chunks in parallel and extracts factual claims.

Usage:
    from src.extraction.claim_extractor import ClaimExtractor
    from src.preprocessing.chunking_service import ChunkingService

    chunker = ChunkingService()
    chunks = chunker.chunk_text(transcript)

    extractor = ClaimExtractor()
    claims = await extractor.extract_from_chunks(chunks)

    print(f"Extracted {len(claims)} claims from {len(chunks)} chunks")
"""

import asyncio
import re
import dspy
from dataclasses import dataclass
from typing import List

from src.config.settings import settings
from src.dspy_models.claim_extractor import ClaimExtractorModel
from src.preprocessing.chunking_service import TextChunk
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedClaim:
    """
    A claim extracted from a transcript chunk.

    Attributes:
        claim_text: The factual claim text
        source_chunk_id: ID of the chunk this claim came from
        confidence: Initial confidence score (optional, for future use)
    """

    claim_text: str
    source_chunk_id: int
    confidence: float = 1.0


class ClaimExtractor:
    """
    Service for extracting claims from transcript chunks using DSPy.

    Features:
    - Uses optimized DSPy claim extraction model
    - Parallel processing of chunks (configurable batch size)
    - Tracks source chunk for each claim
    - Error handling per chunk (failures don't stop pipeline)

    Example:
        ```python
        extractor = ClaimExtractor()

        # Extract from chunks
        claims = await extractor.extract_from_chunks(chunks)

        # Display results
        for claim in claims:
            print(f"Chunk {claim.source_chunk_id}: {claim.claim_text}")
        ```
    """

    def __init__(self, model_path: str = "models/claim_extractor_llm_judge_v1.json"):
        """
        Initialize the claim extractor.

        Args:
            model_path: Path to optimized DSPy model file
        """
        logger.info("Initializing ClaimExtractor with optimized DSPy model")

        # Load DSPy model
        self.model = ClaimExtractorModel(model_path=model_path)

        # Create async-compatible version using dspy.asyncify for thread safety
        # This ensures proper DSPy context propagation across threads
        self.async_extract = dspy.asyncify(self.model.extract_claims)

        # Get parallel batch size from settings
        self.batch_size = settings.parallel_batch_size

        logger.info(f"ClaimExtractor ready (batch_size={self.batch_size})")

    def _is_valid_claim(self, claim_text: str) -> bool:
        """
        Filter vague/generic claims using heuristic rules.

        A claim is INVALID if it:
        - Is too short (<5 words) or too long (>40 words)
        - Matches generic patterns ("crypto is risky", "bitcoin is important")
        - Lacks specificity (no numbers, names, or dates)

        Args:
            claim_text: Claim text to validate

        Returns:
            True if claim passes quality checks, False otherwise
        """
        # Length check (5-40 words)
        word_count = len(claim_text.split())
        if word_count < 5 or word_count > 40:
            logger.debug(f"Filtered claim (length={word_count}): {claim_text[:50]}...")
            return False

        claim_lower = claim_text.lower()

        # Specificity check: must contain at least ONE of:
        # - Number/percentage (digit)
        # - Proper noun (capitalized word, but not sentence start)
        # - Date/year pattern
        has_number = bool(re.search(r"\d", claim_text))

        # TODO Verify if this does not filter too many valid claims
        # Check for proper nouns (capitalized words NOT at sentence start)
        words = claim_text.split()
        has_proper_noun = False
        for i, word in enumerate(words):
            if i > 0 and word and word[0].isupper() and len(word) > 1:
                # Skip common articles/conjunctions
                if word.lower() not in ["and", "or", "but", "the", "a", "an"]:
                    has_proper_noun = True
                    break

        # Common date patterns
        has_date = bool(
            re.search(
                r"\b(19|20)\d{2}\b|"  # Year
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|"  # Month
                r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",  # Month abbreviation
                claim_lower,
            )
        )

        if not (has_number or has_proper_noun or has_date):
            logger.debug(f"Filtered non-specific claim: {claim_text[:50]}...")
            return False

        return True

    async def extract_from_chunks(
        self, chunks: List[TextChunk]
    ) -> List[ExtractedClaim]:
        """
        Extract claims from transcript chunks in parallel.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of extracted claims with source chunk tracking

        Example:
            ```python
            extractor = ClaimExtractor()
            claims = await extractor.extract_from_chunks(chunks)

            # Group claims by chunk
            by_chunk = {}
            for claim in claims:
                if claim.source_chunk_id not in by_chunk:
                    by_chunk[claim.source_chunk_id] = []
                by_chunk[claim.source_chunk_id].append(claim.claim_text)

            for chunk_id, chunk_claims in by_chunk.items():
                print(f"Chunk {chunk_id}: {len(chunk_claims)} claims")
            ```
        """
        if not chunks:
            logger.warning("No chunks provided for claim extraction")
            return []

        logger.info(
            f"Starting claim extraction from {len(chunks)} chunks "
            f"(batch_size={self.batch_size})"
        )

        all_claims: List[ExtractedClaim] = []

        # Process chunks in batches
        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            logger.debug(
                f"Processing batch {batch_start // self.batch_size + 1} "
                f"(chunks {batch_start}-{batch_end - 1})"
            )

            # Process batch in parallel
            batch_results = await asyncio.gather(
                *[self._extract_from_chunk(chunk) for chunk in batch],
                return_exceptions=True,
            )

            # Collect results (handle exceptions)
            for chunk, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Error extracting claims from chunk {chunk.chunk_id}: {result}",
                        exc_info=result,
                    )
                    continue

                # Type assertion: result is List[ExtractedClaim] here (not Exception)
                assert isinstance(result, list), "result should be a list after exception check"
                if result:
                    all_claims.extend(result)

        logger.info(
            f"Extracted {len(all_claims)} total claims from {len(chunks)} chunks "
            f"(avg {len(all_claims) / len(chunks):.1f} claims/chunk)"
        )

        return all_claims

    async def _extract_from_chunk(self, chunk: TextChunk) -> List[ExtractedClaim]:
        """
        Extract claims from a single chunk.

        Args:
            chunk: Text chunk to process

        Returns:
            List of claims from this chunk

        Note:
            This is async to allow parallel processing. The DSPy model call
            is synchronous but wrapped with dspy.asyncify() for thread-safe
            execution with proper context propagation.
        """
        try:
            # Run DSPy model asynchronously with proper context propagation
            claim_texts = await self.async_extract(chunk.text)

            # Convert to ExtractedClaim objects, filtering out invalid claims
            claims_before_filter = []
            claims = []

            for claim_text in claim_texts:
                # Skip empty/whitespace-only claims
                if not claim_text or not claim_text.strip():
                    continue

                claims_before_filter.append(claim_text.strip())

                # Apply quality filter
                if self._is_valid_claim(claim_text.strip()):
                    claims.append(
                        ExtractedClaim(
                            claim_text=claim_text.strip(),
                            source_chunk_id=chunk.chunk_id,
                            confidence=1.0,  # Initial confidence (before scoring)
                        )
                    )

            filtered_count = len(claims_before_filter) - len(claims)

            logger.debug(
                f"Chunk {chunk.chunk_id}: extracted {len(claims)} claims "
                f"({filtered_count} filtered by quality) "
                f"({len(chunk.text)} chars)"
            )

            return claims

        except Exception as e:
            logger.error(
                f"Failed to extract claims from chunk {chunk.chunk_id}: {e}",
                exc_info=True,
            )
            return []

    def get_claims_by_chunk(
        self, claims: List[ExtractedClaim]
    ) -> dict[int, List[ExtractedClaim]]:
        """
        Group claims by source chunk ID.

        Args:
            claims: List of extracted claims

        Returns:
            Dictionary mapping chunk_id to list of claims

        Example:
            ```python
            extractor = ClaimExtractor()
            claims = await extractor.extract_from_chunks(chunks)

            by_chunk = extractor.get_claims_by_chunk(claims)
            for chunk_id, chunk_claims in by_chunk.items():
                print(f"Chunk {chunk_id}: {len(chunk_claims)} claims")
            ```
        """
        by_chunk: dict[int, List[ExtractedClaim]] = {}

        for claim in claims:
            if claim.source_chunk_id not in by_chunk:
                by_chunk[claim.source_chunk_id] = []
            by_chunk[claim.source_chunk_id].append(claim)

        return by_chunk
