"""
Quote verification to catch LLM hallucinations.

Multi-tier verification strategy:
1. Exact substring match (gold standard)
2. Normalized match (whitespace/punctuation differences)
3. Token-based overlap (Jaccard similarity ≥ 0.9)
4. Fuzzy string matching (≥ 0.85 similarity)

Usage:
    from src.search.quote_verification import QuoteVerifier, verify_quote

    # Quick verification
    is_valid, corrected_text, confidence = verify_quote(
        quote_text="Bitcoin hit $69k in 2021",
        transcript=full_transcript,
        claim_text="Bitcoin reached $69,000"
    )

    if is_valid and confidence >= 0.90:
        print(f"Valid quote (confidence: {confidence:.2f})")
        print(f"Corrected text: {corrected_text}")
    else:
        print("Likely hallucination")

    # Or use verifier class for batch processing
    verifier = QuoteVerifier(min_confidence=0.90)
    result = verifier.verify(quote_text, transcript, claim_text)
"""

import re
from typing import Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.infrastructure.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """
    Result from quote verification.

    Attributes:
        is_valid: Whether quote passed verification
        corrected_text: The actual text from transcript (may differ from LLM output)
        confidence: Confidence score (0.0-1.0)
        match_type: Type of match (exact, normalized, token_overlap, fuzzy, failed)
        start_pos: Character position where quote starts in transcript (if found)
        end_pos: Character position where quote ends in transcript (if found)
    """
    is_valid: bool
    corrected_text: str
    confidence: float
    match_type: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"VerificationResult(valid={self.is_valid}, "
            f"confidence={self.confidence:.2f}, "
            f"type={self.match_type})"
        )


class QuoteVerifier:
    """
    Verify that LLM-extracted quotes actually exist in transcript.

    Multi-tier verification catches hallucinations while allowing minor
    formatting differences (whitespace, punctuation).

    Verification tiers (in order):
    1. Exact substring match → confidence 1.00
    2. Normalized match → confidence 0.95
    3. Token overlap (Jaccard ≥ 0.9) → confidence 0.80
    4. Fuzzy match (≥ 0.85 similarity) → confidence 0.70

    Example:
        ```python
        verifier = QuoteVerifier(min_confidence=0.90)

        result = verifier.verify(
            quote_text='Bitcoin hit $69k',
            transcript=full_transcript,
            claim_text='Bitcoin reached $69,000'
        )

        if result.is_valid:
            print(f"Valid quote: {result.corrected_text}")
            print(f"Found at position: {result.start_pos}")
        else:
            print(f"Hallucination detected!")
        ```
    """

    def __init__(self, min_confidence: Optional[float] = None):
        """
        Initialize quote verifier.

        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0, default from settings)
                          0.90 = strict (recommended)
                          0.85 = moderate
                          0.80 = lenient
        """
        self.min_confidence = min_confidence or settings.quote_verification_min_confidence
        logger.info(f"Initialized QuoteVerifier: min_confidence={self.min_confidence}")

    def verify(
        self,
        quote_text: str,
        transcript: str,
        claim_text: str
    ) -> VerificationResult:
        """
        Verify quote against transcript using multi-tier matching.

        Args:
            quote_text: Quote text from LLM
            transcript: Full transcript or relevant chunks
            claim_text: The claim being supported (for logging)

        Returns:
            VerificationResult with validation status and corrected text
        """
        if not quote_text or not transcript:
            logger.warning("Empty quote or transcript provided")
            return VerificationResult(
                is_valid=False,
                corrected_text="",
                confidence=0.0,
                match_type="failed"
            )

        # Tier 1: Exact substring match (gold standard)
        if quote_text in transcript:
            start_pos = transcript.find(quote_text)
            end_pos = start_pos + len(quote_text)

            logger.debug(f"✓ Exact match: {quote_text[:50]}...")
            return VerificationResult(
                is_valid=True,
                corrected_text=quote_text,
                confidence=1.00,
                match_type="exact",
                start_pos=start_pos,
                end_pos=end_pos
            )

        # Tier 2: Normalized match (whitespace/punctuation differences)
        normalized_result = self._try_normalized_match(quote_text, transcript)
        if normalized_result.is_valid:
            logger.debug(
                f"✓ Normalized match: {quote_text[:50]}... "
                f"(confidence: {normalized_result.confidence:.2f})"
            )
            return normalized_result

        # Tier 3: Token-based overlap (Jaccard similarity)
        token_result = self._try_token_overlap_match(quote_text, transcript)
        if token_result.is_valid and token_result.confidence >= self.min_confidence:
            logger.debug(
                f"✓ Token overlap match: {quote_text[:50]}... "
                f"(confidence: {token_result.confidence:.2f})"
            )
            return token_result

        # Tier 4: Fuzzy string matching
        fuzzy_result = self._try_fuzzy_match(quote_text, transcript)
        if fuzzy_result.is_valid and fuzzy_result.confidence >= self.min_confidence:
            logger.warning(
                f"⚠ Fuzzy match: {quote_text[:50]}... "
                f"(confidence: {fuzzy_result.confidence:.2f})"
            )
            return fuzzy_result

        # All tiers failed - likely hallucination
        logger.error(
            f"✗ Verification FAILED for quote: {quote_text[:300]}... "
            f"(claim: {claim_text[:300]}...)"
        )

        return VerificationResult(
            is_valid=False,
            corrected_text="",
            confidence=0.0,
            match_type="failed"
        )

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        - Lowercase
        - Collapse whitespace
        - Remove common punctuation variations

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")

        # Normalize dashes
        text = text.replace('—', '-').replace('–', '-')

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip
        text = text.strip()

        return text

    def _try_normalized_match(
        self,
        quote_text: str,
        transcript: str
    ) -> VerificationResult:
        """
        Try normalized substring match.

        Handles whitespace and punctuation differences.
        """
        normalized_quote = self._normalize_text(quote_text)
        normalized_transcript = self._normalize_text(transcript)

        if normalized_quote in normalized_transcript:
            # Find actual text in original transcript
            # Use normalized position to locate in original
            norm_start = normalized_transcript.find(normalized_quote)

            # Map back to original transcript (approximate)
            # This is tricky with normalization, so we search nearby
            actual_text = self._extract_original_text(
                transcript,
                quote_text,
                approximate_start=norm_start
            )

            if actual_text:
                start_pos = transcript.find(actual_text)
                end_pos = start_pos + len(actual_text) if start_pos >= 0 else None

                return VerificationResult(
                    is_valid=True,
                    corrected_text=actual_text,
                    confidence=0.95,
                    match_type="normalized",
                    start_pos=start_pos if start_pos >= 0 else None,
                    end_pos=end_pos
                )

        return VerificationResult(
            is_valid=False,
            corrected_text="",
            confidence=0.0,
            match_type="failed"
        )

    def _extract_original_text(
        self,
        transcript: str,
        quote_text: str,
        approximate_start: int
    ) -> Optional[str]:
        """
        Extract original text from transcript given approximate position.

        Args:
            transcript: Original transcript
            quote_text: Quote text (for length reference)
            approximate_start: Approximate start position

        Returns:
            Original text from transcript, or None if not found
        """
        # Search in window around approximate position
        window_size = len(quote_text) * 2
        search_start = max(0, approximate_start - window_size)
        search_end = min(len(transcript), approximate_start + window_size + len(quote_text))

        search_window = transcript[search_start:search_end]

        # Try to find best match in window using fuzzy matching
        normalized_quote = self._normalize_text(quote_text)

        # Sliding window search
        best_match = None
        best_score = 0.0

        for i in range(len(search_window) - len(quote_text) + 1):
            candidate = search_window[i:i + len(quote_text)]
            normalized_candidate = self._normalize_text(candidate)

            if normalized_candidate == normalized_quote:
                # Found exact normalized match
                return candidate

            # Score similarity
            score = SequenceMatcher(None, normalized_quote, normalized_candidate).ratio()
            if score > best_score:
                best_score = score
                best_match = candidate

        # Return best match if score is high enough
        if best_score >= 0.95:
            return best_match

        return None

    def _try_token_overlap_match(
        self,
        quote_text: str,
        transcript: str
    ) -> VerificationResult:
        """
        Try token-based overlap matching using Jaccard similarity.

        Checks if quote tokens substantially overlap with transcript.
        Threshold: Jaccard ≥ 0.9

        Args:
            quote_text: Quote text from LLM
            transcript: Full transcript

        Returns:
            VerificationResult
        """
        # Tokenize (simple word splitting)
        quote_tokens = set(self._normalize_text(quote_text).split())
        transcript_tokens = set(self._normalize_text(transcript).split())

        if not quote_tokens:
            return VerificationResult(
                is_valid=False,
                corrected_text="",
                confidence=0.0,
                match_type="failed"
            )

        # Calculate Jaccard similarity
        intersection = quote_tokens & transcript_tokens
        union = quote_tokens | transcript_tokens

        jaccard = len(intersection) / len(union) if union else 0.0

        if jaccard >= 0.90:
            # High token overlap - likely valid but with minor differences
            # Try to find best matching substring in transcript
            best_match = self._find_best_substring_match(quote_text, transcript)

            if best_match:
                return VerificationResult(
                    is_valid=True,
                    corrected_text=best_match,
                    confidence=0.80,
                    match_type="token_overlap"
                )

        return VerificationResult(
            is_valid=False,
            corrected_text="",
            confidence=jaccard * 0.80,  # Scale to max 0.80
            match_type="failed"
        )

    def _try_fuzzy_match(
        self,
        quote_text: str,
        transcript: str
    ) -> VerificationResult:
        """
        Try fuzzy string matching to find best substring match.

        Threshold: SequenceMatcher ratio ≥ 0.85

        Args:
            quote_text: Quote text from LLM
            transcript: Full transcript

        Returns:
            VerificationResult
        """
        best_match = self._find_best_substring_match(quote_text, transcript)

        if best_match:
            # Calculate similarity score
            score = SequenceMatcher(None, quote_text, best_match).ratio()

            if score >= 0.85:
                start_pos = transcript.find(best_match)
                end_pos = start_pos + len(best_match) if start_pos >= 0 else None

                return VerificationResult(
                    is_valid=True,
                    corrected_text=best_match,
                    confidence=0.70,
                    match_type="fuzzy",
                    start_pos=start_pos if start_pos >= 0 else None,
                    end_pos=end_pos
                )

        return VerificationResult(
            is_valid=False,
            corrected_text="",
            confidence=0.0,
            match_type="failed"
        )

    def _find_best_substring_match(
        self,
        quote_text: str,
        transcript: str,
        min_score: float = 0.85
    ) -> Optional[str]:
        """
        Find best matching substring in transcript using sliding window.

        Args:
            quote_text: Quote to find
            transcript: Transcript to search
            min_score: Minimum similarity score

        Returns:
            Best matching substring, or None if no match above threshold
        """
        quote_len = len(quote_text)

        # Optimization: don't search if transcript is way too short
        if len(transcript) < quote_len * 0.5:
            return None

        # Use normalized text for comparison
        normalized_quote = self._normalize_text(quote_text)

        best_match = None
        best_score = min_score

        # Sliding window with stride
        stride = max(1, quote_len // 10)  # 10% stride for efficiency

        for i in range(0, len(transcript) - quote_len + 1, stride):
            # Extract candidate substring
            candidate = transcript[i:i + quote_len]
            normalized_candidate = self._normalize_text(candidate)

            # Calculate similarity
            score = SequenceMatcher(
                None,
                normalized_quote,
                normalized_candidate
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = candidate

            # Early exit if perfect match
            if score == 1.0:
                break

        return best_match


# Convenience function for quick verification
def verify_quote(
    quote_text: str,
    transcript: str,
    claim_text: str,
    min_confidence: Optional[float] = None
) -> Tuple[bool, str, float]:
    """
    Quick quote verification function.

    Args:
        quote_text: Quote text from LLM
        transcript: Full transcript or relevant chunks
        claim_text: The claim being supported (for logging)
        min_confidence: Minimum confidence threshold (default from settings)

    Returns:
        Tuple of (is_valid, corrected_text, confidence)

    Example:
        ```python
        is_valid, corrected, conf = verify_quote(
            "Bitcoin hit $69k",
            full_transcript,
            "Bitcoin reached $69,000"
        )

        if is_valid:
            print(f"Valid! Using: {corrected}")
        ```
    """
    verifier = QuoteVerifier(min_confidence=min_confidence)
    result = verifier.verify(quote_text, transcript, claim_text)

    return result.is_valid, result.corrected_text, result.confidence
