"""
Unit tests for QuoteVerifier.

Tests:
- Exact substring matching
- Normalized matching (whitespace/punctuation)
- Token overlap matching (Jaccard similarity)
- Fuzzy matching
- Hallucination detection
- Confidence scoring
- Edge cases
"""

import pytest
from src.search.quote_verification import QuoteVerifier, VerificationResult, verify_quote


class TestQuoteVerifier:
    """Test suite for QuoteVerifier."""

    def test_initialization(self):
        """Test verifier initialization."""
        # Default confidence
        verifier = QuoteVerifier()
        assert verifier.min_confidence == 0.90

        # Custom confidence
        verifier_custom = QuoteVerifier(min_confidence=0.85)
        assert verifier_custom.min_confidence == 0.85

    def test_exact_match(self):
        """Test exact substring matching (Tier 1)."""
        verifier = QuoteVerifier()

        transcript = "Bitcoin reached $69,000 in November 2021. This was a historic moment."
        quote = "Bitcoin reached $69,000 in November 2021"
        claim = "Bitcoin hit ATH"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is True
        assert result.confidence == 1.00
        assert result.match_type == "exact"
        assert result.corrected_text == quote
        assert result.start_pos == 0
        assert result.end_pos == len(quote)

    def test_exact_match_middle_of_transcript(self):
        """Test exact match in middle of transcript."""
        verifier = QuoteVerifier()

        transcript = "Introduction text here. Bitcoin reached $69,000 in November 2021. More text follows."
        quote = "Bitcoin reached $69,000 in November 2021"
        claim = "Bitcoin ATH"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is True
        assert result.confidence == 1.00
        assert result.match_type == "exact"
        assert result.start_pos > 0

    def test_normalized_match_whitespace(self):
        """Test normalized matching with whitespace differences (Tier 2)."""
        verifier = QuoteVerifier()

        transcript = "Bitcoin  reached   $69,000 in November 2021"
        quote = "Bitcoin reached $69,000 in November 2021"  # Single spaces
        claim = "Bitcoin ATH"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is True
        assert result.confidence == 0.95
        assert result.match_type == "normalized"

    def test_normalized_match_punctuation(self):
        """Test normalized matching with punctuation differences."""
        verifier = QuoteVerifier()

        transcript = 'He said "Bitcoin is the future" yesterday'
        quote = 'He said "Bitcoin is the future" yesterday'  # Different quote marks
        claim = "Bitcoin future"

        result = verifier.verify(quote, transcript, claim)

        # Should match (either exact or normalized)
        assert result.is_valid is True
        assert result.confidence >= 0.95

    def test_normalized_match_case_insensitive(self):
        """Test that normalization handles case differences."""
        verifier = QuoteVerifier()

        transcript = "BITCOIN REACHED $69,000"
        quote = "bitcoin reached $69,000"
        claim = "Bitcoin ATH"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is True
        assert result.confidence >= 0.95
        assert result.match_type in ["exact", "normalized"]

    def test_hallucination_detection(self):
        """Test detection of completely fabricated quotes."""
        verifier = QuoteVerifier()

        transcript = "Bitcoin reached $50,000 last year."
        hallucinated_quote = "Bitcoin reached $69,000 in November 2021"
        claim = "Bitcoin ATH"

        result = verifier.verify(hallucinated_quote, transcript, claim)

        assert result.is_valid is False
        assert result.confidence == 0.0
        assert result.match_type == "failed"
        assert result.corrected_text == ""

    def test_partial_hallucination(self):
        """Test detection of quotes with fabricated details."""
        verifier = QuoteVerifier()

        transcript = "Bitcoin price increased significantly in 2021."
        partial_hallucination = "Bitcoin reached exactly $69,000 on November 10th, 2021"
        claim = "Bitcoin ATH"

        result = verifier.verify(partial_hallucination, transcript, claim)

        # Should fail or have low confidence
        assert result.is_valid is False or result.confidence < 0.90

    def test_token_overlap_high_similarity(self):
        """Test token overlap matching with high Jaccard similarity."""
        verifier = QuoteVerifier(min_confidence=0.70)  # Lower threshold for this test

        transcript = "Bitcoin cryptocurrency reached sixty-nine thousand dollars USD in November 2021 market peak."
        # Quote with mostly same tokens but slightly different ordering/wording
        quote = "Bitcoin reached sixty-nine thousand dollars in November 2021"
        claim = "Bitcoin ATH"

        result = verifier.verify(quote, transcript, claim)

        # This quote exists in the transcript (exact substring)
        # So it should match with high confidence
        assert result.is_valid is True
        assert result.confidence >= 0.70

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        verifier = QuoteVerifier()

        # Empty quote
        result = verifier.verify("", "some transcript", "claim")
        assert result.is_valid is False
        assert result.match_type == "failed"

        # Empty transcript
        result = verifier.verify("some quote", "", "claim")
        assert result.is_valid is False
        assert result.match_type == "failed"

        # Both empty
        result = verifier.verify("", "", "claim")
        assert result.is_valid is False

    def test_min_confidence_threshold(self):
        """Test that min_confidence threshold is respected."""
        # Strict verifier
        verifier_strict = QuoteVerifier(min_confidence=0.95)

        # Lenient verifier
        verifier_lenient = QuoteVerifier(min_confidence=0.70)

        transcript = "Bitcoin reached approximately $69k in late 2021"
        quote = "Bitcoin reached $69,000 in November 2021"
        claim = "Bitcoin ATH"

        result_strict = verifier_strict.verify(quote, transcript, claim)
        result_lenient = verifier_lenient.verify(quote, transcript, claim)

        # Lenient might accept, strict might reject (depending on matching quality)
        # At minimum, lenient should have >= confidence than strict
        if result_lenient.is_valid:
            assert result_lenient.confidence >= verifier_lenient.min_confidence

        if result_strict.is_valid:
            assert result_strict.confidence >= verifier_strict.min_confidence

    def test_verify_convenience_function(self):
        """Test convenience function verify_quote()."""
        transcript = "Bitcoin reached $69,000 in November 2021"
        quote = "Bitcoin reached $69,000"
        claim = "Bitcoin ATH"

        is_valid, corrected, confidence = verify_quote(quote, transcript, claim)

        assert isinstance(is_valid, bool)
        assert isinstance(corrected, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

        if is_valid:
            assert corrected != ""
            assert confidence >= 0.90  # Default threshold

    def test_verification_result_repr(self):
        """Test VerificationResult string representation."""
        result = VerificationResult(
            is_valid=True,
            corrected_text="test",
            confidence=0.95,
            match_type="exact"
        )

        repr_str = repr(result)
        assert "valid=True" in repr_str
        assert "confidence=0.95" in repr_str
        assert "type=exact" in repr_str

    def test_multiline_quote(self):
        """Test verification of multiline quotes."""
        verifier = QuoteVerifier()

        transcript = """Speaker 1: Bitcoin reached $69,000.
Speaker 2: That was in November 2021.
Speaker 1: It was a historic moment."""

        quote = """Bitcoin reached $69,000.
Speaker 2: That was in November 2021."""

        claim = "Bitcoin ATH timing"

        result = verifier.verify(quote, transcript, claim)

        # Should match if quote exists in transcript
        if "Bitcoin reached $69,000" in transcript:
            assert result.is_valid is True or result.confidence > 0

    def test_very_long_transcript(self):
        """Test verification with very long transcript (performance test)."""
        verifier = QuoteVerifier()

        # Simulate 4-hour podcast transcript (~288,000 chars)
        filler = "Some podcast content here. " * 10000
        transcript = filler + "Bitcoin reached $69,000 in November 2021." + filler

        quote = "Bitcoin reached $69,000 in November 2021"
        claim = "Bitcoin ATH"

        result = verifier.verify(quote, transcript, claim)

        # Should still find exact match
        assert result.is_valid is True
        assert result.confidence == 1.00

    def test_special_characters(self):
        """Test verification with special characters."""
        verifier = QuoteVerifier()

        transcript = "He said: 'Bitcoin's price is $69k (approximately)!'"
        quote = "Bitcoin's price is $69k (approximately)"
        claim = "Bitcoin price"

        result = verifier.verify(quote, transcript, claim)

        # Should match
        assert result.is_valid is True

    def test_unicode_characters(self):
        """Test verification with unicode characters."""
        verifier = QuoteVerifier()

        transcript = "Bitcoin's price reached €60,000 in 2021"
        quote = "Bitcoin's price reached €60,000"
        claim = "Bitcoin price"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is True

    def test_numbers_and_formatting(self):
        """Test verification with various number formats."""
        verifier = QuoteVerifier()

        # Test comma formatting
        transcript = "Bitcoin reached $69,000 last year"
        quote = "Bitcoin reached $69,000"
        result = verifier.verify(quote, transcript, "")
        assert result.is_valid is True

        # Test with decimals
        transcript = "Price is $69,000.50"
        quote = "Price is $69,000.50"
        result = verifier.verify(quote, transcript, "")
        assert result.is_valid is True

    def test_corrected_text_returned(self):
        """Test that corrected text from transcript is returned."""
        verifier = QuoteVerifier()

        # Quote with minor difference, transcript has canonical version
        transcript = "Bitcoin  reached  $69,000"  # Extra spaces
        quote = "Bitcoin reached $69,000"  # Normal spaces
        claim = "Bitcoin price"

        result = verifier.verify(quote, transcript, claim)

        if result.is_valid:
            # Corrected text should be from transcript
            assert result.corrected_text in transcript or \
                   verifier._normalize_text(result.corrected_text) in verifier._normalize_text(transcript)

    def test_quote_at_transcript_start(self):
        """Test quote at very start of transcript."""
        verifier = QuoteVerifier()

        transcript = "Bitcoin reached $69,000. More content follows..."
        quote = "Bitcoin reached $69,000"
        claim = "Bitcoin"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is True
        assert result.start_pos == 0

    def test_quote_at_transcript_end(self):
        """Test quote at very end of transcript."""
        verifier = QuoteVerifier()

        transcript = "Earlier content... Bitcoin reached $69,000"
        quote = "Bitcoin reached $69,000"
        claim = "Bitcoin"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is True
        assert result.end_pos == len(transcript)

    def test_multiple_occurrences(self):
        """Test quote that appears multiple times in transcript."""
        verifier = QuoteVerifier()

        transcript = "Bitcoin surged. Bitcoin surged. Bitcoin surged again."
        quote = "Bitcoin surged"
        claim = "Bitcoin"

        result = verifier.verify(quote, transcript, claim)

        # Should find first occurrence
        assert result.is_valid is True
        assert result.start_pos == 0  # First occurrence

    def test_normalization_consistency(self):
        """Test that text normalization is consistent."""
        verifier = QuoteVerifier()

        text1 = "Bitcoin  reached   $69,000"
        text2 = "bitcoin reached $69,000"

        norm1 = verifier._normalize_text(text1)
        norm2 = verifier._normalize_text(text2)

        # Should produce identical normalized versions
        assert norm1 == norm2


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_quote_longer_than_transcript(self):
        """Test when quote is longer than transcript."""
        verifier = QuoteVerifier()

        transcript = "Short text"
        quote = "This is a much longer quote that cannot possibly exist in the short transcript"
        claim = "test"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is False

    def test_identical_quote_and_transcript(self):
        """Test when quote is identical to entire transcript."""
        verifier = QuoteVerifier()

        text = "This is the entire transcript"
        result = verifier.verify(text, text, "claim")

        assert result.is_valid is True
        assert result.confidence == 1.00

    def test_single_word_quote(self):
        """Test verification of single-word quotes."""
        verifier = QuoteVerifier()

        transcript = "Bitcoin reached $69,000"
        quote = "Bitcoin"
        claim = "cryptocurrency"

        result = verifier.verify(quote, transcript, claim)

        assert result.is_valid is True

    def test_very_long_quote(self):
        """Test verification of very long quotes."""
        verifier = QuoteVerifier()

        long_quote = "Bitcoin has been showing interesting price patterns. " * 100
        transcript = "Intro text. " + long_quote + " Outro text."

        result = verifier.verify(long_quote, transcript, "Bitcoin")

        assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
