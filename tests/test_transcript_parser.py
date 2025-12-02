"""
Unit tests for transcript parsers.

Tests cover:
1. AssemblyParser: Letter-based speaker parsing (Speaker A, Speaker B, etc.)
2. Edge cases: Empty transcripts, malformed timestamps
3. Integration: TranscriptParser factory routing

Run with:
    pytest tests/test_transcript_parser.py -v
    pytest tests/test_transcript_parser.py -v -k "test_assembly"  # Run Assembly tests only
"""

import pytest
from src.preprocessing.transcript_parser import (
    TranscriptParser,
    AssemblyParser,
    PodscribeParser,
    BanklessParser,
    ParsedTranscript,
    TranscriptSegment,
)


# ============================================================================
# ASSEMBLY PARSER TESTS
# ============================================================================


class TestAssemblyParser:
    """Test AssemblyParser for Assembly transcript format."""

    def test_parse_basic_assembly_transcript(self):
        """Test parsing basic Assembly format with letter-based speakers."""
        parser = AssemblyParser()

        transcript = """Speaker A (0s):
Hi, I'm Solana Pyne.

Speaker B (2s):
I'm the director of video at the New York Times.

Speaker A (5s):
For years, my team has made videos."""

        result = parser.parse(transcript)

        # Should parse 3 segments
        assert len(result.segments) == 3

        # Check first segment
        assert result.segments[0].speaker == "Speaker_A"
        assert "Solana Pyne" in result.segments[0].clean_text
        assert result.segments[0].timestamp_seconds == 0

        # Check second segment
        assert result.segments[1].speaker == "Speaker_B"
        assert "New York Times" in result.segments[1].clean_text
        assert result.segments[1].timestamp_seconds == 2

        # Check third segment
        assert result.segments[2].speaker == "Speaker_A"
        assert "made videos" in result.segments[2].clean_text
        assert result.segments[2].timestamp_seconds == 5

        # Check full text is concatenated
        assert "Solana Pyne" in result.full_text
        assert "New York Times" in result.full_text
        assert "made videos" in result.full_text

    def test_parse_assembly_with_minutes(self):
        """Test parsing Assembly timestamps with minutes and seconds."""
        parser = AssemblyParser()

        transcript = """Speaker A (1m 30s):
First statement with minute and seconds.

Speaker B (2m 5s):
Second statement.

Speaker C (3m):
Third statement with just minutes."""

        result = parser.parse(transcript)

        assert len(result.segments) == 3

        # Check timestamp conversions
        assert result.segments[0].timestamp_seconds == 90  # 1m 30s = 90s
        assert result.segments[0].speaker == "Speaker_A"

        assert result.segments[1].timestamp_seconds == 125  # 2m 5s = 125s
        assert result.segments[1].speaker == "Speaker_B"

        assert result.segments[2].timestamp_seconds == 180  # 3m = 180s
        assert result.segments[2].speaker == "Speaker_C"

    def test_parse_assembly_multiple_speakers(self):
        """Test parsing with multiple letter-based speakers (A through Z)."""
        parser = AssemblyParser()

        transcript = """Speaker A (0s):
Speaker A talks first.

Speaker B (5s):
Speaker B responds.

Speaker C (10s):
Speaker C joins.

Speaker A (15s):
Speaker A continues.

Speaker D (20s):
Speaker D adds input."""

        result = parser.parse(transcript)

        assert len(result.segments) == 5

        # Verify speaker labels
        assert result.segments[0].speaker == "Speaker_A"
        assert result.segments[1].speaker == "Speaker_B"
        assert result.segments[2].speaker == "Speaker_C"
        assert result.segments[3].speaker == "Speaker_A"  # Speaker A appears again
        assert result.segments[4].speaker == "Speaker_D"

    def test_parse_assembly_empty_transcript(self):
        """Test parsing empty transcript."""
        parser = AssemblyParser()

        result = parser.parse("")

        assert len(result.segments) == 0
        assert result.full_text == ""

    def test_parse_assembly_no_segments(self):
        """Test parsing transcript with no valid segments."""
        parser = AssemblyParser()

        # Transcript without proper format
        transcript = "This is just plain text without any speaker markers."

        result = parser.parse(transcript)

        # Should return single unknown segment
        assert len(result.segments) == 1
        assert result.segments[0].speaker == "Speaker_Unknown"
        assert "plain text" in result.segments[0].clean_text

    def test_parse_assembly_segment_positions(self):
        """Test that segment positions are correctly tracked."""
        parser = AssemblyParser()

        transcript = """Speaker A (0s):
First segment text.

Speaker B (5s):
Second segment text."""

        result = parser.parse(transcript)

        # Verify positions are tracked
        assert result.segments[0].start_position > 0
        assert result.segments[0].end_position > result.segments[0].start_position

        assert result.segments[1].start_position > result.segments[0].end_position
        assert result.segments[1].end_position > result.segments[1].start_position

    def test_parse_assembly_multiline_text(self):
        """Test parsing segments with multi-line text content."""
        parser = AssemblyParser()

        transcript = """Speaker A (0s):
This is a long statement.
It spans multiple lines.
With several sentences.

Speaker B (10s):
Another multi-line statement.
Also with multiple lines."""

        result = parser.parse(transcript)

        assert len(result.segments) == 2

        # Check that multiline content is preserved
        assert "multiple lines" in result.segments[0].clean_text
        assert "several sentences" in result.segments[0].clean_text
        assert "Also with multiple lines" in result.segments[1].clean_text

    def test_parse_timestamp_edge_cases(self):
        """Test timestamp parsing edge cases."""
        parser = AssemblyParser()

        # Test various timestamp formats
        assert parser._parse_timestamp("0s") == 0
        assert parser._parse_timestamp("30s") == 30
        assert parser._parse_timestamp("1m") == 60
        assert parser._parse_timestamp("1m 0s") == 60
        assert parser._parse_timestamp("2m 30s") == 150
        assert parser._parse_timestamp("10m 5s") == 605

        # Test hour-based timestamps
        assert parser._parse_timestamp("1h") == 3600  # 1 hour
        assert parser._parse_timestamp("1h 0m 0s") == 3600  # 1 hour
        assert parser._parse_timestamp("1h 3m 21s") == 3801  # 1h 3m 21s = 3600 + 180 + 21
        assert parser._parse_timestamp("1h 8m 9s") == 4089  # 1h 8m 9s = 3600 + 480 + 9
        assert parser._parse_timestamp("2h 30m") == 9000  # 2h 30m = 7200 + 1800
        assert parser._parse_timestamp("2h 30m 15s") == 9015  # 2h 30m 15s = 7200 + 1800 + 15

        # Invalid timestamp should return 0
        assert parser._parse_timestamp("invalid") == 0
        assert parser._parse_timestamp("") == 0

    def test_remove_assemblyai_header(self):
        """Test that AssemblyAI header is correctly removed."""
        parser = AssemblyParser()

        # Transcript with header
        transcript_with_header = """AssemblyAI Transcription Result
Processing Time: 23.80 seconds
============================================================

Speaker A (2s):
One of the two National Guard members shot allegedly by an Afghan man in Washington, D.C. has died.

Speaker B (58s):
Our Common Nature is a musical journey with Yo Yo Ma."""

        result = parser.parse(transcript_with_header)

        # Should successfully parse despite header
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "Speaker_A"
        assert "National Guard" in result.segments[0].clean_text
        assert result.segments[0].timestamp_seconds == 2

        assert result.segments[1].speaker == "Speaker_B"
        assert "Common Nature" in result.segments[1].clean_text
        assert result.segments[1].timestamp_seconds == 58

        # Header text should NOT appear in output
        assert "AssemblyAI" not in result.full_text
        assert "Processing Time" not in result.full_text

    def test_parse_without_header(self):
        """Test that transcripts without header still work correctly."""
        parser = AssemblyParser()

        # Transcript without header (clean)
        transcript_clean = """Speaker A (0s):
This is a test.

Speaker B (5s):
Another test."""

        result = parser.parse(transcript_clean)

        # Should parse normally
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "Speaker_A"
        assert result.segments[1].speaker == "Speaker_B"


# ============================================================================
# TRANSCRIPT PARSER FACTORY TESTS
# ============================================================================


class TestTranscriptParserFactory:
    """Test TranscriptParser factory routing."""

    def test_factory_routes_to_assembly_parser(self):
        """Test that factory correctly routes to AssemblyParser."""
        factory = TranscriptParser()

        transcript = """Speaker A (0s):
Test content."""

        result = factory.parse(transcript, format="assembly")

        assert len(result.segments) == 1
        assert result.segments[0].speaker == "Speaker_A"

    def test_factory_routes_to_podscribe_parser(self):
        """Test that factory correctly routes to PodscribeParser."""
        factory = TranscriptParser()

        transcript = """0 (30s):
Test content."""

        result = factory.parse(transcript, format="podscribe")

        assert len(result.segments) == 1
        assert result.segments[0].speaker == "Speaker_0"

    def test_factory_routes_to_bankless_parser(self):
        """Test that factory correctly routes to BanklessParser."""
        factory = TranscriptParser()

        transcript = """David:
[0:03] Test content."""

        result = factory.parse(transcript, format="bankless")

        assert len(result.segments) == 1
        assert result.segments[0].speaker == "David"

    def test_factory_unsupported_format_raises_error(self):
        """Test that unsupported format raises ValueError."""
        factory = TranscriptParser()

        with pytest.raises(ValueError) as exc_info:
            factory.parse("test", format="unsupported")  # type: ignore

        assert "Unsupported transcript format" in str(exc_info.value)

    def test_factory_default_format_is_podscribe(self):
        """Test that default format is podscribe."""
        factory = TranscriptParser()

        transcript = """0 (30s):
Test content."""

        # Should use podscribe by default
        result = factory.parse(transcript)

        assert len(result.segments) == 1
        assert result.segments[0].speaker == "Speaker_0"


# ============================================================================
# COMPARISON TESTS (Assembly vs Podscribe)
# ============================================================================


class TestAssemblyVsPodscribe:
    """Test differences between Assembly and Podscribe formats."""

    def test_speaker_format_difference(self):
        """Test that Assembly uses letters while Podscribe uses numbers."""
        assembly_parser = AssemblyParser()
        podscribe_parser = PodscribeParser()

        # Assembly format
        assembly_transcript = """Speaker A (0s):
Test content."""
        assembly_result = assembly_parser.parse(assembly_transcript)

        # Podscribe format
        podscribe_transcript = """0 (0s):
Test content."""
        podscribe_result = podscribe_parser.parse(podscribe_transcript)

        # Both should work, but with different speaker labels
        assert assembly_result.segments[0].speaker == "Speaker_A"
        assert podscribe_result.segments[0].speaker == "Speaker_0"

    def test_timestamp_format_identical(self):
        """Test that Assembly and Podscribe use identical timestamp formats."""
        assembly_parser = AssemblyParser()
        podscribe_parser = PodscribeParser()

        # Both formats use same timestamp style
        assembly_transcript = """Speaker A (1m 30s):
Test."""
        podscribe_transcript = """0 (1m 30s):
Test."""

        assembly_result = assembly_parser.parse(assembly_transcript)
        podscribe_result = podscribe_parser.parse(podscribe_transcript)

        # Timestamps should be identical
        assert assembly_result.segments[0].timestamp_seconds == 90
        assert podscribe_result.segments[0].timestamp_seconds == 90


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestTranscriptParserIntegration:
    """Integration tests for transcript parsing."""

    def test_assembly_transcript_end_to_end(self):
        """Test full Assembly transcript parsing workflow."""
        parser = TranscriptParser()

        # Real-world Assembly transcript sample
        transcript = """Speaker A (0s):
Hi, I'm Solana Pyne.

Speaker B (2s):
I'm the director of video at the New York Times.

Speaker A (5s):
For years, my team has made videos.

Speaker B (6s):
That bring you closer to big news moments, videos by Times journalists that have the expertise to help you understand what's going on.

Speaker A (14s):
Now we're bringing those videos to you.

Speaker B (15s):
In the Watch tab in the New York Times app.

Speaker A (18s):
It's a dedicated video feed where you know you can trust what you're seeing.

Speaker B (22s):
All the videos there are free for anyone to watch."""

        result = parser.parse(transcript, format="assembly")

        # Verify parsing
        assert len(result.segments) == 8
        assert result.segments[0].speaker == "Speaker_A"
        assert result.segments[1].speaker == "Speaker_B"

        # Verify content is preserved
        assert "Solana Pyne" in result.full_text
        assert "New York Times" in result.full_text
        assert "video feed" in result.full_text

        # Verify timestamps
        assert result.segments[0].timestamp_seconds == 0
        assert result.segments[7].timestamp_seconds == 22


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
