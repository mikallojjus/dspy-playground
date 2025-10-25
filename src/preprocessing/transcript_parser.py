"""
Transcript parser for extracting speakers, timestamps, and text segments.

Parses podcast transcript format:
    <speaker_number> (<timestamp>):
    <text content>

Example:
    0 (30s):
    All right. Hey guys, it's Tim Miller...

    2 (1m 27s):
    I mean, this is one of those...
"""

import re
from dataclasses import dataclass
from typing import List

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptSegment:
    """
    A single speaker segment from the transcript.

    Attributes:
        speaker: Speaker identifier (e.g., "Speaker_0", "Speaker_2")
        clean_text: Text without timestamp markers (for LLM processing)
        start_position: Character position where segment starts in original transcript
        end_position: Character position where segment ends in original transcript
        timestamp_seconds: Timestamp in seconds from start of episode
    """
    speaker: str
    clean_text: str
    start_position: int
    end_position: int
    timestamp_seconds: int


@dataclass
class ParsedTranscript:
    """
    Parsed transcript with all segments.

    Attributes:
        segments: List of all speaker segments
        full_text: Complete text without timestamp markers (concatenated segments)
    """
    segments: List[TranscriptSegment]
    full_text: str


class TranscriptParser:
    """
    Parser for podcast transcripts with speaker and timestamp extraction.

    Handles formats:
    - Timestamps: "2s", "30s", "1m 16s", "21m 33s"
    - Multi-line segments
    - Position tracking for quote extraction

    Example:
        ```python
        parser = TranscriptParser()
        result = parser.parse(transcript)

        print(f"Found {len(result.segments)} segments")
        for seg in result.segments[:3]:
            print(f"{seg.speaker} at {seg.timestamp_seconds}s: {seg.clean_text[:50]}...")
        ```
    """

    # Pattern: <speaker_num> (<timestamp>):
    # Examples: "0 (30s):", "2 (1m 16s):", "1 (2s):"
    SEGMENT_PATTERN = re.compile(
        r'^(\d+)\s+\(([^)]+)\):',
        re.MULTILINE
    )

    # Timestamp patterns: "2s", "1m 16s", "21m 33s", "12m"
    TIMESTAMP_PATTERN = re.compile(
        r'(?:(\d+)m)?(?:\s*(\d+)s)?'
    )

    def parse(self, transcript: str) -> ParsedTranscript:
        """
        Parse transcript into segments with speaker and timestamp information.

        Args:
            transcript: Raw transcript text

        Returns:
            ParsedTranscript with segments and full clean text

        Example:
            ```python
            parser = TranscriptParser()
            transcript = '''
            0 (30s):
            Hello world. This is a test.

            2 (1m 16s):
            Another speaker here.
            '''

            result = parser.parse(transcript)
            assert len(result.segments) == 2
            assert result.segments[0].speaker == "Speaker_0"
            assert result.segments[0].timestamp_seconds == 30
            ```
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided")
            return ParsedTranscript(segments=[], full_text="")

        # Find all segment markers
        matches = list(self.SEGMENT_PATTERN.finditer(transcript))

        if not matches:
            logger.warning("No speaker segments found in transcript")
            # Return entire transcript as single segment
            return ParsedTranscript(
                segments=[
                    TranscriptSegment(
                        speaker="Speaker_Unknown",
                        clean_text=transcript.strip(),
                        start_position=0,
                        end_position=len(transcript),
                        timestamp_seconds=0
                    )
                ],
                full_text=transcript.strip()
            )

        segments: List[TranscriptSegment] = []

        for i, match in enumerate(matches):
            # Extract speaker number and timestamp
            speaker_num = match.group(1)
            timestamp_str = match.group(2)

            # Parse timestamp to seconds
            timestamp_seconds = self._parse_timestamp(timestamp_str)

            # Find start of text (after ":")
            text_start = match.end()

            # Find end of text (start of next segment or end of transcript)
            if i + 1 < len(matches):
                text_end = matches[i + 1].start()
            else:
                text_end = len(transcript)

            # Extract text
            text = transcript[text_start:text_end].strip()

            # Create segment
            segment = TranscriptSegment(
                speaker=f"Speaker_{speaker_num}",
                clean_text=text,
                start_position=text_start,
                end_position=text_end,
                timestamp_seconds=timestamp_seconds
            )

            segments.append(segment)

            logger.debug(
                f"Parsed segment: {segment.speaker} at {timestamp_seconds}s "
                f"({len(text)} chars)"
            )

        # Create full clean text (all segments concatenated)
        full_text = "\n\n".join(seg.clean_text for seg in segments)

        logger.info(
            f"Parsed {len(segments)} segments from transcript "
            f"({len(transcript)} chars → {len(full_text)} clean chars)"
        )

        return ParsedTranscript(segments=segments, full_text=full_text)

    def _parse_timestamp(self, timestamp_str: str) -> int:
        """
        Parse timestamp string to seconds.

        Args:
            timestamp_str: Timestamp like "2s", "1m 16s", "21m 33s", "12m"

        Returns:
            Total seconds

        Example:
            ```python
            parser = TranscriptParser()
            assert parser._parse_timestamp("2s") == 2
            assert parser._parse_timestamp("12m") == 720
            assert parser._parse_timestamp("1m 16s") == 76
            assert parser._parse_timestamp("21m 33s") == 1293
            ```
        """
        match = self.TIMESTAMP_PATTERN.search(timestamp_str)

        if not match:
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return 0

        minutes_str = match.group(1)
        seconds_str = match.group(2)

        # Check if at least one part matched
        if not minutes_str and not seconds_str:
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return 0

        minutes = int(minutes_str) if minutes_str else 0
        seconds = int(seconds_str) if seconds_str else 0

        total_seconds = minutes * 60 + seconds

        return total_seconds

    def get_text_at_position(
        self,
        transcript: str,
        start_pos: int,
        end_pos: int
    ) -> str:
        """
        Extract text from transcript at specific character positions.

        Useful for extracting quotes based on position.

        Args:
            transcript: Original transcript
            start_pos: Start character position
            end_pos: End character position

        Returns:
            Text at specified position

        Example:
            ```python
            parser = TranscriptParser()
            text = parser.get_text_at_position(transcript, 100, 200)
            print(f"Quote: {text}")
            ```
        """
        if start_pos < 0 or end_pos > len(transcript) or start_pos >= end_pos:
            logger.warning(
                f"Invalid position range: {start_pos}-{end_pos} "
                f"(transcript length: {len(transcript)})"
            )
            return ""

        return transcript[start_pos:end_pos].strip()
