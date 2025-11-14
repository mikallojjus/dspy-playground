"""
Transcript parser for extracting speakers, timestamps, and text segments.

Supports multiple transcript formats:
- Podscribe: "0 (30s):\nText here"
- Bankless: "David:\n[0:03] Text here"

Example:
    parser = TranscriptParser()
    result = parser.parse(transcript, format="podscribe")
    print(f"Found {len(result.segments)} segments")
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

TranscriptFormat = Literal["podscribe", "bankless"]


@dataclass
class TranscriptSegment:
    """
    A single speaker segment from the transcript.

    Attributes:
        speaker: Speaker identifier (e.g., "Speaker_0", "David", "Sam")
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


class TranscriptParserProtocol(ABC):
    """
    Abstract base class for transcript parsers.

    Each format (Podscribe, Bankless, etc.) implements this protocol.
    """

    @abstractmethod
    def parse(self, transcript: str) -> ParsedTranscript:
        """
        Parse transcript into segments with speaker and timestamp information.

        Args:
            transcript: Raw transcript text

        Returns:
            ParsedTranscript with segments and full clean text
        """
        pass


class PodscribeParser(TranscriptParserProtocol):
    """
    Parser for Podscribe transcript format.

    Format:
        <speaker_number> (<timestamp>):
        <text content>

    Example:
        0 (30s):
        All right. Hey guys, it's Tim Miller...

        2 (1m 27s):
        I mean, this is one of those...
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
        Parse Podscribe transcript into segments.

        Args:
            transcript: Raw Podscribe transcript text

        Returns:
            ParsedTranscript with segments and full clean text
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided")
            return ParsedTranscript(segments=[], full_text="")

        # Find all segment markers
        matches = list(self.SEGMENT_PATTERN.finditer(transcript))

        if not matches:
            logger.warning("No Podscribe speaker segments found in transcript")
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
            f"Parsed {len(segments)} Podscribe segments from transcript "
            f"({len(transcript)} chars → {len(full_text)} clean chars)"
        )

        return ParsedTranscript(segments=segments, full_text=full_text)

    def _parse_timestamp(self, timestamp_str: str) -> int:
        """
        Parse Podscribe timestamp string to seconds.

        Args:
            timestamp_str: Timestamp like "2s", "1m 16s", "21m 33s", "12m"

        Returns:
            Total seconds
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


class BanklessParser(TranscriptParserProtocol):
    """
    Parser for Bankless transcript format.

    Format:
        <speaker_name>:
        [<timestamp>] <text content>

    Example:
        David:
        [0:03] Bankless Nation, I'm here with Sam Ragsdale...

        Sam:
        [0:08] Excited to be here.
    """

    # Pattern: <speaker_name>:\n[<timestamp>] <text>
    # Examples: "David:\n[0:03]", "Sam:\n[1:23]"
    SEGMENT_PATTERN = re.compile(
        r'^([A-Za-z][A-Za-z0-9\s]*?):\s*\n\s*\[([^\]]+)\]',
        re.MULTILINE
    )

    # Timestamp pattern: "[M:SS]" or "[MM:SS]" or "[H:MM:SS]"
    TIMESTAMP_PATTERN = re.compile(
        r'(?:(\d+):)?(\d+):(\d+)'
    )

    def parse(self, transcript: str) -> ParsedTranscript:
        """
        Parse Bankless transcript into segments.

        Args:
            transcript: Raw Bankless transcript text

        Returns:
            ParsedTranscript with segments and full clean text
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided")
            return ParsedTranscript(segments=[], full_text="")

        # Find all segment markers
        matches = list(self.SEGMENT_PATTERN.finditer(transcript))

        if not matches:
            logger.warning("No Bankless speaker segments found in transcript")
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
            # Extract speaker name and timestamp
            speaker_name = match.group(1).strip()
            timestamp_str = match.group(2)

            # Parse timestamp to seconds
            timestamp_seconds = self._parse_timestamp(timestamp_str)

            # Find start of text (after "[M:SS] ")
            # Text starts after the timestamp bracket + space
            text_start = match.end()

            # Find end of text (start of next segment or end of transcript)
            if i + 1 < len(matches):
                text_end = matches[i + 1].start()
            else:
                text_end = len(transcript)

            # Extract text
            text = transcript[text_start:text_end].strip()

            # Create segment (preserve original speaker name)
            segment = TranscriptSegment(
                speaker=speaker_name,
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
            f"Parsed {len(segments)} Bankless segments from transcript "
            f"({len(transcript)} chars → {len(full_text)} clean chars)"
        )

        return ParsedTranscript(segments=segments, full_text=full_text)

    def _parse_timestamp(self, timestamp_str: str) -> int:
        """
        Parse Bankless timestamp string to seconds.

        Args:
            timestamp_str: Timestamp like "0:03", "1:23", "1:02:45"

        Returns:
            Total seconds
        """
        match = self.TIMESTAMP_PATTERN.search(timestamp_str)

        if not match:
            logger.warning(f"Could not parse Bankless timestamp: {timestamp_str}")
            return 0

        hours_str = match.group(1)
        minutes_str = match.group(2)
        seconds_str = match.group(3)

        hours = int(hours_str) if hours_str else 0
        minutes = int(minutes_str) if minutes_str else 0
        seconds = int(seconds_str) if seconds_str else 0

        total_seconds = hours * 3600 + minutes * 60 + seconds

        return total_seconds


class TranscriptParser:
    """
    Factory/dispatcher for transcript parsers.

    Routes to appropriate parser based on format parameter.

    Example:
        parser = TranscriptParser()
        result = parser.parse(transcript, format="podscribe")
        print(f"Found {len(result.segments)} segments")
    """

    def __init__(self):
        """Initialize parser with all format implementations."""
        self._parsers = {
            "podscribe": PodscribeParser(),
            "bankless": BanklessParser(),
        }

    def parse(self, transcript: str, format: TranscriptFormat = "podscribe") -> ParsedTranscript:
        """
        Parse transcript using specified format parser.

        Args:
            transcript: Raw transcript text
            format: Transcript format ("podscribe" or "bankless")

        Returns:
            ParsedTranscript with segments and full clean text

        Raises:
            ValueError: If format is not supported

        Example:
            parser = TranscriptParser()

            # Parse Podscribe format
            result = parser.parse(podscribe_transcript, format="podscribe")

            # Parse Bankless format
            result = parser.parse(bankless_transcript, format="bankless")
        """
        if format not in self._parsers:
            raise ValueError(
                f"Unsupported transcript format: {format}. "
                f"Supported formats: {list(self._parsers.keys())}"
            )

        parser = self._parsers[format]
        return parser.parse(transcript)

    def get_text_at_position(
        self,
        transcript: str,
        start_pos: int,
        end_pos: int
    ) -> str:
        """
        Extract text from transcript at specific character positions.

        Useful for extracting quotes based on position.
        Format-agnostic utility method.

        Args:
            transcript: Original transcript
            start_pos: Start character position
            end_pos: End character position

        Returns:
            Text at specified position
        """
        if start_pos < 0 or end_pos > len(transcript) or start_pos >= end_pos:
            logger.warning(
                f"Invalid position range: {start_pos}-{end_pos} "
                f"(transcript length: {len(transcript)})"
            )
            return ""

        return transcript[start_pos:end_pos].strip()
