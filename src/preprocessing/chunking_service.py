"""
Chunking service for splitting transcripts into manageable chunks.

Splits text into chunks with overlap for LLM processing, respecting sentence boundaries.

Usage:
    from src.preprocessing.chunking_service import ChunkingService

    chunker = ChunkingService(max_chunk_size=16000, overlap=1000)
    chunks = chunker.chunk_text(transcript)

    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}: {len(chunk.text)} chars")
"""

import re
from dataclasses import dataclass
from typing import List

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """
    A chunk of text with position tracking.

    Attributes:
        chunk_id: Unique identifier for this chunk (0-indexed)
        text: The chunk text
        start_position: Character position where chunk starts in original text
        end_position: Character position where chunk ends in original text
    """
    chunk_id: int
    text: str
    start_position: int
    end_position: int


class ChunkingService:
    """
    Text chunking service with sentence-boundary detection.

    Features:
    - Configurable chunk size and overlap
    - Sentence-boundary detection (doesn't cut mid-sentence)
    - Position tracking for quote extraction
    - Handles edge cases (short texts, no sentences)

    Example:
        ```python
        chunker = ChunkingService(max_chunk_size=16000, overlap=1000)
        chunks = chunker.chunk_text(long_transcript)

        # Verify overlap
        assert chunks[1].text[:1000] == chunks[0].text[-1000:]

        # Verify positions
        for chunk in chunks:
            assert chunk.end_position - chunk.start_position == len(chunk.text)
        ```
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]\s+')

    def __init__(
        self,
        max_chunk_size: int = None,
        overlap: int = None,
        boundary_margin: int = 500
    ):
        """
        Initialize the chunking service.

        Args:
            max_chunk_size: Maximum characters per chunk (default from settings)
            overlap: Overlap between chunks in characters (default from settings)
            boundary_margin: Look for sentence boundaries within this margin
                           (default: 500 chars before/after target position)
        """
        self.max_chunk_size = max_chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.boundary_margin = boundary_margin

        logger.info(
            f"Initialized ChunkingService: chunk_size={self.max_chunk_size}, "
            f"overlap={self.overlap}, boundary_margin={self.boundary_margin}"
        )

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks with overlap and sentence-boundary detection.

        Args:
            text: Text to chunk

        Returns:
            List of TextChunk objects

        Example:
            ```python
            chunker = ChunkingService(max_chunk_size=1000, overlap=100)
            text = "First sentence. Second sentence. " * 100
            chunks = chunker.chunk_text(text)

            print(f"Split into {len(chunks)} chunks")
            for chunk in chunks:
                print(f"  Chunk {chunk.chunk_id}: {len(chunk.text)} chars")
            ```
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        text_length = len(text)

        # If text is smaller than max chunk size, return as single chunk
        if text_length <= self.max_chunk_size:
            logger.info(f"Text ({text_length} chars) fits in single chunk")
            return [
                TextChunk(
                    chunk_id=0,
                    text=text,
                    start_position=0,
                    end_position=text_length
                )
            ]

        chunks: List[TextChunk] = []
        current_position = 0
        chunk_id = 0

        while current_position < text_length:
            # Calculate chunk end position (target)
            target_end = current_position + self.max_chunk_size

            # If this is the last chunk, take everything
            if target_end >= text_length:
                chunk_text = text[current_position:]
                chunk_end = text_length
            else:
                # Find sentence boundary near target position
                chunk_end = self._find_sentence_boundary(
                    text,
                    target_end,
                    self.boundary_margin
                )

                # Extract chunk text
                chunk_text = text[current_position:chunk_end]

            # Create chunk
            chunk = TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_position=current_position,
                end_position=chunk_end
            )

            chunks.append(chunk)

            logger.debug(
                f"Created chunk {chunk_id}: "
                f"pos {current_position}-{chunk_end} ({len(chunk_text)} chars)"
            )

            # Move position forward (with overlap)
            # Next chunk starts at (current_end - overlap)
            current_position = chunk_end - self.overlap

            # Prevent infinite loop (should not happen, but safety check)
            if current_position <= chunks[-1].start_position:
                current_position = chunk_end

            chunk_id += 1

            # Safety: Don't create too many chunks
            if chunk_id > 1000:
                logger.error(
                    f"Too many chunks ({chunk_id}) - possible infinite loop. "
                    f"Stopping chunking."
                )
                break

        logger.info(
            f"Split text ({text_length} chars) into {len(chunks)} chunks "
            f"(avg {text_length // len(chunks)} chars/chunk)"
        )

        return chunks

    def _find_sentence_boundary(
        self,
        text: str,
        target_position: int,
        margin: int
    ) -> int:
        """
        Find sentence boundary near target position.

        Looks for sentence endings (. ! ?) within margin of target position.
        If no boundary found, returns target position.

        Args:
            text: Full text
            target_position: Desired position to split at
            margin: Look within this many chars before/after target

        Returns:
            Position of sentence boundary (or target if none found)
        """
        # Define search range
        search_start = max(0, target_position - margin)
        search_end = min(len(text), target_position + margin)

        # Extract search region
        search_text = text[search_start:search_end]

        # Find all sentence endings in search region
        endings = list(self.SENTENCE_ENDINGS.finditer(search_text))

        if not endings:
            # No sentence boundaries found, use target position
            logger.debug(
                f"No sentence boundary found near {target_position}, "
                f"using target position"
            )
            return target_position

        # Find ending closest to target position
        target_offset = target_position - search_start

        best_match = None
        best_distance = float('inf')

        for match in endings:
            # Position of sentence ending (after the punctuation + space)
            ending_pos = match.end()

            # Distance from target
            distance = abs(ending_pos - target_offset)

            if distance < best_distance:
                best_distance = distance
                best_match = match

        if best_match:
            # Return absolute position in original text
            boundary_pos = search_start + best_match.end()
            logger.debug(
                f"Found sentence boundary at {boundary_pos} "
                f"({best_distance} chars from target {target_position})"
            )
            return boundary_pos

        return target_position

    def get_chunk_overlap_text(self, chunk1: TextChunk, chunk2: TextChunk) -> str:
        """
        Get the overlapping text between two consecutive chunks.

        Args:
            chunk1: First chunk
            chunk2: Second chunk

        Returns:
            Overlapping text (empty if no overlap)

        Example:
            ```python
            chunker = ChunkingService(max_chunk_size=1000, overlap=100)
            chunks = chunker.chunk_text(text)

            if len(chunks) > 1:
                overlap = chunker.get_chunk_overlap_text(chunks[0], chunks[1])
                print(f"Overlap: {len(overlap)} chars")
            ```
        """
        if chunk2.start_position >= chunk1.end_position:
            # No overlap
            return ""

        # Calculate overlap region
        overlap_start = chunk2.start_position
        overlap_end = min(chunk1.end_position, chunk2.end_position)

        # Extract from chunk1 (end portion)
        overlap_in_chunk1 = chunk1.text[
            overlap_start - chunk1.start_position:
            overlap_end - chunk1.start_position
        ]

        return overlap_in_chunk1
