"""
Coarse-grained transcript chunking for efficient quote finding.

Creates large chunks (3000 tokens) with overlap (500 tokens) to:
- Reduce embedding cost by 50x (20-30 chunks vs 900+ segments)
- Preserve narrative context
- Prevent quote boundary issues

Usage:
    from src.search.coarse_chunker import CoarseChunker

    chunker = CoarseChunker(chunk_size=3000, overlap=500)
    chunks = chunker.chunk_transcript(transcript)

    print(f"Created {len(chunks)} chunks")
    for chunk in chunks:
        print(f"Chunk tokens: {chunk['token_count']}, chars: {len(chunk['text'])}")
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from src.infrastructure.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class TranscriptChunk:
    """
    A coarse-grained chunk of transcript.

    Attributes:
        text: The chunk text content
        start_pos: Character position in original transcript where chunk starts
        end_pos: Character position in original transcript where chunk ends
        token_count: Estimated token count for this chunk
        chunk_index: Sequential index of this chunk (0-based)
    """
    text: str
    start_pos: int
    end_pos: int
    token_count: int
    chunk_index: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index
        }


class CoarseChunker:
    """
    Create coarse-grained chunks for efficient semantic search.

    Strategy:
    - Large chunks (3000 tokens ≈ 2-3 minutes of speech)
    - Overlap (500 tokens) to prevent boundary issues
    - Token counting using character approximation (1 token ≈ 4 chars for English)

    Benefits:
    - 50x fewer embeddings: 20-30 chunks vs 900+ segments
    - Better context preservation
    - Faster retrieval

    Example:
        ```python
        chunker = CoarseChunker(chunk_size=3000, overlap=500)
        chunks = chunker.chunk_transcript(transcript)

        # 4-hour podcast: ~72,000 tokens → ~24 chunks
        print(f"Created {len(chunks)} chunks")
        ```
    """

    # Token estimation: rough approximation for English text
    # More accurate than pure word count, conservative for safety
    CHARS_PER_TOKEN = 4  # Conservative estimate (OpenAI uses ~4 chars/token for English)

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ):
        """
        Initialize the coarse chunker.

        Args:
            chunk_size: Target chunk size in tokens (default from settings)
            overlap: Overlap size in tokens (default from settings)
        """
        self.chunk_size = chunk_size or settings.coarse_chunk_size
        self.overlap = overlap or settings.coarse_chunk_overlap

        # Convert to characters for processing
        self.chunk_chars = self.chunk_size * self.CHARS_PER_TOKEN
        self.overlap_chars = self.overlap * self.CHARS_PER_TOKEN

        logger.info(
            f"Initialized CoarseChunker: "
            f"chunk_size={self.chunk_size} tokens ({self.chunk_chars} chars), "
            f"overlap={self.overlap} tokens ({self.overlap_chars} chars)"
        )

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses character-based approximation: 1 token ≈ 4 characters.
        This is conservative and works well for English text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // self.CHARS_PER_TOKEN

    def chunk_transcript(self, transcript: str) -> List[TranscriptChunk]:
        """
        Split transcript into coarse-grained chunks with overlap.

        Algorithm:
        1. Start at position 0
        2. Take chunk_size tokens worth of text
        3. Move forward by (chunk_size - overlap) tokens
        4. Repeat until end of transcript

        Args:
            transcript: Full transcript text

        Returns:
            List of TranscriptChunk objects

        Example:
            ```python
            # 4-hour podcast (72,000 tokens)
            chunks = chunker.chunk_transcript(long_transcript)
            # Result: 24-26 chunks of ~3000 tokens each

            # Verify overlap
            assert chunks[1].start_pos < chunks[0].end_pos
            overlap_size = chunks[0].end_pos - chunks[1].start_pos
            print(f"Overlap: {overlap_size} chars ≈ {overlap_size//4} tokens")
            ```
        """
        if not transcript or len(transcript.strip()) == 0:
            logger.warning("Empty transcript provided, returning empty list")
            return []

        transcript_length = len(transcript)
        estimated_tokens = self.estimate_token_count(transcript)

        logger.info(
            f"Chunking transcript: {transcript_length} chars, "
            f"~{estimated_tokens} tokens"
        )

        chunks = []
        start = 0
        chunk_index = 0

        while start < transcript_length:
            # Calculate end position
            end = min(start + self.chunk_chars, transcript_length)

            # Extract chunk text
            chunk_text = transcript[start:end]

            # Estimate tokens
            token_count = self.estimate_token_count(chunk_text)

            # Create chunk
            chunk = TranscriptChunk(
                text=chunk_text,
                start_pos=start,
                end_pos=end,
                token_count=token_count,
                chunk_index=chunk_index
            )
            chunks.append(chunk)

            logger.debug(
                f"Created chunk {chunk_index}: "
                f"pos=[{start}:{end}], "
                f"tokens={token_count}, "
                f"chars={len(chunk_text)}"
            )

            # Move forward with overlap
            # Step size = chunk_size - overlap
            step_chars = self.chunk_chars - self.overlap_chars
            start = start + step_chars

            chunk_index += 1

            # Stop if we're too close to the end (avoid tiny final chunk)
            if start >= transcript_length - self.overlap_chars:
                break

        logger.info(
            f"Created {len(chunks)} coarse chunks "
            f"(avg {estimated_tokens // len(chunks)} tokens/chunk)"
        )

        return chunks

    def find_chunk_containing_position(
        self,
        chunks: List[TranscriptChunk],
        position: int
    ) -> List[TranscriptChunk]:
        """
        Find all chunks that contain a given character position.

        Due to overlap, multiple chunks may contain the same position.

        Args:
            chunks: List of chunks to search
            position: Character position in original transcript

        Returns:
            List of chunks containing this position (may be multiple due to overlap)

        Example:
            ```python
            chunks = chunker.chunk_transcript(transcript)

            # Find chunks containing position 50000
            containing = chunker.find_chunk_containing_position(chunks, 50000)
            print(f"Position 50000 appears in {len(containing)} chunks")
            # Usually 2 chunks due to overlap
            ```
        """
        containing = [
            chunk for chunk in chunks
            if chunk.start_pos <= position < chunk.end_pos
        ]

        logger.debug(
            f"Position {position} found in {len(containing)} chunks: "
            f"{[c.chunk_index for c in containing]}"
        )

        return containing

    def get_statistics(self, chunks: List[TranscriptChunk]) -> Dict:
        """
        Get statistics about chunked transcript.

        Args:
            chunks: List of chunks

        Returns:
            Dict with statistics

        Example:
            ```python
            chunks = chunker.chunk_transcript(transcript)
            stats = chunker.get_statistics(chunks)

            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Avg tokens/chunk: {stats['avg_tokens_per_chunk']}")
            print(f"Total coverage: {stats['total_chars']} chars")
            ```
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "total_chars": 0,
                "transcript_length": 0
            }

        total_tokens = sum(c.token_count for c in chunks)
        total_chars = chunks[-1].end_pos  # Last chunk's end position

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens // len(chunks),
            "total_chars": total_chars,
            "transcript_length": total_chars,
            "chunk_size_tokens": self.chunk_size,
            "overlap_tokens": self.overlap
        }
