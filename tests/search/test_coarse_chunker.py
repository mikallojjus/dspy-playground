"""
Unit tests for CoarseChunker.

Tests:
- Basic chunking with small transcript
- Large transcript chunking (4-hour podcast simulation)
- Overlap verification
- Empty/edge case handling
- Token estimation accuracy
- Chunk position finding
- Statistics calculation
"""

import pytest
from src.search.coarse_chunker import CoarseChunker, TranscriptChunk


class TestCoarseChunker:
    """Test suite for CoarseChunker."""

    def test_initialization(self):
        """Test chunker initialization with default and custom settings."""
        # Default settings
        chunker = CoarseChunker()
        assert chunker.chunk_size == 3000
        assert chunker.overlap == 500
        assert chunker.chunk_chars == 3000 * 4
        assert chunker.overlap_chars == 500 * 4

        # Custom settings
        chunker_custom = CoarseChunker(chunk_size=2000, overlap=300)
        assert chunker_custom.chunk_size == 2000
        assert chunker_custom.overlap == 300

    def test_token_estimation(self):
        """Test token count estimation."""
        chunker = CoarseChunker()

        # Empty text
        assert chunker.estimate_token_count("") == 0

        # Known length text
        text = "a" * 400  # 400 chars
        estimated = chunker.estimate_token_count(text)
        assert estimated == 100  # 400 / 4 = 100 tokens

        # Realistic text
        realistic = "This is a sample podcast transcript with normal English words."
        estimated = chunker.estimate_token_count(realistic)
        expected = len(realistic) // 4
        assert estimated == expected

    def test_empty_transcript(self):
        """Test handling of empty transcript."""
        chunker = CoarseChunker()

        # Empty string
        chunks = chunker.chunk_transcript("")
        assert len(chunks) == 0

        # Whitespace only
        chunks = chunker.chunk_transcript("   \n\n  ")
        assert len(chunks) == 0

    def test_small_transcript(self):
        """Test chunking of small transcript (smaller than chunk size)."""
        chunker = CoarseChunker(chunk_size=1000, overlap=200)

        # Small transcript (500 tokens = 2000 chars)
        small_text = "a" * 2000
        chunks = chunker.chunk_transcript(small_text)

        # Should create only 1 chunk
        assert len(chunks) == 1
        assert chunks[0].text == small_text
        assert chunks[0].start_pos == 0
        assert chunks[0].end_pos == 2000
        assert chunks[0].chunk_index == 0
        assert chunks[0].token_count == 500

    def test_medium_transcript_with_overlap(self):
        """Test chunking with overlap on medium transcript."""
        chunker = CoarseChunker(chunk_size=1000, overlap=200)

        # Medium transcript (2500 tokens = 10000 chars)
        # Should create 3-4 chunks with overlap
        medium_text = "a" * 10000

        chunks = chunker.chunk_transcript(medium_text)

        # Verify we have multiple chunks
        assert len(chunks) >= 3

        # Verify chunk sizes are approximately correct
        for chunk in chunks[:-1]:  # All except last
            assert chunk.token_count >= 900  # Should be close to 1000
            assert chunk.token_count <= 1100

        # Verify overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]

            # Chunk 2 should start before chunk 1 ends (overlap)
            assert chunk2.start_pos < chunk1.end_pos

            # Calculate overlap
            overlap_start = chunk2.start_pos
            overlap_end = chunk1.end_pos
            overlap_chars = overlap_end - overlap_start

            # Overlap should be approximately 200 tokens = 800 chars
            assert overlap_chars >= 600  # Allow some tolerance
            assert overlap_chars <= 1000

    def test_large_transcript_4hour_podcast(self):
        """Test chunking of large transcript (4-hour podcast simulation)."""
        chunker = CoarseChunker(chunk_size=3000, overlap=500)

        # 4-hour podcast: ~72,000 tokens = ~288,000 chars
        # Should create ~24 chunks
        large_text = "x" * 288000

        chunks = chunker.chunk_transcript(large_text)

        # Verify chunk count is in expected range
        assert 20 <= len(chunks) <= 30
        print(f"Created {len(chunks)} chunks for 4-hour podcast")

        # Verify all chunks except last are approximately chunk_size
        for i, chunk in enumerate(chunks[:-1]):
            assert 2800 <= chunk.token_count <= 3200
            assert chunk.chunk_index == i

        # Verify sequential start positions
        for i in range(len(chunks) - 1):
            assert chunks[i].start_pos < chunks[i + 1].start_pos

        # Verify coverage (last chunk should end at transcript end)
        assert chunks[-1].end_pos == len(large_text)

    def test_chunk_indices(self):
        """Test that chunk indices are sequential."""
        chunker = CoarseChunker(chunk_size=1000, overlap=200)

        text = "a" * 15000  # Large enough for multiple chunks
        chunks = chunker.chunk_transcript(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_find_chunk_containing_position(self):
        """Test finding chunks that contain a specific position."""
        chunker = CoarseChunker(chunk_size=1000, overlap=200)

        text = "a" * 10000
        chunks = chunker.chunk_transcript(text)

        # Test position in first chunk
        containing = chunker.find_chunk_containing_position(chunks, 100)
        assert len(containing) >= 1
        assert chunks[0] in containing

        # Test position in overlap region (should be in 2 chunks)
        # Find overlap region between chunks[0] and chunks[1]
        if len(chunks) > 1:
            overlap_pos = chunks[1].start_pos + 100  # Position in overlap
            containing = chunker.find_chunk_containing_position(chunks, overlap_pos)

            # Should be in at least 1 chunk (possibly 2 due to overlap)
            assert len(containing) >= 1

        # Test position in last chunk
        last_pos = chunks[-1].start_pos + 100
        containing = chunker.find_chunk_containing_position(chunks, last_pos)
        assert len(containing) >= 1
        assert chunks[-1] in containing

        # Test position outside transcript
        containing = chunker.find_chunk_containing_position(chunks, len(text) + 1000)
        assert len(containing) == 0

    def test_get_statistics(self):
        """Test statistics calculation."""
        chunker = CoarseChunker(chunk_size=1000, overlap=200)

        text = "a" * 10000
        chunks = chunker.chunk_transcript(text)

        stats = chunker.get_statistics(chunks)

        assert stats["total_chunks"] == len(chunks)
        assert stats["total_tokens"] > 0
        assert stats["avg_tokens_per_chunk"] > 0
        assert stats["total_chars"] == len(text)
        assert stats["transcript_length"] == len(text)
        assert stats["chunk_size_tokens"] == 1000
        assert stats["overlap_tokens"] == 200

        # Test empty chunks
        empty_stats = chunker.get_statistics([])
        assert empty_stats["total_chunks"] == 0
        assert empty_stats["total_tokens"] == 0

    def test_transcript_chunk_to_dict(self):
        """Test TranscriptChunk serialization to dict."""
        chunk = TranscriptChunk(
            text="sample text",
            start_pos=0,
            end_pos=11,
            token_count=3,
            chunk_index=0
        )

        chunk_dict = chunk.to_dict()

        assert chunk_dict["text"] == "sample text"
        assert chunk_dict["start_pos"] == 0
        assert chunk_dict["end_pos"] == 11
        assert chunk_dict["token_count"] == 3
        assert chunk_dict["chunk_index"] == 0

    def test_realistic_podcast_text(self):
        """Test with realistic podcast transcript text."""
        chunker = CoarseChunker(chunk_size=500, overlap=100)

        # Realistic podcast transcript excerpt (repeated for size)
        realistic_text = """
        Speaker 1 (0s): Welcome to the podcast today. We're discussing cryptocurrency markets.

        Speaker 2 (15s): Bitcoin has been showing interesting patterns lately. The price action
        suggests we might see a breakout soon.

        Speaker 1 (45s): What about Ethereum? How is it performing?

        Speaker 2 (60s): Ethereum is holding strong. The upgrade has been successful.
        """ * 50  # Repeat to get enough text

        chunks = chunker.chunk_transcript(realistic_text)

        # Verify we have multiple chunks
        assert len(chunks) > 1

        # Verify chunks contain actual text (not just markers)
        for chunk in chunks:
            assert len(chunk.text) > 0
            assert chunk.token_count > 0

        # Verify all chunks are represented in dict form
        chunk_dicts = [c.to_dict() for c in chunks]
        assert len(chunk_dicts) == len(chunks)

    def test_chunk_size_accuracy(self):
        """Test that chunk sizes are approximately as specified."""
        chunker = CoarseChunker(chunk_size=2000, overlap=400)

        # Large uniform text
        text = "word " * 50000  # Many words

        chunks = chunker.chunk_transcript(text)

        # All chunks except possibly last should be close to target size
        for chunk in chunks[:-1]:
            # Should be within 10% of target (2000 tokens)
            assert 1800 <= chunk.token_count <= 2200

    def test_no_chunk_duplication(self):
        """Test that no text is lost or duplicated (except in overlap)."""
        chunker = CoarseChunker(chunk_size=1000, overlap=200)

        text = "a" * 10000
        chunks = chunker.chunk_transcript(text)

        # Verify coverage: first chunk starts at 0, last chunk ends at len(text)
        assert chunks[0].start_pos == 0
        assert chunks[-1].end_pos == len(text)

        # Verify no gaps: each chunk should start at or before previous chunk ends
        for i in range(len(chunks) - 1):
            # Due to overlap, next chunk starts before current ends
            assert chunks[i + 1].start_pos <= chunks[i].end_pos


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
