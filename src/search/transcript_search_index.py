"""
Transcript search index for semantic quote finding.

Builds an in-memory search index from transcript segments with embeddings
for fast semantic similarity search.

Usage:
    from src.search.transcript_search_index import TranscriptSearchIndex
    from src.preprocessing.transcript_parser import TranscriptParser
    from src.infrastructure.embedding_service import EmbeddingService

    parser = TranscriptParser()
    embedder = EmbeddingService()

    parsed = parser.parse(transcript)
    index = await TranscriptSearchIndex.build_from_transcript(
        parsed, embedder
    )

    # Search for quotes
    candidates = await index.find_quotes_for_claim(
        "Bitcoin reached $69,000 in November 2021",
        top_k=30
    )
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from src.infrastructure.embedding_service import EmbeddingService
from src.preprocessing.transcript_parser import ParsedTranscript, TranscriptSegment
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuoteCandidate:
    """
    A quote candidate from semantic search.

    Attributes:
        quote_text: The quote text
        similarity_score: Cosine similarity score with claim (0.0-1.0)
        start_position: Character position in original transcript
        end_position: Character position in original transcript
        speaker: Speaker identifier
        timestamp_seconds: Timestamp in seconds
    """
    quote_text: str
    similarity_score: float
    start_position: int
    end_position: int
    speaker: str
    timestamp_seconds: int


@dataclass
class IndexedSegment:
    """
    Internal representation of an indexed transcript segment.

    Attributes:
        text: Segment text
        embedding: 768-dimensional embedding vector
        start_position: Start position in original transcript
        end_position: End position in original transcript
        speaker: Speaker identifier
        timestamp_seconds: Timestamp in seconds
    """
    text: str
    embedding: List[float]
    start_position: int
    end_position: int
    speaker: str
    timestamp_seconds: int


class TranscriptSearchIndex:
    """
    In-memory semantic search index for transcript segments.

    Features:
    - Windowed segmentation (2-3 sentences per segment)
    - Embedding-based semantic search
    - Fast cosine similarity search
    - Position and speaker tracking

    Example:
        ```python
        parser = TranscriptParser()
        embedder = EmbeddingService()

        parsed = parser.parse(transcript)
        index = await TranscriptSearchIndex.build_from_transcript(
            parsed, embedder
        )

        # Search for quotes related to a claim
        candidates = await index.find_quotes_for_claim(
            "Bitcoin reached all-time high",
            top_k=10
        )

        for candidate in candidates:
            print(f"{candidate.speaker} ({candidate.similarity_score:.3f}): "
                  f"{candidate.quote_text[:50]}...")
        ```
    """

    # Sentence splitting pattern
    SENTENCE_PATTERN = re.compile(r'([.!?])\s+')

    def __init__(
        self,
        segments: List[IndexedSegment],
        embedder: EmbeddingService
    ):
        """
        Initialize the search index.

        Args:
            segments: List of indexed segments with embeddings
            embedder: Embedding service for query embedding
        """
        self.segments = segments
        self.embedder = embedder

        logger.info(f"Initialized TranscriptSearchIndex with {len(segments)} segments")

    @classmethod
    async def build_from_transcript(
        cls,
        parsed_transcript: ParsedTranscript,
        embedder: EmbeddingService,
        window_sentences: int = 2,
        window_overlap: int = 1
    ) -> 'TranscriptSearchIndex':
        """
        Build search index from parsed transcript.

        Args:
            parsed_transcript: Parsed transcript with segments
            embedder: Embedding service
            window_sentences: Number of sentences per window (default: 2)
            window_overlap: Overlap between windows in sentences (default: 1)

        Returns:
            TranscriptSearchIndex ready for searching

        Example:
            ```python
            parser = TranscriptParser()
            embedder = EmbeddingService()

            parsed = parser.parse(transcript)
            index = await TranscriptSearchIndex.build_from_transcript(
                parsed, embedder
            )

            print(f"Built index with {len(index.segments)} searchable segments")
            ```
        """
        logger.info(
            f"Building search index from {len(parsed_transcript.segments)} "
            f"transcript segments (window={window_sentences} sentences, "
            f"overlap={window_overlap})"
        )

        # Create windowed segments
        windowed_segments = cls._create_windowed_segments(
            parsed_transcript.segments,
            window_sentences=window_sentences,
            window_overlap=window_overlap
        )

        logger.info(f"Created {len(windowed_segments)} windowed segments")

        # Generate embeddings for all segments
        segment_texts = [seg.clean_text for seg in windowed_segments]
        embeddings = await embedder.embed_texts(segment_texts)

        # Create indexed segments
        indexed_segments = [
            IndexedSegment(
                text=seg.clean_text,
                embedding=embedding,
                start_position=seg.start_position,
                end_position=seg.end_position,
                speaker=seg.speaker,
                timestamp_seconds=seg.timestamp_seconds
            )
            for seg, embedding in zip(windowed_segments, embeddings)
        ]

        logger.info(
            f"Built search index with {len(indexed_segments)} indexed segments "
            f"({sum(len(s.text) for s in indexed_segments)} total chars)"
        )

        return cls(indexed_segments, embedder)

    @classmethod
    def _create_windowed_segments(
        cls,
        transcript_segments: List[TranscriptSegment],
        window_sentences: int = 2,
        window_overlap: int = 1
    ) -> List[TranscriptSegment]:
        """
        Create windowed segments from transcript segments.

        Splits each segment into sentences, then creates overlapping windows.

        Args:
            transcript_segments: Original transcript segments
            window_sentences: Sentences per window
            window_overlap: Overlap between windows

        Returns:
            List of windowed segments
        """
        windowed: List[TranscriptSegment] = []

        for seg in transcript_segments:
            # Split segment into sentences
            sentences = cls._split_into_sentences(seg.clean_text)

            if not sentences:
                continue

            # Create sliding windows
            for i in range(0, len(sentences), window_sentences - window_overlap):
                # Get window of sentences
                window = sentences[i:i + window_sentences]

                if not window:
                    break

                # Combine sentences
                window_text = " ".join(window)

                # Estimate positions (approximate, since we split by sentences)
                # We'll use the original segment's start position + offset
                windowed_segment = TranscriptSegment(
                    speaker=seg.speaker,
                    clean_text=window_text,
                    start_position=seg.start_position,  # Approximate
                    end_position=seg.start_position + len(window_text),
                    timestamp_seconds=seg.timestamp_seconds
                )

                windowed.append(windowed_segment)

        return windowed

    @classmethod
    def _split_into_sentences(cls, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Split on sentence endings
        sentences = cls.SENTENCE_PATTERN.split(text)

        # Recombine punctuation with sentences
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence:
                result.append(sentence + punct)

        # Add last sentence if it doesn't end with punctuation
        if len(sentences) % 2 == 1:
            last = sentences[-1].strip()
            if last:
                result.append(last)

        return result

    async def find_quotes_for_claim(
        self,
        claim_text: str,
        top_k: int = 30
    ) -> List[QuoteCandidate]:
        """
        Find top K most similar quotes for a claim.

        Args:
            claim_text: The claim to find quotes for
            top_k: Number of top candidates to return

        Returns:
            List of quote candidates sorted by similarity (highest first)

        Example:
            ```python
            candidates = await index.find_quotes_for_claim(
                "Bitcoin reached $69,000 in November 2021",
                top_k=10
            )

            print(f"Found {len(candidates)} candidates:")
            for i, cand in enumerate(candidates, 1):
                print(f"{i}. [{cand.similarity_score:.3f}] {cand.quote_text[:60]}...")
            ```
        """
        logger.debug(f"Searching for quotes for claim: {claim_text[:60]}...")

        # Generate embedding for claim
        claim_embedding = await self.embedder.embed_text(claim_text)

        # Calculate similarity scores for all segments
        candidates: List[QuoteCandidate] = []

        for seg in self.segments:
            similarity = self.embedder.cosine_similarity(
                claim_embedding,
                seg.embedding
            )

            candidate = QuoteCandidate(
                quote_text=seg.text,
                similarity_score=similarity,
                start_position=seg.start_position,
                end_position=seg.end_position,
                speaker=seg.speaker,
                timestamp_seconds=seg.timestamp_seconds
            )

            candidates.append(candidate)

        # Sort by similarity (highest first)
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)

        # Return top K
        top_candidates = candidates[:top_k]

        logger.debug(
            f"Found {len(top_candidates)} candidates "
            f"(similarity range: {top_candidates[0].similarity_score:.3f} - "
            f"{top_candidates[-1].similarity_score:.3f})"
        )

        return top_candidates

    def get_segment_count(self) -> int:
        """
        Get number of indexed segments.

        Returns:
            Number of segments in index
        """
        return len(self.segments)

    def get_total_text_length(self) -> int:
        """
        Get total character length of all indexed segments.

        Returns:
            Total characters
        """
        return sum(len(seg.text) for seg in self.segments)
