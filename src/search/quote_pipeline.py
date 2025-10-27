"""
Quote finding pipeline using coarse-grained retrieval and DSPy extraction.

Architecture:
1. Coarse retrieval: Create 20-30 large chunks (3000 tokens) with embeddings
2. Semantic search: Find top-K most relevant chunks for each claim
3. DSPy extraction: LLM finds quotes from relevant chunks
4. Verification: Multi-tier verification catches hallucinations
5. Entailment: Filter quotes to keep only those that directly support claims

Usage:
    from src.search.quote_pipeline import QuoteFindingPipeline
    from src.preprocessing.transcript_parser import TranscriptParser
    from src.infrastructure.embedding_service import EmbeddingService

    parser = TranscriptParser()
    embedder = EmbeddingService()

    parsed = parser.parse(transcript)
    pipeline = await QuoteFindingPipeline.build_from_transcript(
        parsed, embedder
    )

    # Find quotes for a claim
    quotes = await pipeline.find_quotes_for_claim(
        "Bitcoin reached $69,000 in November 2021"
    )
"""

from dataclasses import dataclass
from typing import List, Optional
import dspy

from src.config.settings import settings
from src.infrastructure.embedding_service import EmbeddingService
from src.preprocessing.transcript_parser import ParsedTranscript, TranscriptSegment
from src.search.coarse_chunker import CoarseChunker, TranscriptChunk
from src.search.llm_quote_finder import QuoteFinder as DSPyQuoteFinder
from src.search.quote_verification import QuoteVerifier
from src.extraction.quote_finder import Quote
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IndexedChunk:
    """
    A coarse-grained chunk with embedding for semantic search.

    Attributes:
        text: Chunk text
        embedding: 768-dimensional embedding vector
        start_pos: Start position in original transcript
        end_pos: End position in original transcript
        chunk_index: Index in the chunking sequence
    """
    text: str
    embedding: List[float]
    start_pos: int
    end_pos: int
    chunk_index: int


class QuoteFindingPipeline:
    """
    Quote finding pipeline using coarse retrieval and DSPy extraction.

    Features:
    - Coarse-grained chunking (3000 tokens, 500 overlap)
    - Top-K chunk retrieval via semantic search
    - DSPy-based quote extraction
    - Multi-tier verification (exact → normalized → fuzzy)
    - Hallucination detection (90% confidence threshold)

    Example:
        ```python
        parser = TranscriptParser()
        embedder = EmbeddingService()

        parsed = parser.parse(transcript)
        pipeline = await QuoteFindingPipeline.build_from_transcript(
            parsed, embedder
        )

        # Find quotes for a claim
        quotes = await pipeline.find_quotes_for_claim(
            "Bitcoin reached all-time high in 2021",
            top_k_chunks=4
        )

        for quote in quotes:
            print(f"{quote.speaker} ({quote.relevance_score:.3f}): "
                  f"{quote.quote_text[:50]}...")
        ```
    """

    def __init__(
        self,
        chunks: List[IndexedChunk],
        parsed_transcript: ParsedTranscript,
        embedder: EmbeddingService,
        dspy_finder: DSPyQuoteFinder,
        verifier: QuoteVerifier
    ):
        """
        Initialize the quote finding pipeline.

        Args:
            chunks: List of coarse-grained chunks with embeddings
            parsed_transcript: Original parsed transcript (for position lookup)
            embedder: Embedding service for query embedding
            dspy_finder: DSPy QuoteFinder module
            verifier: Quote verification service
        """
        self.chunks = chunks
        self.parsed_transcript = parsed_transcript
        self.embedder = embedder
        self.dspy_finder = dspy_finder
        self.verifier = verifier

        logger.info(
            f"Initialized QuoteFindingPipeline with {len(chunks)} coarse chunks"
        )

    @classmethod
    async def build_from_transcript(
        cls,
        parsed_transcript: ParsedTranscript,
        embedder: EmbeddingService,
        dspy_finder: Optional[DSPyQuoteFinder] = None,
        verifier: Optional[QuoteVerifier] = None
    ) -> 'QuoteFindingPipeline':
        """
        Build quote finding pipeline from parsed transcript.

        Args:
            parsed_transcript: Parsed transcript with segments
            embedder: Embedding service
            dspy_finder: Optional DSPy QuoteFinder (creates new if None)
            verifier: Optional QuoteVerifier (creates new if None)

        Returns:
            QuoteFindingPipeline ready for quote finding

        Example:
            ```python
            parser = TranscriptParser()
            embedder = EmbeddingService()

            parsed = parser.parse(transcript)
            pipeline = await QuoteFindingPipeline.build_from_transcript(
                parsed, embedder
            )

            print(f"Built pipeline with {len(pipeline.chunks)} chunks")
            ```
        """
        logger.info("Building QuoteFindingPipeline...")

        # Create coarse chunks
        chunker = CoarseChunker(
            chunk_size=settings.coarse_chunk_size,
            overlap=settings.coarse_chunk_overlap
        )

        transcript_text = parsed_transcript.full_text
        coarse_chunks = chunker.chunk_transcript(transcript_text)

        logger.info(
            f"Created {len(coarse_chunks)} coarse chunks "
            f"(chunk_size={settings.coarse_chunk_size} tokens, "
            f"overlap={settings.coarse_chunk_overlap} tokens)"
        )

        # Generate embeddings for all chunks
        chunk_texts = [chunk.text for chunk in coarse_chunks]
        embeddings = await embedder.embed_texts(chunk_texts)

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Create indexed chunks
        indexed_chunks = [
            IndexedChunk(
                text=chunk.text,
                embedding=embedding,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                chunk_index=chunk.chunk_index
            )
            for chunk, embedding in zip(coarse_chunks, embeddings)
        ]

        # Initialize DSPy finder and verifier
        if dspy_finder is None:
            # Use optimized model if configured, otherwise zero-shot baseline
            model_path = settings.quote_finder_model_path if settings.quote_finder_model_path else None
            dspy_finder = DSPyQuoteFinder(model_path=model_path)

        if verifier is None:
            verifier = QuoteVerifier(
                min_confidence=settings.quote_verification_min_confidence
            )

        logger.info(
            f"Built QuoteFindingPipeline with {len(indexed_chunks)} chunks "
            f"(ready for semantic search)"
        )

        return cls(
            indexed_chunks,
            parsed_transcript,
            embedder,
            dspy_finder,
            verifier
        )

    async def find_quotes_for_claim(
        self,
        claim_text: str,
        top_k_chunks: Optional[int] = None
    ) -> List[Quote]:
        """
        Find supporting quotes for a claim using coarse retrieval and DSPy extraction.

        Process:
        1. Embed the claim
        2. Find top-K most relevant chunks via cosine similarity
        3. Feed claim + chunks to DSPy QuoteFinder
        4. Verify each quote with QuoteVerifier (catch hallucinations)
        5. Look up speaker/timestamp from original transcript
        6. Return Quote objects

        Args:
            claim_text: The claim to find quotes for
            top_k_chunks: Number of top chunks to retrieve (default: from settings)

        Returns:
            List of verified Quote objects with positions and metadata

        Example:
            ```python
            quotes = await pipeline.find_quotes_for_claim(
                "Bitcoin reached $69,000 in November 2021",
                top_k_chunks=4
            )

            print(f"Found {len(quotes)} verified quotes:")
            for quote in quotes:
                print(f"  [{quote.relevance_score:.3f}] {quote.quote_text[:60]}...")
            ```
        """
        if top_k_chunks is None:
            top_k_chunks = settings.top_k_chunks

        logger.debug(f"Finding quotes for claim: {claim_text[:60]}...")

        # Step 1: Embed the claim
        claim_embedding = await self.embedder.embed_text(claim_text)

        # Step 2: Find top-K most relevant chunks
        chunk_scores = []

        for chunk in self.chunks:
            similarity = self.embedder.cosine_similarity(
                claim_embedding,
                chunk.embedding
            )
            chunk_scores.append((chunk, similarity))

        # Sort by similarity (highest first)
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-K chunks
        top_chunks = chunk_scores[:top_k_chunks]

        logger.debug(
            f"Retrieved top {top_k_chunks} chunks "
            f"(similarity range: {top_chunks[0][1]:.3f} - {top_chunks[-1][1]:.3f})"
        )

        # Step 3: Combine chunks into context
        transcript_chunks = "\n\n".join(
            f"[Chunk {chunk.chunk_index}]\n{chunk.text}"
            for chunk, score in top_chunks
        )

        ## DEBUG - Log LLM input
        logger.info("=" * 80)
        logger.info(f"📝 QUOTE FINDING REQUEST FOR CLAIM:")
        logger.info(f"   Claim: {claim_text}")
        logger.info(f"   Context size: {len(transcript_chunks)} chars ({len(transcript_chunks.split())} words)")
        logger.info(f"   Chunks sent: {[chunk.chunk_index for chunk, _ in top_chunks]}")
        logger.info(f"\n   FULL CONTEXT SENT TO LLM:\n{transcript_chunks}")
        logger.info("=" * 80)
        ## DEBUG - End log LLM input

        # Step 4: DSPy extraction
        logger.debug("Running DSPy QuoteFinder...")
        try:
            prediction = self.dspy_finder(
                claim=claim_text,
                transcript_chunks=transcript_chunks
            )

            # Extract quotes from prediction
            if not hasattr(prediction, 'quotes') or not isinstance(prediction.quotes, list):
                logger.warning("DSPy returned invalid prediction format")
                return []

            predicted_quotes = prediction.quotes

        except Exception as e:
            logger.warning(f"DSPy extraction failed: {e}")
            return []

        ## DEBUG - Log LLM output
        logger.info("=" * 80)
        logger.info(f"🤖 LLM RESPONSE:")
        logger.info(f"   Extracted {len(predicted_quotes)} quotes\n")
        for i, quote in enumerate(predicted_quotes, 1):
            quote_text = quote.get('text', quote) if isinstance(quote, dict) else str(quote)
            quote_reasoning = quote.get('reasoning', 'N/A') if isinstance(quote, dict) else 'N/A'
            logger.info(f"   Quote {i}:")
            logger.info(f"      Text: {quote_text}")
            logger.info(f"      Reasoning: {quote_reasoning}\n")
        logger.info("=" * 80)
        ## DEBUG - End log LLM output

        # Step 5: Verify each quote
        verified_quotes = []
        full_transcript = self.parsed_transcript.full_text

        for quote_data in predicted_quotes:
            # Parse quote data (should be dict with 'text' field)
            if isinstance(quote_data, dict) and 'text' in quote_data:
                quote_text = quote_data['text']
            elif isinstance(quote_data, str):
                quote_text = quote_data
            else:
                logger.warning(f"Invalid quote format: {quote_data}")
                continue

            # Verify quote exists in transcript
            verification_result = self.verifier.verify(
                quote_text=quote_text,
                transcript=full_transcript,
                claim_text=claim_text
            )

            if not verification_result.is_valid:
                logger.debug(f"Quote failed verification: {quote_text[:50]}...")
                continue

            if verification_result.confidence < settings.quote_verification_min_confidence:
                logger.debug(
                    f"Quote below confidence threshold "
                    f"({verification_result.confidence:.2f} < {settings.quote_verification_min_confidence}): "
                    f"{quote_text[:50]}..."
                )
                continue

            # Ensure positions are set (should always be true if is_valid is True)
            assert verification_result.start_pos is not None, "Valid quote must have start_pos"
            assert verification_result.end_pos is not None, "Valid quote must have end_pos"

            # Step 6: Look up speaker and timestamp
            speaker, timestamp = self._lookup_speaker_and_timestamp(
                verification_result.start_pos,
                verification_result.end_pos
            )

            # Create Quote object
            quote = Quote(
                quote_text=verification_result.corrected_text,
                relevance_score=verification_result.confidence,
                start_position=verification_result.start_pos,
                end_position=verification_result.end_pos,
                speaker=speaker,
                timestamp_seconds=timestamp
            )

            verified_quotes.append(quote)

        logger.debug(
            f"Verified {len(verified_quotes)} quotes "
            f"({len(predicted_quotes) - len(verified_quotes)} failed verification)"
        )

        return verified_quotes

    def _lookup_speaker_and_timestamp(
        self,
        start_pos: int,
        end_pos: int
    ) -> tuple[str, int]:
        """
        Look up speaker and timestamp for a quote position.

        Args:
            start_pos: Quote start position
            end_pos: Quote end position

        Returns:
            (speaker, timestamp_seconds)
        """
        # Find the segment that contains this quote
        for segment in self.parsed_transcript.segments:
            # Check if quote is within this segment
            if (start_pos >= segment.start_position and
                start_pos < segment.end_position):
                return segment.speaker, segment.timestamp_seconds

        # Default if not found
        return "Unknown", 0

    def get_chunk_count(self) -> int:
        """
        Get number of indexed chunks.

        Returns:
            Number of chunks
        """
        return len(self.chunks)

    def get_total_text_length(self) -> int:
        """
        Get total character length of all chunks.

        Returns:
            Total characters
        """
        return sum(len(chunk.text) for chunk in self.chunks)
