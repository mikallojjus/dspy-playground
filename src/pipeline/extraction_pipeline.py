"""
End-to-end extraction pipeline for claims and quotes.

Orchestrates the full two-pass extraction process:
1. Parse transcript (speakers, timestamps)
2. Chunk transcript for LLM processing
3. Extract claims using DSPy model (Pass 1)
4. Build semantic search index
5. Find supporting quotes for each claim (Pass 2)

Usage:
    from src.pipeline.extraction_pipeline import ExtractionPipeline

    pipeline = ExtractionPipeline()
    result = await pipeline.process_episode(episode_id=123)

    print(f"Extracted {len(result.claims)} claims")
    print(f"Average {result.avg_quotes_per_claim:.1f} quotes per claim")
"""

from dataclasses import dataclass
from typing import List, Optional
import time

from src.database.connection import get_db_session
from src.database.models import PodcastEpisode
from src.preprocessing.transcript_parser import TranscriptParser
from src.preprocessing.chunking_service import ChunkingService
from src.extraction.claim_extractor import ClaimExtractor
from src.search.transcript_search_index import TranscriptSearchIndex
from src.extraction.quote_finder import QuoteFinder, ClaimWithQuotes, Quote
from src.infrastructure.embedding_service import EmbeddingService
from src.infrastructure.reranker_service import RerankerService
from src.deduplication.quote_deduplicator import QuoteDeduplicator
from src.deduplication.claim_deduplicator import ClaimDeduplicator
from src.scoring.confidence_calculator import ConfidenceCalculator
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStats:
    """
    Statistics from pipeline execution.

    Attributes:
        episode_id: Episode ID processed
        transcript_length: Original transcript character count
        chunks_count: Number of chunks created
        claims_extracted: Number of claims extracted (before filtering)
        claims_after_dedup: Number of claims after deduplication
        claims_with_quotes: Number of claims that have at least one quote
        quotes_before_dedup: Number of quotes before deduplication
        quotes_after_dedup: Number of quotes after deduplication
        total_quotes: Total quotes in final result
        avg_quotes_per_claim: Average quotes per claim
        processing_time_seconds: Total processing time
    """
    episode_id: int
    transcript_length: int
    chunks_count: int
    claims_extracted: int
    claims_after_dedup: int
    claims_with_quotes: int
    quotes_before_dedup: int
    quotes_after_dedup: int
    total_quotes: int
    avg_quotes_per_claim: float
    processing_time_seconds: float


@dataclass
class PipelineResult:
    """
    Result from pipeline execution.

    Attributes:
        episode_id: Episode ID processed
        claims: Claims with supporting quotes
        stats: Pipeline statistics
    """
    episode_id: int
    claims: List[ClaimWithQuotes]
    stats: PipelineStats


class ExtractionPipeline:
    """
    End-to-end extraction pipeline orchestrator.

    Coordinates all components to extract claims and quotes from podcast episodes.

    Features:
    - Two-pass extraction (claims → quotes)
    - Semantic search for quote finding
    - Comprehensive error handling
    - Performance tracking

    NOT included in Sprint 2 (coming later):
    - Deduplication (Sprint 3)
    - Entailment validation (Sprint 4)
    - Database persistence (Sprint 4)

    Example:
        ```python
        pipeline = ExtractionPipeline()

        # Process single episode
        result = await pipeline.process_episode(episode_id=123)

        print(f"Episode {result.episode_id}:")
        print(f"  Claims: {len(result.claims)}")
        print(f"  Quotes: {result.stats.total_quotes}")
        print(f"  Time: {result.stats.processing_time_seconds:.1f}s")

        # Display claims
        for claim in result.claims[:5]:
            print(f"\nClaim: {claim.claim_text}")
            print(f"  Quotes: {len(claim.quotes)}")
            for quote in claim.quotes[:2]:
                print(f"    [{quote.relevance_score:.3f}] {quote.quote_text[:50]}...")
        ```
    """

    def __init__(self):
        """
        Initialize the extraction pipeline.

        Creates all service instances needed for processing.
        """
        logger.info("Initializing ExtractionPipeline")

        self.parser = TranscriptParser()
        self.chunker = ChunkingService()
        self.claim_extractor = ClaimExtractor()
        self.embedder = EmbeddingService()
        self.reranker = RerankerService()
        self.quote_deduplicator = QuoteDeduplicator()
        self.claim_deduplicator = ClaimDeduplicator(self.embedder, self.reranker)
        self.confidence_calculator = ConfidenceCalculator()

        logger.info("ExtractionPipeline ready")

    async def process_episode(
        self,
        episode_id: int
    ) -> PipelineResult:
        """
        Process a single episode through the full pipeline.

        Args:
            episode_id: Database ID of episode to process

        Returns:
            PipelineResult with claims, quotes, and statistics

        Raises:
            ValueError: If episode not found or has no transcript
            Exception: If pipeline fails

        Example:
            ```python
            pipeline = ExtractionPipeline()

            try:
                result = await pipeline.process_episode(episode_id=123)
                print(f"✅ Processed episode {result.episode_id}")
                print(f"   {len(result.claims)} claims extracted")
            except ValueError as e:
                print(f"❌ Error: {e}")
            ```
        """
        start_time = time.time()

        logger.info(f"Starting pipeline for episode {episode_id}")

        if self.reranker.enabled:
            await self.reranker.wait_for_ready()

        episode = self._load_episode(episode_id)

        if not episode.transcript:
            raise ValueError(f"Episode {episode_id} has no transcript")

        transcript = episode.transcript
        transcript_length = len(transcript)

        logger.info(
            f"Loaded episode {episode_id}: '{episode.name}' "
            f"({transcript_length} chars)"
        )

        logger.info("Step 1/9: Parsing transcript...")
        parsed_transcript = self.parser.parse(transcript)
        logger.info(f"  ✓ Parsed {len(parsed_transcript.segments)} segments")

        logger.info("Step 2/9: Chunking transcript...")
        chunks = self.chunker.chunk_text(parsed_transcript.full_text)
        logger.info(f"  ✓ Created {len(chunks)} chunks")

        logger.info("Step 3/9: Extracting claims from chunks...")
        claims = await self.claim_extractor.extract_from_chunks(chunks)
        logger.info(f"  ✓ Extracted {len(claims)} claims")

        if not claims:
            logger.warning("No claims extracted, ending pipeline")
            return self._create_empty_result(
                episode_id,
                transcript_length,
                len(chunks),
                time.time() - start_time
            )

        logger.info("Step 4/9: Building search index...")
        search_index = await TranscriptSearchIndex.build_from_transcript(
            parsed_transcript,
            self.embedder
        )
        logger.info(f"  ✓ Indexed {search_index.get_segment_count()} segments")

        logger.info("Step 5/9: Finding quotes for claims...")
        quote_finder = QuoteFinder(search_index, self.reranker)
        claims_with_quotes = await quote_finder.find_quotes_for_claims(claims)
        logger.info(f"  ✓ Found quotes for {len(claims_with_quotes)} claims")

        logger.info("Step 6/9: Deduplicating quotes...")
        all_quotes = [q for c in claims_with_quotes for q in c.quotes]
        unique_quotes = self.quote_deduplicator.deduplicate(all_quotes)
        logger.info(
            f"  ✓ {len(all_quotes)} → {len(unique_quotes)} unique quotes "
            f"({len(all_quotes) - len(unique_quotes)} duplicates removed)"
        )

        claims_with_quotes = self._remap_quotes_to_claims(
            claims_with_quotes, all_quotes, unique_quotes
        )

        logger.info("Step 7/9: Deduplicating claims...")
        deduplicated_claims = await self.claim_deduplicator.deduplicate_batch(
            claims_with_quotes
        )
        logger.info(
            f"  ✓ {len(claims_with_quotes)} → {len(deduplicated_claims)} unique claims "
            f"({len(claims_with_quotes) - len(deduplicated_claims)} duplicates removed)"
        )

        logger.info("Step 8/9: Calculating confidence scores...")
        for claim in deduplicated_claims:
            conf = self.confidence_calculator.calculate(claim.quotes)
            claim.confidence = conf.final_confidence
            claim.confidence_components = conf
        logger.info(f"  ✓ Calculated confidence for {len(deduplicated_claims)} claims")

        logger.info("Step 9/9: Filtering low-confidence claims...")
        from src.config.settings import settings
        high_confidence_claims = [
            c for c in deduplicated_claims
            if c.confidence >= settings.min_confidence
        ]
        logger.info(
            f"  ✓ {len(high_confidence_claims)} claims above threshold "
            f"({len(deduplicated_claims) - len(high_confidence_claims)} filtered)"
        )

        claims_with_quotes = [c for c in high_confidence_claims if c.quotes]

        if not claims_with_quotes:
            logger.warning("No claims have supporting quotes after filtering")
            return self._create_empty_result(
                episode_id,
                transcript_length,
                len(chunks),
                time.time() - start_time
            )

        total_quotes = sum(len(c.quotes) for c in claims_with_quotes)
        avg_quotes = total_quotes / len(claims_with_quotes)
        processing_time = time.time() - start_time

        stats = PipelineStats(
            episode_id=episode_id,
            transcript_length=transcript_length,
            chunks_count=len(chunks),
            claims_extracted=len(claims),
            claims_after_dedup=len(deduplicated_claims),
            claims_with_quotes=len(claims_with_quotes),
            quotes_before_dedup=len(all_quotes),
            quotes_after_dedup=len(unique_quotes),
            total_quotes=total_quotes,
            avg_quotes_per_claim=avg_quotes,
            processing_time_seconds=processing_time
        )

        logger.info(
            f"✅ Pipeline complete for episode {episode_id} "
            f"({processing_time:.1f}s)"
        )
        logger.info(
            f"   📊 {len(claims_with_quotes)} claims, {total_quotes} quotes "
            f"(avg {avg_quotes:.1f} quotes/claim)"
        )

        return PipelineResult(
            episode_id=episode_id,
            claims=claims_with_quotes,
            stats=stats
        )

    def _load_episode(self, episode_id: int) -> PodcastEpisode:
        """
        Load episode from database.

        Args:
            episode_id: Episode ID

        Returns:
            PodcastEpisode instance

        Raises:
            ValueError: If episode not found
        """
        session = get_db_session()
        try:
            episode = session.query(PodcastEpisode).filter(
                PodcastEpisode.id == episode_id
            ).first()

            if not episode:
                raise ValueError(f"Episode {episode_id} not found in database")

            return episode

        finally:
            session.close()

    def _remap_quotes_to_claims(
        self,
        claims_with_quotes: List[ClaimWithQuotes],
        all_quotes: List[Quote],
        unique_quotes: List[Quote]
    ) -> List[ClaimWithQuotes]:
        """
        Remap deduplicated quotes back to claims.

        After global quote deduplication, we need to update each claim's quote list
        to reference the unique quotes instead of the original duplicates.

        Args:
            claims_with_quotes: Claims with original quotes
            all_quotes: Original quotes (flattened from all claims)
            unique_quotes: Deduplicated quotes

        Returns:
            Claims with remapped unique quotes
        """
        quote_mapping = {}

        for i, original in enumerate(all_quotes):
            for unique in unique_quotes:
                if (original.start_position == unique.start_position and
                    original.end_position == unique.end_position):
                    quote_mapping[i] = unique
                    break

        result = []
        quote_index = 0

        for claim in claims_with_quotes:
            remapped_quotes = []

            for _ in claim.quotes:
                if quote_index in quote_mapping:
                    unique_quote = quote_mapping[quote_index]
                    if unique_quote not in remapped_quotes:
                        remapped_quotes.append(unique_quote)
                quote_index += 1

            claim.quotes = remapped_quotes
            result.append(claim)

        return result

    def _create_empty_result(
        self,
        episode_id: int,
        transcript_length: int,
        chunks_count: int,
        processing_time: float
    ) -> PipelineResult:
        """
        Create empty result (no claims or quotes).

        Args:
            episode_id: Episode ID
            transcript_length: Transcript character count
            chunks_count: Number of chunks
            processing_time: Processing time in seconds

        Returns:
            PipelineResult with empty claims list
        """
        stats = PipelineStats(
            episode_id=episode_id,
            transcript_length=transcript_length,
            chunks_count=chunks_count,
            claims_extracted=0,
            claims_after_dedup=0,
            claims_with_quotes=0,
            quotes_before_dedup=0,
            quotes_after_dedup=0,
            total_quotes=0,
            avg_quotes_per_claim=0.0,
            processing_time_seconds=processing_time
        )

        return PipelineResult(
            episode_id=episode_id,
            claims=[],
            stats=stats
        )

    async def process_episodes(
        self,
        episode_ids: List[int]
    ) -> List[PipelineResult]:
        """
        Process multiple episodes.

        Args:
            episode_ids: List of episode IDs to process

        Returns:
            List of pipeline results

        Example:
            ```python
            pipeline = ExtractionPipeline()
            results = await pipeline.process_episodes([123, 456, 789])

            for result in results:
                print(f"Episode {result.episode_id}: {len(result.claims)} claims")
            ```
        """
        logger.info(f"Processing {len(episode_ids)} episodes")

        results: List[PipelineResult] = []

        for i, episode_id in enumerate(episode_ids, 1):
            logger.info(f"Processing episode {i}/{len(episode_ids)}: {episode_id}")

            try:
                result = await self.process_episode(episode_id)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to process episode {episode_id}: {e}",
                    exc_info=True
                )
                continue

        logger.info(
            f"✅ Processed {len(results)}/{len(episode_ids)} episodes successfully"
        )

        return results
