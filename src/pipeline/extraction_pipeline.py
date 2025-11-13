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

from dataclasses import dataclass, field
from typing import List, Dict, Optional, cast
import time

from src.config.settings import settings
from src.database.connection import get_db_session
from src.database.models import PodcastEpisode
from src.database.claim_repository import ClaimRepository
from src.database.chunk_repository import ChunkRepository
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
from src.dspy_models.entailment_validator import EntailmentValidatorModel
from src.dspy_models.ad_classifier import AdClassifierModel
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FilteredItem:
    """Represents a filtered item with its reason."""
    text: str
    reason: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FilteredItemsLog:
    """Log of filtered items with samples and reasons."""
    specificity_filtered: list[FilteredItem] = field(default_factory=list)
    ad_claims: list[FilteredItem] = field(default_factory=list)
    low_confidence_claims: list[FilteredItem] = field(default_factory=list)
    claims_without_quotes: list[FilteredItem] = field(default_factory=list)
    database_duplicates: list[FilteredItem] = field(default_factory=list)
    question_quotes: list[FilteredItem] = field(default_factory=list)
    quality_filtered_quotes: list[FilteredItem] = field(default_factory=list)
    relevance_filtered_quotes: list[FilteredItem] = field(default_factory=list)
    duplicate_quotes: list[FilteredItem] = field(default_factory=list)
    entailment_filtered_quotes: list[FilteredItem] = field(default_factory=list)


@dataclass
class PipelineStats:
    """
    Statistics from pipeline execution.

    Attributes:
        episode_id: Episode ID processed
        transcript_length: Original transcript character count
        chunks_count: Number of chunks created
        claims_extracted: Number of claims extracted (before filtering)
        claims_after_specificity_filter: Number of claims after specificity filtering
        specificity_filtered_count: Number of claims filtered by specificity
        claims_after_ad_filter: Number of claims after ad filtering (if enabled)
        ad_claims_filtered: Number of advertisement claims filtered out
        claims_after_dedup: Number of claims after deduplication
        claims_after_confidence_filter: Number of claims after confidence filtering
        low_confidence_filtered_count: Number of claims filtered by low confidence
        claims_with_quotes: Number of claims that have at least one quote
        claims_without_quotes_count: Number of claims that have no supporting quotes
        quotes_initial_candidates: Number of initial quote candidates found
        quotes_after_question_filter: Number of quotes after question filtering
        question_filtered_count: Number of rhetorical questions filtered
        quotes_after_quality_filter: Number of quotes after quality filtering
        quality_filtered_count: Number of quotes filtered by quality
        quotes_after_relevance_filter: Number of quotes after relevance filtering
        relevance_filtered_count: Number of quotes filtered by relevance
        quotes_before_dedup: Number of quotes before deduplication
        quotes_after_dedup: Number of quotes after deduplication
        duplicate_quotes_count: Number of duplicate quotes removed
        quotes_before_entailment: Number of quotes before entailment filtering
        quotes_after_entailment: Number of quotes after entailment filtering
        entailment_filtered_quotes: Number of quotes filtered by entailment
        database_duplicates_found: Number of duplicate claims found in database
        claims_merged_to_existing: Number of claims merged to existing episodes
        claims_saved: Number of new claims saved to database
        quotes_saved: Number of unique quotes saved to database
        total_quotes: Total quotes in final result
        avg_quotes_per_claim: Average quotes per claim
        processing_time_seconds: Total processing time
        stage_timings: Dict[str, float] = field(default_factory=dict)
        filtered_items: FilteredItemsLog = field(default_factory=FilteredItemsLog)
    """
    episode_id: int
    transcript_length: int
    chunks_count: int
    claims_extracted: int
    claims_after_specificity_filter: int = 0
    specificity_filtered_count: int = 0
    claims_after_ad_filter: int = 0
    ad_claims_filtered: int = 0
    claims_after_dedup: int = 0
    claims_after_confidence_filter: int = 0
    low_confidence_filtered_count: int = 0
    claims_with_quotes: int = 0
    claims_without_quotes_count: int = 0
    quotes_initial_candidates: int = 0
    quotes_after_question_filter: int = 0
    question_filtered_count: int = 0
    quotes_after_quality_filter: int = 0
    quality_filtered_count: int = 0
    quotes_after_relevance_filter: int = 0
    relevance_filtered_count: int = 0
    quotes_before_dedup: int = 0
    quotes_after_dedup: int = 0
    duplicate_quotes_count: int = 0
    quotes_before_entailment: int = 0
    quotes_after_entailment: int = 0
    entailment_filtered_quotes: int = 0
    database_duplicates_found: int = 0
    claims_merged_to_existing: int = 0
    claims_saved: int = 0
    quotes_saved: int = 0
    total_quotes: int = 0
    avg_quotes_per_claim: float = 0.0
    processing_time_seconds: float = 0.0
    stage_timings: dict[str, float] = field(default_factory=dict)
    filtered_items: FilteredItemsLog = field(default_factory=FilteredItemsLog)


@dataclass
class PipelineResult:
    """
    Result from pipeline execution.

    Attributes:
        episode_id: Episode ID processed
        claims: Claims with supporting quotes
        stats: Pipeline statistics
        saved_claim_ids: IDs of claims saved to database
        duplicate_details: Information about duplicate claims found
    """
    episode_id: int
    claims: List[ClaimWithQuotes]
    stats: PipelineStats
    saved_claim_ids: List[int] = field(default_factory=list)
    duplicate_details: List[Dict] = field(default_factory=list)


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
        self.entailment_validator = EntailmentValidatorModel()

        # Initialize ad classifier (only if enabled and model exists)
        self.ad_classifier = None
        if settings.filter_advertisement_claims:
            try:
                self.ad_classifier = AdClassifierModel()
                logger.info(
                    f"Ad classifier enabled (threshold: {settings.ad_classification_threshold})"
                )
            except FileNotFoundError:
                logger.warning(
                    "Ad filtering enabled but model not found. "
                    "Run src/training/train_ad_classifier.py to create model. "
                    "Proceeding without ad filtering."
                )

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
        stage_timings: Dict[str, float] = {}
        filtered_items_log = FilteredItemsLog()

        logger.info(f"Starting pipeline for episode {episode_id}")

        if self.reranker.enabled:
            await self.reranker.wait_for_ready()

        episode = self._load_episode(episode_id)

        # SQLAlchemy's typing shows Column[str], but at runtime it's actually str | None
        # Use cast to tell the type checker what the actual runtime type is
        transcript = cast(Optional[str], episode.podscribe_transcript)
        if not transcript:
            raise ValueError(f"Episode {episode_id} has no transcript")

        transcript_length = len(transcript)

        logger.info(
            f"Loaded episode {episode_id}: '{episode.name}' "
            f"({transcript_length} chars)"
        )

        logger.info("Step 1/9: Parsing transcript...")
        parsed_transcript = self.parser.parse(transcript)
        logger.info(f"  ✓ Parsed {len(parsed_transcript.segments)} segments")

        logger.info("Step 2/13: Chunking transcript...")
        chunks = self.chunker.chunk_text(parsed_transcript.full_text)
        logger.info(f"  ✓ Created {len(chunks)} chunks")

        # NOTE: Database session is opened AFTER extraction completes (see line ~673)
        # This prevents holding DB connections during slow LLM processing (~5-6 minutes)
        duplicate_details = []
        unique_claims_for_db = []

        try:
            # Step 3: Defer chunk save until after extraction completes
            # (Prevents stale chunks if pipeline fails during extraction)
            logger.info("Step 3/13: Skipping early chunk save (will save atomically with claims)...")

            logger.info("Step 4/13: Extracting claims from chunks...")
            stage_start = time.time()
            claims = await self.claim_extractor.extract_from_chunks(chunks)
            stage_timings["claim_extraction"] = time.time() - stage_start

            # Collect filtering stats from claim_extractor
            claims_extracted_count = self.claim_extractor.claims_before_filter_count
            claims_after_specificity = self.claim_extractor.claims_after_filter_count
            specificity_filtered_count = claims_extracted_count - claims_after_specificity

            # Collect filtered items
            for text, reason in self.claim_extractor.specificity_filtered_items[:5]:  # Limit to 5 samples
                filtered_items_log.specificity_filtered.append(FilteredItem(text=text, reason=reason))

            logger.info(f"  ✓ Extracted {claims_extracted_count} claims → {claims_after_specificity} after specificity filter")

            # NOTE: Keep ephemeral chunk IDs for now, will map to DB IDs after chunks are saved

            if not claims:
                logger.warning("No claims extracted, ending pipeline")
                return self._create_empty_result(
                    episode_id,
                    transcript_length,
                    len(chunks),
                    time.time() - start_time
                )

            # Step 4: Filter advertisement claims (if enabled)
            claims_after_ad_filter = claims_after_specificity
            ad_claims_filtered = 0

            if self.ad_classifier is not None:
                logger.info("Step 4/13: Filtering advertisement claims...")
                stage_start = time.time()
                claim_texts = [c.claim_text for c in claims]

                # Classify all claims in parallel
                classification_results = await self.ad_classifier.classify_batch_parallel(
                    claim_texts,
                    max_concurrency=settings.max_ad_classification_concurrency
                )

                # Filter out ads above threshold
                content_claims = []
                for claim, result in zip(claims, classification_results):
                    is_ad = result["is_advertisement"]
                    confidence = result["confidence"]

                    if is_ad and confidence >= settings.ad_classification_threshold:
                        ad_claims_filtered += 1
                        # Track filtered ad (limit to 5 samples)
                        if len(filtered_items_log.ad_claims) < 5:
                            filtered_items_log.ad_claims.append(FilteredItem(
                                text=claim.claim_text,
                                reason=f"Advertisement (confidence={confidence:.2f})",
                                metadata={"confidence": confidence}
                            ))
                        logger.debug(
                            f"Filtered ad claim (confidence={confidence:.2f}): "
                            f"{claim.claim_text[:60]}..."
                        )
                    else:
                        content_claims.append(claim)

                claims = content_claims
                claims_after_ad_filter = len(claims)
                stage_timings["ad_classification"] = time.time() - stage_start

                logger.info(
                    f"  ✓ {len(claim_texts)} → {claims_after_ad_filter} claims "
                    f"({ad_claims_filtered} advertisements filtered)"
                )

                if not claims:
                    logger.warning("No claims remaining after ad filtering, ending pipeline")
                    return self._create_empty_result(
                        episode_id,
                        transcript_length,
                        len(chunks),
                        time.time() - start_time
                    )
            else:
                logger.info("Step 4/13: Skipping ad filtering (disabled or model not found)")

            logger.info("Step 5/13: Building search index...")
            stage_start = time.time()
            search_index = await TranscriptSearchIndex.build_from_transcript(
                parsed_transcript,
                self.embedder
            )
            stage_timings["search_index_build"] = time.time() - stage_start
            logger.info(f"  ✓ Indexed {search_index.get_segment_count()} segments")

            logger.info("Step 5/13: Finding quotes for claims...")
            stage_start = time.time()
            quote_finder = QuoteFinder(search_index, self.reranker)
            claims_with_quotes = await quote_finder.find_quotes_for_claims(claims)
            stage_timings["quote_finding"] = time.time() - stage_start

            # Collect quote filtering stats from quote_finder
            quotes_initial_candidates = quote_finder.quotes_initial_candidates_count
            quotes_after_question = quote_finder.quotes_after_question_filter_count
            quotes_after_quality = quote_finder.quotes_after_quality_filter_count
            quotes_after_relevance = quote_finder.quotes_after_relevance_filter_count

            # Collect filtered items (limit to 3 samples each)
            for text, reason in quote_finder.question_filtered_items[:3]:
                filtered_items_log.question_quotes.append(FilteredItem(text=text, reason=reason))
            for text, reason in quote_finder.quality_filtered_items[:3]:
                filtered_items_log.quality_filtered_quotes.append(FilteredItem(text=text, reason=reason))
            for text, reason in quote_finder.relevance_filtered_items[:3]:
                filtered_items_log.relevance_filtered_quotes.append(FilteredItem(text=text, reason=reason))

            logger.info(f"  ✓ Found quotes for {len(claims_with_quotes)} claims")

            # Step 6: Deduplicate quotes BEFORE validation (NEW - Sprint 4)
            # This ensures we validate the FINAL merged quote texts, not intermediate versions
            logger.info("Step 6/13: Deduplicating quotes...")
            stage_start = time.time()
            quotes_before_dedup = sum(len(c.quotes) for c in claims_with_quotes)
            all_quotes = [q for c in claims_with_quotes for q in c.quotes]
            unique_quotes = self.quote_deduplicator.deduplicate(all_quotes)
            stage_timings["quote_deduplication"] = time.time() - stage_start

            duplicate_quotes_count = len(all_quotes) - len(unique_quotes)

            logger.info(
                f"  ✓ {len(all_quotes)} → {len(unique_quotes)} unique quotes "
                f"({duplicate_quotes_count} duplicates removed)"
            )

            claims_with_quotes = self._remap_quotes_to_claims(
                claims_with_quotes, all_quotes, unique_quotes
            )

            # Step 7: Entailment validation on FINAL deduplicated quotes (NEW - Sprint 4)
            # We validate after deduplication to ensure the actual text being saved
            # to the database has been validated (not an intermediate text that gets replaced)
            logger.info("Step 7/13: Validating entailment (filtering non-SUPPORTS quotes)...")
            stage_start = time.time()
            quotes_before_entailment = sum(len(c.quotes) for c in claims_with_quotes)

            # Collect ALL claim-quote pairs for global parallel validation (optimized)
            # This validates all pairs at once instead of sequentially per claim
            all_pairs = []
            pair_to_claim_quote_map = []  # Track which claim/quote each pair belongs to

            for claim_idx, claim in enumerate(claims_with_quotes):
                if not claim.quotes:
                    continue

                for quote_idx, quote in enumerate(claim.quotes):
                    all_pairs.append((claim.claim_text, quote.quote_text))
                    pair_to_claim_quote_map.append((claim_idx, quote_idx))

            # Validate ALL pairs in parallel (max_concurrency from settings)
            if all_pairs:
                logger.info(f"Validating {len(all_pairs)} claim-quote pairs globally in parallel...")
                validation_results = await self.entailment_validator.validate_batch_parallel(all_pairs)

                # Map results back to Quote objects
                for (claim_idx, quote_idx), result in zip(pair_to_claim_quote_map, validation_results):
                    quote = claims_with_quotes[claim_idx].quotes[quote_idx]
                    quote.entailment_score = result.get('confidence', 0.0)
                    quote.entailment_relationship = result.get('relationship', 'UNKNOWN')

            # Filter to keep only SUPPORTS quotes
            for claim in claims_with_quotes:
                for quote in claim.quotes:
                    if quote.entailment_relationship != 'SUPPORTS':
                        # Track filtered entailment quote (limit to 3 samples)
                        if len(filtered_items_log.entailment_filtered_quotes) < 3:
                            filtered_items_log.entailment_filtered_quotes.append(FilteredItem(
                                text=quote.quote_text,
                                reason=f"Relationship: {quote.entailment_relationship} (score={quote.entailment_score:.2f})",
                                metadata={"relationship": quote.entailment_relationship, "score": quote.entailment_score}
                            ))

                claim.quotes = [
                    q for q in claim.quotes
                    if q.entailment_relationship == 'SUPPORTS'
                ]

            quotes_after_entailment = sum(len(c.quotes) for c in claims_with_quotes)
            entailment_filtered = quotes_before_entailment - quotes_after_entailment
            stage_timings["entailment_validation"] = time.time() - stage_start

            logger.info(
                f"  ✓ {quotes_before_entailment} → {quotes_after_entailment} quotes "
                f"({entailment_filtered} non-SUPPORTS filtered)"
            )

            logger.info("Step 8/13: Deduplicating claims (batch level)...")
            stage_start = time.time()
            deduplicated_claims = await self.claim_deduplicator.deduplicate_batch(
                claims_with_quotes
            )
            stage_timings["claim_deduplication"] = time.time() - stage_start
            logger.info(
                f"  ✓ {len(claims_with_quotes)} → {len(deduplicated_claims)} unique claims "
                f"({len(claims_with_quotes) - len(deduplicated_claims)} duplicates removed)"
            )

            # Step 9: Filter claims without quotes FIRST (before confidence calculation)
            # This prevents claims with 0 quotes from showing as "low confidence (0.0)"
            logger.info("Step 9/13: Filtering claims without quotes...")
            claims_with_quotes_list = []
            claims_without_quotes_count = 0
            for c in deduplicated_claims:
                if not c.quotes:
                    claims_without_quotes_count += 1
                    # Track claim without quotes (limit to 3 samples)
                    if len(filtered_items_log.claims_without_quotes) < 3:
                        filtered_items_log.claims_without_quotes.append(FilteredItem(
                            text=c.claim_text,
                            reason="No supporting quotes found after entailment validation",
                            metadata={"quote_count": 0}
                        ))
                else:
                    claims_with_quotes_list.append(c)

            logger.info(
                f"  ✓ {len(claims_with_quotes_list)} claims with quotes "
                f"({claims_without_quotes_count} filtered - no quotes)"
            )

            # Step 10: Calculate confidence ONLY for claims WITH quotes
            logger.info("Step 10/13: Calculating confidence scores...")
            stage_start = time.time()
            for claim in claims_with_quotes_list:
                conf = self.confidence_calculator.calculate(claim.quotes)
                claim.confidence = conf.final_confidence
                claim.confidence_components = conf
            stage_timings["confidence_calculation"] = time.time() - stage_start
            logger.info(f"  ✓ Calculated confidence for {len(claims_with_quotes_list)} claims")

            # Step 11: Filter by confidence threshold
            logger.info("Step 11/13: Filtering low-confidence claims...")
            claims_before_confidence_filter = len(claims_with_quotes_list)
            high_confidence_claims = []
            for c in claims_with_quotes_list:
                if c.confidence < settings.min_confidence:
                    # Track low confidence claim (limit to 3 samples)
                    if len(filtered_items_log.low_confidence_claims) < 3:
                        filtered_items_log.low_confidence_claims.append(FilteredItem(
                            text=c.claim_text,
                            reason=f"Low confidence ({c.confidence:.2f} < {settings.min_confidence})",
                            metadata={"confidence": c.confidence, "quote_count": len(c.quotes)}
                        ))
                else:
                    high_confidence_claims.append(c)

            low_confidence_filtered_count = claims_before_confidence_filter - len(high_confidence_claims)
            claims_after_confidence_filter = len(high_confidence_claims)

            logger.info(
                f"  ✓ {len(high_confidence_claims)} claims above threshold "
                f"({low_confidence_filtered_count} filtered)"
            )

            claims_with_quotes = high_confidence_claims

            if not claims_with_quotes:
                logger.warning("No claims have supporting quotes after filtering")
                return self._create_empty_result(
                    episode_id,
                    transcript_length,
                    len(chunks),
                    time.time() - start_time
                )

            # ========================================================================
            # Step 11: Database deduplication (CURRENTLY DISABLED - COMMENTED OUT)
            # ========================================================================
            # NOTE: Cross-episode deduplication requires holding a DB session open during extraction,
            # which causes connection timeouts on long-running jobs (4+ hours).
            #
            # Previously, the DB session was opened at the start and held for ~6 minutes per episode
            # while doing LLM calls. After 43 episodes, the connection timed out.
            #
            # To re-enable this feature:
            # 1. Set ENABLE_CROSS_EPISODE_DEDUPLICATION=true in .env
            # 2. Uncomment the code block below
            # 3. Refactor to use a separate read-only session for dedup queries (not the save session)
            # 4. Test with batch jobs to ensure no timeouts
            #
            # Commented out code:
            # if settings.enable_cross_episode_deduplication:
            #     logger.info("Step 11/13: Checking database for duplicate claims...")
            #     stage_start = time.time()
            #
            #     for claim in claims_with_quotes:
            #         # Generate embedding for database search
            #         embedding = await self.embedder.embed_text(claim.claim_text)
            #
            #         # Check against database
            #         dedup_result = await self.claim_deduplicator.deduplicate_against_database(
            #             claim.claim_text,
            #             embedding,
            #             episode_id,
            #             db_session
            #         )
            #
            #         if dedup_result.is_duplicate:
            #             # Found duplicate - merge quotes to existing claim
            #             # Type narrowing: existing_claim_id must be set when is_duplicate is True
            #             assert dedup_result.existing_claim_id is not None, "existing_claim_id must be set when is_duplicate is True"
            #
            #             duplicate_details.append({
            #                 "claim_text": claim.claim_text,
            #                 "existing_claim_id": dedup_result.existing_claim_id,
            #                 "reranker_score": dedup_result.reranker_score,
            #                 "new_quotes_count": len(claim.quotes)
            #             })
            #
            #             # Track database duplicate (limit to 3 samples)
            #             if len(filtered_items_log.database_duplicates) < 3:
            #                 filtered_items_log.database_duplicates.append(FilteredItem(
            #                     text=claim.claim_text,
            #                     reason=f"Duplicate of claim {dedup_result.existing_claim_id} (score={dedup_result.reranker_score:.2f})",
            #                     metadata={
            #                         "existing_claim_id": dedup_result.existing_claim_id,
            #                         "reranker_score": dedup_result.reranker_score
            #                     }
            #                 ))
            #
            #             # Merge quotes to existing claim
            #             repo = ClaimRepository(db_session)
            #             await repo.merge_quotes_to_existing_claim(
            #                 dedup_result.existing_claim_id,
            #                 claim.quotes,
            #                 episode_id
            #             )
            #
            #             logger.info(
            #                 f"  Merged {len(claim.quotes)} quotes to existing claim {dedup_result.existing_claim_id}"
            #             )
            #         else:
            #             # Unique claim - prepare for insertion
            #             claim.metadata["embedding"] = embedding
            #             unique_claims_for_db.append(claim)
            #
            #     stage_timings["database_deduplication"] = time.time() - stage_start
            #     logger.info(
            #         f"  ✓ Found {len(duplicate_details)} duplicates, "
            #         f"{len(unique_claims_for_db)} unique claims to save"
            #     )
            # else:

            logger.info("Step 11/13: Cross-episode deduplication disabled (commented out)")

            # Generate embeddings for all claims
            for claim in claims_with_quotes:
                embedding = await self.embedder.embed_text(claim.claim_text)
                claim.metadata["embedding"] = embedding
                unique_claims_for_db.append(claim)

            logger.info(f"  ✓ Prepared {len(unique_claims_for_db)} claims for saving")

            # Track merged count (will be 0 since dedup is disabled)
            claims_merged_to_existing = len(duplicate_details)

            # ========================================================================
            # OPEN DATABASE SESSION NOW - All extraction work is complete
            # ========================================================================
            # We only open the DB connection after all LLM processing is done
            # This prevents holding connections during the slow extraction (~5-6 min)
            logger.info("Step 12/13: Opening database session for atomic save...")
            db_session = get_db_session()

            try:
                # Save chunks to database FIRST
                logger.info("  Saving transcript chunks to database...")
                chunk_repo = ChunkRepository(db_session)
                chunk_db_ids = await chunk_repo.save_chunks(chunks, episode_id)

                # Create mapping from ephemeral chunk_id to database ID
                chunk_id_mapping = {
                    chunk.chunk_id: db_id
                    for chunk, db_id in zip(chunks, chunk_db_ids)
                }
                logger.info(f"  ✓ Saved {len(chunk_db_ids)} chunks to database")

                # Map claim source_chunk_ids from ephemeral to database IDs
                for claim in unique_claims_for_db:
                    claim.source_chunk_id = chunk_id_mapping[claim.source_chunk_id]

                # Save claims and quotes to database
                logger.info("  Saving claims and quotes to database...")
                stage_start = time.time()

                if unique_claims_for_db:
                    repo = ClaimRepository(db_session)

                    # Save claims (without embeddings first)
                    saved_claim_ids = await repo.save_claims(unique_claims_for_db, episode_id)

                    # Update embeddings
                    embeddings_dict = {
                        claim_id: claim.metadata["embedding"]
                        for claim_id, claim in zip(saved_claim_ids, unique_claims_for_db)
                    }
                    await repo.update_claim_embeddings(embeddings_dict)

                    # Count unique quotes saved
                    unique_quote_positions = set()
                    for claim in unique_claims_for_db:
                        for quote in claim.quotes:
                            unique_quote_positions.add((quote.start_position, quote.end_position))
                    quotes_saved = len(unique_quote_positions)

                    logger.info(
                        f"  ✓ Saved {len(saved_claim_ids)} claims, {quotes_saved} unique quotes"
                    )
                else:
                    saved_claim_ids = []
                    quotes_saved = 0
                    logger.info("  ✓ No new claims to save")

                stage_timings["database_save"] = time.time() - stage_start

                # Step 13: Commit atomic transaction (chunks + claims together)
                logger.info("Step 13/13: Committing atomic transaction (chunks + claims)...")
                db_session.commit()
                logger.info("  ✓ Transaction committed successfully")

            except Exception as e:
                logger.error(f"Error saving to database: {e}", exc_info=True)
                db_session.rollback()
                logger.warning("Transaction rolled back")
                raise
            finally:
                db_session.close()

        except Exception as e:
            # Outer exception handler for extraction failures (before DB session opens)
            logger.error(f"Error during extraction: {e}", exc_info=True)
            raise

        # Calculate final stats
        total_quotes = sum(len(c.quotes) for c in claims_with_quotes)
        avg_quotes = total_quotes / len(claims_with_quotes)
        processing_time = time.time() - start_time

        stats = PipelineStats(
            episode_id=episode_id,
            transcript_length=transcript_length,
            chunks_count=len(chunks),
            claims_extracted=claims_extracted_count,
            claims_after_specificity_filter=claims_after_specificity,
            specificity_filtered_count=specificity_filtered_count,
            claims_after_ad_filter=claims_after_ad_filter,
            ad_claims_filtered=ad_claims_filtered,
            claims_after_dedup=len(deduplicated_claims),
            claims_after_confidence_filter=claims_after_confidence_filter,
            low_confidence_filtered_count=low_confidence_filtered_count,
            claims_with_quotes=len(claims_with_quotes),
            claims_without_quotes_count=claims_without_quotes_count,
            quotes_initial_candidates=quotes_initial_candidates,
            quotes_after_question_filter=quotes_after_question,
            question_filtered_count=quotes_initial_candidates - quotes_after_question,
            quotes_after_quality_filter=quotes_after_quality,
            quality_filtered_count=quotes_after_question - quotes_after_quality,
            quotes_after_relevance_filter=quotes_after_relevance,
            relevance_filtered_count=quotes_after_quality - quotes_after_relevance,
            quotes_before_dedup=quotes_before_dedup,
            quotes_after_dedup=len(unique_quotes),
            duplicate_quotes_count=duplicate_quotes_count,
            quotes_before_entailment=quotes_before_entailment,
            quotes_after_entailment=quotes_after_entailment,
            entailment_filtered_quotes=entailment_filtered,
            database_duplicates_found=len(duplicate_details),
            claims_merged_to_existing=claims_merged_to_existing,
            claims_saved=len(saved_claim_ids) if unique_claims_for_db else 0,
            quotes_saved=quotes_saved if unique_claims_for_db else 0,
            total_quotes=total_quotes,
            avg_quotes_per_claim=avg_quotes,
            processing_time_seconds=processing_time,
            stage_timings=stage_timings,
            filtered_items=filtered_items_log
        )

        logger.info(
            f"✅ Pipeline complete for episode {episode_id} "
            f"({processing_time:.1f}s)"
        )
        logger.info(
            f"   📊 {len(claims_with_quotes)} claims, {total_quotes} quotes "
            f"(avg {avg_quotes:.1f} quotes/claim)"
        )

        # Cleanup: Unload all models from GPU to prevent memory accumulation between runs
        self._cleanup_gpu_models()

        return PipelineResult(
            episode_id=episode_id,
            claims=claims_with_quotes,
            stats=stats,
            saved_claim_ids=saved_claim_ids if unique_claims_for_db else [],
            duplicate_details=duplicate_details
        )

    def _cleanup_gpu_models(self):
        """
        Unload all Ollama models from GPU to prevent memory accumulation between runs.

        This prevents the GPU OOM issue where models from previous runs stay resident
        and cause crashes when the next run tries to initialize new model instances.

        Critical models to unload:
        - qwen2.5:7b (inference model, ~7GB VRAM)
        - nomic-embed-text (embedding model, ~200MB VRAM)
        """
        import httpx

        models_to_unload = [
            settings.ollama_model,      # qwen2.5:7b
            "nomic-embed-text"          # embedding model
        ]

        for model_name in models_to_unload:
            try:
                httpx.post(
                    f"{settings.ollama_url}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=5.0
                )
                logger.debug(f"Unloaded {model_name} from GPU")
            except Exception as e:
                logger.debug(f"Failed to unload {model_name}: {e}")

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
            claims_after_ad_filter=0,
            ad_claims_filtered=0,
            claims_after_dedup=0,
            claims_with_quotes=0,
            quotes_before_dedup=0,
            quotes_after_dedup=0,
            quotes_before_entailment=0,
            quotes_after_entailment=0,
            entailment_filtered_quotes=0,
            database_duplicates_found=0,
            claims_saved=0,
            quotes_saved=0,
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
