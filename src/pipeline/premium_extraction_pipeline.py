"""
Premium extraction pipeline using Gemini 3 Pro for full-transcript processing.

Differences from standard pipeline:
1. NO DSPy (direct Gemini API calls)
2. NO chunking (full transcript processing with 1M context)
3. NO deduplication (simplified)
4. NO ad filtering (simplified)
5. NO quote processing (focus on speed)
6. Single Gemini call instead of parallel chunk processing
7. Much faster (~30-60s vs 5-6 minutes)

Usage:
    from src.pipeline.premium_extraction_pipeline import PremiumExtractionPipeline

    pipeline = PremiumExtractionPipeline()
    result = await pipeline.process_episode(episode_id=123)

    print(f"Extracted {len(result.claims)} claims in {result.processing_time_seconds:.1f}s")
"""

from dataclasses import dataclass
from typing import cast, Optional
import time

from src.config.settings import settings
from src.database.connection import get_db_session
from src.database.models import PodcastEpisode
from src.database.claim_repository import ClaimRepository
from src.preprocessing.transcript_parser import TranscriptParser
from src.extraction.premium_claim_extractor import PremiumClaimExtractor
from src.extraction.quote_finder import ClaimWithQuotes
from src.infrastructure.embedding_service import EmbeddingService
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PremiumPipelineResult:
    """Result from premium pipeline execution."""
    episode_id: int
    claims: list[ClaimWithQuotes]
    processing_time_seconds: float
    claims_extracted: int
    model_used: str  # "gemini-3-pro-preview"


class PremiumExtractionPipeline:
    """Premium pipeline using Gemini 3 Pro for full-transcript extraction."""

    def __init__(self):
        """Initialize premium pipeline components."""
        logger.info("Initializing PremiumExtractionPipeline")

        self.parser = TranscriptParser()
        self.premium_extractor = PremiumClaimExtractor()
        self.embedder = EmbeddingService()

        logger.info("Premium pipeline ready (no chunking, no dedup, no quotes)")

    async def process_episode(
        self,
        episode_id: int,
        save_to_db: bool = True
    ) -> PremiumPipelineResult:
        """
        Process episode through premium pipeline.

        Key differences from standard pipeline:
        - No chunking (processes full transcript)
        - Single Gemini API call
        - Faster processing (~30-60 seconds vs 5-6 minutes)
        - No quote processing
        - No deduplication

        Args:
            episode_id: Episode ID to process
            save_to_db: Whether to save results to database (default True)

        Returns:
            PremiumPipelineResult with claims and stats

        Raises:
            ValueError: If episode not found or has no transcript
            Exception: If pipeline fails
        """
        start_time = time.time()

        logger.info(f"Starting PREMIUM pipeline for episode {episode_id}")

        # Step 1: Load episode
        episode = self._load_episode(episode_id)
        transcript, transcript_format = self._select_transcript(episode)
        transcript_length = len(transcript)

        logger.info(
            f"Loaded episode {episode_id}: '{episode.name}' "
            f"({transcript_length} chars, {transcript_format} format)"
        )

        # Step 2: Parse transcript
        logger.info("Step 1/3: Parsing transcript...")
        parsed_transcript = self.parser.parse(transcript, format=transcript_format)
        logger.info(f"  ✓ Parsed {len(parsed_transcript.segments)} segments")

        # Step 3: Extract claims from FULL transcript (no chunking!)
        logger.info("Step 2/3: Extracting claims from full transcript (Gemini 3 Pro)...")
        stage_start = time.time()
        claim_texts = await self.premium_extractor.extract_claims_from_transcript(
            parsed_transcript.full_text
        )
        extraction_time = time.time() - stage_start
        claims_extracted = len(claim_texts)
        logger.info(f"  ✓ Extracted {claims_extracted} claims in {extraction_time:.1f}s")

        # Convert to ClaimWithQuotes objects
        claims = [
            ClaimWithQuotes(
                claim_text=text,
                source_chunk_id=None,  # No chunks in premium pipeline
                quotes=[],
                confidence=0.8  # Default confidence (no quotes)
            )
            for text in claim_texts
        ]

        if not claims:
            logger.warning("No claims extracted, ending pipeline")
            processing_time = time.time() - start_time
            return PremiumPipelineResult(
                episode_id=episode_id,
                claims=[],
                processing_time_seconds=processing_time,
                claims_extracted=0,
                model_used=settings.gemini_premium_model
            )

        # Step 4: Save to database
        if save_to_db:
            logger.info("Step 3/3: Saving claims to database...")
            db_session = get_db_session()

            try:
                # Generate embeddings for all claims
                logger.info("  Generating embeddings for claims...")
                for claim in claims:
                    embedding = await self.embedder.embed_text(claim.claim_text)
                    claim.metadata["embedding"] = embedding

                # Save claims to database
                logger.info("  Saving claims to database...")
                repo = ClaimRepository(db_session)
                saved_claim_ids = await repo.save_claims(claims, episode_id)

                # Update embeddings
                embeddings_dict = {
                    claim_id: claim.metadata["embedding"]
                    for claim_id, claim in zip(saved_claim_ids, claims)
                }
                await repo.update_claim_embeddings(embeddings_dict)

                logger.info(f"  ✓ Saved {len(saved_claim_ids)} claims to database")

                # Commit transaction
                db_session.commit()
                logger.info("  ✓ Transaction committed successfully")

            except Exception as e:
                logger.error(f"Error saving to database: {e}", exc_info=True)
                db_session.rollback()
                logger.warning("Transaction rolled back")
                raise
            finally:
                db_session.close()
        else:
            logger.info("Step 3/3: Skipping database save (API mode)")

        processing_time = time.time() - start_time

        logger.info(
            f"✅ PREMIUM pipeline complete for episode {episode_id} "
            f"({processing_time:.1f}s, {len(claims)} claims)"
        )

        return PremiumPipelineResult(
            episode_id=episode_id,
            claims=claims,
            processing_time_seconds=processing_time,
            claims_extracted=claims_extracted,
            model_used=settings.gemini_premium_model
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

    def _select_transcript(self, episode: PodcastEpisode) -> tuple[str, str]:
        """
        Select transcript from episode with priority-based format detection.

        Priority order:
        1. Podscribe transcript (if available)
        2. Bankless transcript (if available)
        3. Assembly transcript (if available)
        4. Raise error if none available

        Args:
            episode: Episode to get transcript from

        Returns:
            Tuple of (transcript_text, format_name)
            format_name is "podscribe", "bankless", or "assembly"

        Raises:
            ValueError: If episode has no transcript in any format
        """
        # Priority 1: Podscribe
        podscribe_transcript = cast(Optional[str], episode.podscribe_transcript)
        if podscribe_transcript:
            return (podscribe_transcript, "podscribe")

        # Priority 2: Bankless
        bankless_transcript = cast(Optional[str], episode.bankless_transcript)
        if bankless_transcript:
            return (bankless_transcript, "bankless")

        # Priority 3: Assembly
        assembly_transcript = cast(Optional[str], episode.assembly_transcript)
        if assembly_transcript:
            return (assembly_transcript, "assembly")

        # No transcript available
        raise ValueError(
            f"Episode {episode.id} has no transcript "
            f"(checked podscribe_transcript, bankless_transcript, and assembly_transcript)"
        )
