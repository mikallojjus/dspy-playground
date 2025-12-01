"""Extraction service for API endpoints."""

import asyncio
from typing import List
from sqlalchemy.orm import Session

from src.pipeline.extraction_pipeline import ExtractionPipeline, PipelineResult
from src.cli.episode_query import EpisodeQueryService
from src.api.schemas.responses import (
    BatchExtractionSummary,
)
from src.api.exceptions import (
    EpisodeNotFoundError,
    PodcastNotFoundError,
    ProcessingError,
    ProcessingTimeoutError,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ExtractionService:
    """
    Service for handling extraction requests.

    Wraps the ExtractionPipeline and converts domain models to API schemas.
    """

    def __init__(self):
        """Initialize the extraction service."""
        self.pipeline: ExtractionPipeline | None = None

    def _get_pipeline(self) -> ExtractionPipeline:
        """Get or create the extraction pipeline singleton."""
        if self.pipeline is None:
            logger.info("Initializing ExtractionPipeline...")
            self.pipeline = ExtractionPipeline()
        return self.pipeline

    async def _extract_single_episode(
        self, episode_id: int, force: bool = False, db_session: Session | None = None
    ) -> "SimplifiedExtractionResponse":
        """
        Extract claims and quotes from a single episode (internal use only).

        Args:
            episode_id: Episode ID to process
            force: Force reprocessing even if claims exist
            db_session: Optional database session for queries

        Returns:
            SimplifiedExtractionResponse with statistics only

        Raises:
            EpisodeNotFoundError: If episode doesn't exist
            ProcessingError: If extraction fails
            ProcessingTimeoutError: If processing exceeds timeout
        """
        logger.info(f"Processing episode {episode_id} (force={force})")

        # Verify episode exists
        query_service = EpisodeQueryService(db_session)
        episode = query_service.get_episode_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(episode_id)

        # Check if already processed (unless force=True)
        if not force and query_service.is_episode_processed(episode_id):
            logger.info(f"Episode {episode_id} already processed (use force=True to reprocess)")

        # Process with timeout (if configured)
        try:
            pipeline = self._get_pipeline()

            if settings.api_timeout > 0:
                # Process with timeout
                result = await asyncio.wait_for(
                    pipeline.process_episode(episode_id, save_to_db=True),
                    timeout=settings.api_timeout
                )
            else:
                # No timeout - process indefinitely
                result = await pipeline.process_episode(episode_id, save_to_db=True)

        except asyncio.TimeoutError:
            logger.error(f"Processing episode {episode_id} timed out after {settings.api_timeout}s")
            raise ProcessingTimeoutError(settings.api_timeout)
        except Exception as e:
            logger.error(f"Error processing episode {episode_id}: {e}", exc_info=True)
            raise ProcessingError(str(e))

        # Convert to simplified API response
        return self._convert_to_simplified_response(result)

    async def extract_batch_episodes(
        self,
        podcast_ids: List[int],
        target: int | None = None,
        force: bool = False,
        continue_on_error: bool = False,
        db_session: Session | None = None,
    ) -> "SimplifiedBatchExtractionResponse":
        """
        Extract claims from multiple episodes based on podcast IDs and target.

        Args:
            podcast_ids: List of podcast IDs to process
            target: Maintain claims for latest N episodes per podcast
            force: Force reprocessing even if claims exist
            continue_on_error: Continue processing if an episode fails
            db_session: Optional database session for queries

        Returns:
            SimplifiedBatchExtractionResponse with statistics for all episodes

        Raises:
            PodcastNotFoundError: If no episodes found for podcasts
            ProcessingError: If extraction fails (only when continue_on_error=False)
        """
        from src.api.schemas.responses import SimplifiedBatchExtractionResponse

        logger.info(
            f"Processing batch: podcasts={podcast_ids}, target={target}, "
            f"force={force}, continue_on_error={continue_on_error}"
        )

        # Query episodes to process
        query_service = EpisodeQueryService(db_session)
        episodes_to_process = query_service.get_episodes_to_process(
            podcast_ids=podcast_ids,
            target=target or 0,
            force=force
        )

        if not episodes_to_process:
            logger.info(f"No new episodes to process for podcasts {podcast_ids} (all up-to-date)")
            # Return successful empty response (nothing to process is success, not error)
            summary = BatchExtractionSummary(
                total_episodes=0,
                successful_episodes=0,
                failed_episodes=0,
                total_claims_extracted=0,
                total_processing_time_seconds=0.0,
            )
            return SimplifiedBatchExtractionResponse(
                results=[],
                summary=summary,
                errors={},
            )

        logger.info(f"Found {len(episodes_to_process)} episodes to process")

        # Process each episode
        results: List["SimplifiedExtractionResponse"] = []
        errors: dict[int, str] = {}
        total_claims = 0
        total_time = 0.0

        for episode in episodes_to_process:
            try:
                response = await self._extract_single_episode(
                    episode.id, force=force, db_session=db_session
                )
                results.append(response)
                total_claims += response.claims_count
                total_time += response.processing_time_seconds
                logger.info(
                    f"✓ Episode {episode.id}: {response.claims_count} claims "
                    f"in {response.processing_time_seconds:.1f}s"
                )
            except Exception as e:
                error_msg = str(e)
                errors[episode.id] = error_msg
                logger.error(f"✗ Episode {episode.id} failed: {error_msg}")

                if not continue_on_error:
                    raise ProcessingError(f"Episode {episode.id} failed: {error_msg}")

        # Create summary
        summary = BatchExtractionSummary(
            total_episodes=len(episodes_to_process),
            successful_episodes=len(results),
            failed_episodes=len(errors),
            total_claims_extracted=total_claims,
            total_processing_time_seconds=total_time,
        )

        return SimplifiedBatchExtractionResponse(
            results=results,
            summary=summary,
            errors=errors,
        )

    def _convert_to_simplified_response(
        self, result: PipelineResult
    ) -> "SimplifiedExtractionResponse":
        """
        Convert PipelineResult to SimplifiedExtractionResponse.

        Args:
            result: Pipeline result from extraction

        Returns:
            Simplified API response with statistics only
        """
        from src.api.schemas.responses import SimplifiedExtractionResponse

        claims_count = len(result.claims)
        quotes_count = sum(len(claim.quotes) for claim in result.claims)

        return SimplifiedExtractionResponse(
            episode_id=result.episode_id,
            processing_time_seconds=result.stats.processing_time_seconds,
            claims_count=claims_count,
            quotes_count=quotes_count,
        )
