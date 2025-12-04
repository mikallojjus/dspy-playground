"""Validation service for API endpoints."""

import time
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from src.infrastructure.gemini_service import (
    GeminiService,
    ClaimValidationInput,
)
from src.database.claim_repository import ClaimRepository
from src.cli.episode_query import EpisodeQueryService
from src.database.models import PodcastEpisode
from src.api.schemas.validation_schema import (
    EpisodeValidationResult,
    ValidationSummary,
    BatchValidationResponse,
)
from src.api.exceptions import ProcessingError
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ValidationService:
    """
    Service for handling claim validation requests.

    Wraps the CLI validate_claims logic and converts it to async API-friendly methods.
    Follows the same pattern as ExtractionService.
    """

    def __init__(self):
        """Initialize the validation service."""
        self.gemini_service: GeminiService | None = None
        self.validation_prompt: str | None = None

    def _get_gemini_service(self) -> GeminiService:
        """Get or create the Gemini service singleton."""
        if self.gemini_service is None:
            logger.info("Initializing GeminiService...")
            self.gemini_service = GeminiService()
        return self.gemini_service

    def _get_validation_prompt(self) -> str:
        """Get the validation prompt (cached)."""
        if self.validation_prompt is None:
            gemini_service = self._get_gemini_service()
            self.validation_prompt = gemini_service.get_validation_prompt()
            logger.info(f"Loaded validation prompt ({len(self.validation_prompt)} chars)")
        return self.validation_prompt

    async def _validate_single_episode(
        self,
        episode: PodcastEpisode,
        gemini_service: GeminiService,
        claim_repo: ClaimRepository,
        validation_prompt: str,
        dry_run: bool,
    ) -> Tuple[int, int, bool, Optional[str], float]:
        """
        Validate claims for a single episode.

        Args:
            episode: Episode to validate
            gemini_service: Gemini service instance
            claim_repo: Claim repository instance
            validation_prompt: Validation prompt
            dry_run: If True, don't actually flag claims

        Returns:
            Tuple of (claims_checked, claims_flagged, skipped, skip_reason, processing_time)
        """
        start_time = time.time()
        episode_id = int(episode.id)

        # Get unverified claim count
        claim_counts = claim_repo.get_episode_claim_counts([episode_id], only_unverified=True)
        claim_count = claim_counts.get(episode_id, 0)

        # Guardrail: Skip episodes with too few unverified claims
        if claim_count < settings.min_claims_for_validation:
            processing_time = time.time() - start_time
            return (
                0,
                0,
                True,
                f"Only {claim_count} unverified claims (min {settings.min_claims_for_validation})",
                processing_time,
            )

        # Get unverified claims for this episode
        claims_by_episode = claim_repo.get_claims_by_episodes(
            [episode_id], include_flagged=False, include_verified=False
        )
        claims = claims_by_episode.get(episode_id, [])

        if not claims:
            processing_time = time.time() - start_time
            return (0, 0, True, "No unverified, unflagged claims to validate", processing_time)

        logger.info(f"Validating {len(claims)} unverified claims for episode {episode_id}")

        # Convert to validation input
        validation_inputs = [
            ClaimValidationInput(
                claim_id=int(claim.id),
                claim_text=claim.claim_text,
                confidence=claim.confidence,
                episode_id=episode_id,
            )
            for claim in claims
        ]

        # Process in batches
        all_results = []
        batch_size = settings.gemini_validation_batch_size

        for i in range(0, len(validation_inputs), batch_size):
            batch = validation_inputs[i : i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} claims)")

            results = await gemini_service.validate_claims_batch(batch, validation_prompt)
            all_results.extend(results)

        # Filter invalid claims
        invalid_claim_ids = [result.claim_id for result in all_results if not result.is_valid]

        # Flag bad claims
        claims_flagged = 0
        if invalid_claim_ids:
            claims_flagged = claim_repo.flag_claims(invalid_claim_ids, dry_run=dry_run)

        # Mark all validated claims as verified (both good and bad)
        all_claim_ids = [inp.claim_id for inp in validation_inputs]
        claim_repo.mark_claims_verified(all_claim_ids, dry_run=dry_run)

        processing_time = time.time() - start_time
        return (len(claims), claims_flagged, False, None, processing_time)

    async def validate_batch_episodes(
        self,
        podcast_ids: List[int],
        target: int | None = None,
        dry_run: bool = False,
        continue_on_error: bool = False,
        db_session: Session | None = None,
    ) -> BatchValidationResponse:
        """
        Validate claims from multiple episodes based on podcast IDs and target.

        Args:
            podcast_ids: List of podcast IDs to validate
            target: Validate claims from latest N episodes per podcast
            dry_run: If True, don't actually flag claims
            continue_on_error: Continue processing if an episode fails
            db_session: Optional database session

        Returns:
            BatchValidationResponse with validation results and statistics

        Raises:
            ProcessingError: If validation fails (only when continue_on_error=False)
        """
        logger.info(
            f"Processing validation batch: podcasts={podcast_ids}, target={target}, "
            f"dry_run={dry_run}, continue_on_error={continue_on_error}"
        )

        # Initialize services
        gemini_service = self._get_gemini_service()
        validation_prompt = self._get_validation_prompt()

        # Query episodes to validate
        query_service = EpisodeQueryService(db_session)
        episodes_to_validate = query_service.get_episodes_to_validate(
            podcast_ids=podcast_ids, target=target or 0
        )

        if not episodes_to_validate:
            logger.info(f"No episodes with unverified claims found for podcasts {podcast_ids}")
            # Return successful empty response
            summary = ValidationSummary(
                total_episodes=0,
                successful_episodes=0,
                failed_episodes=0,
                skipped_episodes=0,
                total_claims_checked=0,
                total_claims_flagged=0,
                total_processing_time_seconds=0.0,
            )
            return BatchValidationResponse(results=[], summary=summary, errors={})

        logger.info(f"Found {len(episodes_to_validate)} episodes to validate")

        # Initialize claim repository
        from src.database.connection import get_db_session
        session = db_session or get_db_session()
        claim_repo = ClaimRepository(session)

        # Process each episode
        results: List[EpisodeValidationResult] = []
        errors: dict[int, str] = {}
        skipped_count = 0
        total_claims_checked = 0
        total_claims_flagged = 0
        total_time = 0.0

        for episode in episodes_to_validate:
            try:
                (
                    claims_checked,
                    claims_flagged,
                    skipped,
                    skip_reason,
                    processing_time,
                ) = await self._validate_single_episode(
                    episode, gemini_service, claim_repo, validation_prompt, dry_run
                )

                total_time += processing_time

                if skipped:
                    skipped_count += 1
                    logger.info(
                        f"Episode {episode.id} skipped: {skip_reason}"
                    )
                else:
                    result = EpisodeValidationResult(
                        episode_id=episode.id,
                        claims_checked=claims_checked,
                        claims_flagged=claims_flagged,
                        processing_time_seconds=processing_time,
                    )
                    results.append(result)
                    total_claims_checked += claims_checked
                    total_claims_flagged += claims_flagged

                    logger.info(
                        f"Episode {episode.id}: {claims_checked} checked, "
                        f"{claims_flagged} flagged in {processing_time:.1f}s"
                    )

                # Commit after each episode
                if not dry_run:
                    session.commit()

            except Exception as e:
                error_msg = str(e)
                errors[episode.id] = error_msg
                logger.error(f"Episode {episode.id} failed: {error_msg}")

                # Rollback on error
                session.rollback()

                if not continue_on_error:
                    raise ProcessingError(f"Episode {episode.id} failed: {error_msg}")

        # Close session if we created it
        if db_session is None:
            session.close()

        # Create summary
        summary = ValidationSummary(
            total_episodes=len(episodes_to_validate),
            successful_episodes=len(results),
            failed_episodes=len(errors),
            skipped_episodes=skipped_count,
            total_claims_checked=total_claims_checked,
            total_claims_flagged=total_claims_flagged,
            total_processing_time_seconds=total_time,
        )

        return BatchValidationResponse(results=results, summary=summary, errors=errors)
