"""Validation API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.services.validation_service import ValidationService
from src.api.schemas.validation_schema import (
    ClaimValidationRequest,
    BatchValidationResponse,
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/validate", tags=["validation"])


@router.post(
    "/claims",
    response_model=BatchValidationResponse,
    summary="Validate claims from multiple episodes",
    description="""
    Validate claims from multiple episodes using Google Gemini API to identify bad claims.

    This endpoint:
    - Validates claims from episodes with unverified claims
    - Flags invalid claims in the database (sets is_flagged=TRUE)
    - Marks all validated claims as verified (sets is_verified=TRUE)
    - Supports target-based selection (validate latest N episodes per podcast)
    - Supports dry-run mode (identify bad claims without flagging)

    Processing time: ~2-5 seconds per episode (depends on claim count and batch size).
    The API processes episodes sequentially and returns all results when complete.
    """,
    responses={
        200: {
            "description": "Batch validation completed (may include partial failures)",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "episode_id": 123,
                                "claims_checked": 47,
                                "claims_flagged": 3,
                                "processing_time_seconds": 2.3
                            },
                            {
                                "episode_id": 124,
                                "claims_checked": 52,
                                "claims_flagged": 5,
                                "processing_time_seconds": 2.8
                            }
                        ],
                        "summary": {
                            "total_episodes": 2,
                            "successful_episodes": 2,
                            "failed_episodes": 0,
                            "skipped_episodes": 0,
                            "total_claims_checked": 99,
                            "total_claims_flagged": 8,
                            "total_processing_time_seconds": 5.1
                        },
                        "errors": {}
                    }
                }
            },
        },
        404: {"description": "No episodes with unverified claims found"},
        500: {"description": "Processing error (when continue_on_error=false)"},
    },
)
async def validate_claims_batch(
    request: ClaimValidationRequest, db: Session = Depends(get_db)
) -> BatchValidationResponse:
    """
    Validate claims from multiple episodes in batch.

    Args:
        request: Batch validation request with podcast IDs and settings
        db: Database session (injected)

    Returns:
        BatchValidationResponse with validation results and statistics
    """
    logger.info(
        f"API request: batch validate podcasts={request.podcast_ids}, "
        f"target={request.target}, dry_run={request.dry_run}, "
        f"continue_on_error={request.continue_on_error}"
    )

    service = ValidationService()
    result = await service.validate_batch_episodes(
        podcast_ids=request.podcast_ids,
        target=request.target,
        dry_run=request.dry_run,
        continue_on_error=request.continue_on_error,
        db_session=db,
    )

    logger.info(
        f"API response: validation completed - {result.summary.successful_episodes}/"
        f"{result.summary.total_episodes} successful, "
        f"{result.summary.total_claims_flagged} claims flagged"
    )
    return result
