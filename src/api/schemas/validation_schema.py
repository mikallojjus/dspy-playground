"""Schemas for claim validation API endpoints."""

from pydantic import BaseModel, Field


class ClaimValidationRequest(BaseModel):
    """Request body for batch claim validation."""

    podcast_ids: list[int] = Field(
        ...,
        description="List of podcast IDs to validate claims from",
        min_length=1
    )
    target: int | None = Field(
        None,
        description="Validate claims from latest N episodes per podcast. If None, validate all episodes with unverified claims.",
        ge=1
    )
    dry_run: bool = Field(
        default=False,
        description="If true, identify bad claims without actually flagging them in the database"
    )
    continue_on_error: bool = Field(
        default=False,
        description="Continue processing remaining episodes if one fails"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "podcast_ids": [854, 907, 994],
                "target": 20,
                "dry_run": False,
                "continue_on_error": True
            }
        }


class EpisodeValidationResult(BaseModel):
    """Validation result for a single episode."""

    episode_id: int = Field(..., description="Episode ID that was validated")
    claims_checked: int = Field(..., description="Number of claims checked")
    claims_flagged: int = Field(..., description="Number of claims flagged as bad")
    processing_time_seconds: float = Field(..., description="Time taken to validate this episode")

    class Config:
        json_schema_extra = {
            "example": {
                "episode_id": 123,
                "claims_checked": 47,
                "claims_flagged": 3,
                "processing_time_seconds": 2.3
            }
        }


class ValidationSummary(BaseModel):
    """Summary statistics for batch validation."""

    total_episodes: int = Field(..., description="Total episodes to validate")
    successful_episodes: int = Field(..., description="Successfully validated episodes")
    failed_episodes: int = Field(..., description="Failed episodes")
    skipped_episodes: int = Field(..., description="Episodes skipped (e.g., too few claims)")
    total_claims_checked: int = Field(..., description="Total claims checked across all episodes")
    total_claims_flagged: int = Field(..., description="Total claims flagged as bad")
    total_processing_time_seconds: float = Field(..., description="Total processing time")


class BatchValidationResponse(BaseModel):
    """Response for batch validation endpoint."""

    results: list[EpisodeValidationResult] = Field(
        default_factory=list,
        description="Validation results for each episode processed"
    )
    summary: ValidationSummary = Field(..., description="Summary statistics")
    errors: dict[int, str] = Field(
        default_factory=dict,
        description="Map of episode_id to error message for failed episodes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "episode_id": 123,
                        "claims_checked": 47,
                        "claims_flagged": 3,
                        "processing_time_seconds": 2.3
                    }
                ],
                "summary": {
                    "total_episodes": 1,
                    "successful_episodes": 1,
                    "failed_episodes": 0,
                    "skipped_episodes": 0,
                    "total_claims_checked": 47,
                    "total_claims_flagged": 3,
                    "total_processing_time_seconds": 2.3
                },
                "errors": {}
            }
        }
