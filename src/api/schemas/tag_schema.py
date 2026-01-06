"""Schemas for tag query endpoints."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TagQueryRequest(BaseModel):
    """Request body for querying tags by created_at window."""

    start_datetime: datetime = Field(
        ...,
        description="Start of the created_at window (inclusive)",
    )
    end_datetime: datetime = Field(
        ...,
        description="End of the created_at window (inclusive)",
    )

    @field_validator("end_datetime")
    @classmethod
    def validate_range(cls, end_datetime: datetime, info) -> datetime:
        """Ensure end_datetime is not before start_datetime."""
        start = info.data.get("start_datetime")
        if start and end_datetime < start:
            raise ValueError("end_datetime must be greater than or equal to start_datetime")
        return end_datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_datetime": "2024-01-01T00:00:00Z",
                "end_datetime": "2024-01-31T23:59:59Z",
            }
        }
    )


class TagMergeDirectiveResponse(BaseModel):
    """Directive to merge one tag id into another."""

    source_tag_id: int = Field(..., description="Tag id that should be merged and removed")
    target_tag_id: int = Field(..., description="Canonical tag id to merge into")


class TagUpdateDirectiveResponse(BaseModel):
    """Directive to rename/update a tag."""

    id: int = Field(..., description="Tag id that should be updated")
    old_tag: str = Field(..., description="Original tag label")
    new_tag: str = Field(..., description="Suggested replacement label")


class TagQueryResponse(BaseModel):
    """Response containing merge directives."""

    merges: list[TagMergeDirectiveResponse] = Field(
        default_factory=list,
        description="Simplified merge directives (source -> target)",
    )
    updates: list[TagUpdateDirectiveResponse] = Field(
        default_factory=list,
        description="Tag rename suggestions from the tag checker",
    )
