"""Tag merge API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas.tag_schema import TagMergeDirectiveResponse, TagQueryRequest, TagQueryResponse
from src.api.services.tag_service import TagService
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/tags", tags=["tags"])


@router.post(
    "/merge",
    response_model=TagQueryResponse,
    summary="Find and merge synonymous tags",
    description=(
        "Returns merge directives (source -> target) for synonymous tags created between "
        "start_datetime and end_datetime (inclusive)."
    ),
)
def merge_tags(
    request: TagQueryRequest, db: Session = Depends(get_db)
) -> TagQueryResponse:
    """Get tags filtered by created_at window alongside the full tag list."""
    logger.info(
        "Tag merge request",
        extra={
            "start_datetime": request.start_datetime,
            "end_datetime": request.end_datetime,
        },
    )

    service = TagService(db)
    merges = service.fetch_tag_merge_snapshot(
        start_datetime=request.start_datetime,
        end_datetime=request.end_datetime,
    )

    return TagQueryResponse(merges=[TagMergeDirectiveResponse(**merge) for merge in merges])
