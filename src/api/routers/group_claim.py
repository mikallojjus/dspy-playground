from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import json

from src.api.schemas.group_claim_schema import GroupClaimRequest, GroupClaimResponse
from src.api.services.group_claim_service import assign_claim_tag
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
  "/group-claim",
  response_model=GroupClaimResponse,
  status_code=status.HTTP_200_OK,
)
async def group_claim(request: GroupClaimRequest) -> GroupClaimResponse | JSONResponse:
  logger.info(
    f"Group claim request - Tags: {len(request.tags)}"
  )

  try:
    result = await assign_claim_tag(
      claim=request.claim,
      candidate_tags=request.tags,
    )

    response_data = {
      "error": None,
      **result,
    }

    logger.debug(f"Response body: {json.dumps(response_data)}")

    return GroupClaimResponse(**response_data)
  except Exception as e:
    logger.error(f"Group claim failed - {type(e).__name__}: {str(e)}")

    return JSONResponse(
      content={
        "error": "An internal error occurred. Please try again later.",
        "claim": request.claim,
        "tags": request.tags,
        "assigned_tag": None,
      },
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
