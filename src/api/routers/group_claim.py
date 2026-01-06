from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import json

from src.api.schemas.group_claim_schema import GroupClaimRequest, GroupClaimResponse
from src.api.services.group_claim_service import group_claims_by_topic
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
    f"Group claim request - Claims: {len(request.claims)}"
  )

  try:
    results = await group_claims_by_topic(
      claims=request.claims,
    )

    response_data = {
      "error": None,
      "results": results,
    }

    logger.debug(f"Response body: {json.dumps(response_data)}")

    return GroupClaimResponse(**response_data)
  except Exception as e:
    logger.error(f"Group claim failed - {type(e).__name__}: {str(e)}")

    return JSONResponse(
      content={
        "error": "An internal error occurred. Please try again later.",
        "results": [],
      },
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
