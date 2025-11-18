from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from services.keyword_extraction_service import extract_keyword_and_topics
from schemas.keyword_extraction_schema import (
  KeywordExtractionRequest,
)

router = APIRouter()

@router.post("/keyword_extraction")
def keyword_extraction(request: KeywordExtractionRequest) -> JSONResponse:
  try:
    keywords, topics = extract_keyword_and_topics(
      episode=request.episode,
      topics_list=request.topics_list,
      min_keywords=request.min_keywords,
      max_keywords=request.max_keywords,
      min_topics=request.min_topics,
      max_topics=request.max_topics,
    )
    return JSONResponse(
      content={
        "error":None,
        "keywords":keywords,
        "topics":topics
      }, 
      status_code=status.HTTP_200_OK
    )
  except Exception as e:
    return JSONResponse(
      content={
        "error":"An internal error occurred. Please try again later.",
        "keywords":None,
        "topics":None
      }, 
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )