from pydantic import BaseModel

class GuestExtractionRequest(BaseModel):
  title: str
  description: str
  transcript: str | None