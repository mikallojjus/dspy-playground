from pydantic import BaseModel

class HostExtractionRequest(BaseModel):
  title: str
  description: str
  truncated_transcript: str
