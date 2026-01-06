from typing import List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class GroupClaimRequest(BaseModel):
  model_config = ConfigDict(populate_by_name=True)

  claims: List[str] = Field(validation_alias=AliasChoices("claims", "claim"))


class GroupClaimResult(BaseModel):
  claim: str
  discussion_topics: List[str]


class GroupClaimResponse(BaseModel):
  error: str | None = None
  results: List[GroupClaimResult] = Field(default_factory=list)
