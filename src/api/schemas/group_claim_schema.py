from typing import List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class GroupClaimRequest(BaseModel):
  model_config = ConfigDict(populate_by_name=True)

  claim: str = Field(validation_alias=AliasChoices("claim", "claims"))
  tags: List[str]


class GroupClaimResponse(BaseModel):
  error: str | None = None
  claim: str
  tags: List[str]
  assigned_tag: str | None = None
