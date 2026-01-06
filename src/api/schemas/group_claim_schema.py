from typing import Dict, List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class GroupClaimRequest(BaseModel):
  model_config = ConfigDict(populate_by_name=True)

  claims: List[str] = Field(validation_alias=AliasChoices("claims", "claim"))


class GroupClaimResponse(BaseModel):
  error: str | None = None
  result: Dict[str, List[str]] = Field(default_factory=dict)
