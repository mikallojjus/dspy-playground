import asyncio
import json
import re
from typing import Any, Dict, List

from src.api.utils import llm_model
from src.config.prompts.group_claim_prompt import GROUP_CLAIM_PROMPT


def _parse_results(
  raw_response: str,
  claims: List[str] | None = None,
) -> List[Dict[str, Any]]:
  try:
    response = json.loads(raw_response)
  except Exception:
    try:
      response_text = raw_response.strip()
      fenced_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        response_text,
        re.DOTALL | re.IGNORECASE,
      )

      if fenced_match:
        response_text = fenced_match.group(1)

      response = json.loads(response_text)
    except Exception:
      raise Exception("Failed parsing response")

  if not isinstance(response, dict):
    raise Exception("Invalid response format")

  grouped_topics = response.get("grouped_topics")
  if grouped_topics is not None:
    if not isinstance(grouped_topics, dict):
      raise Exception("Invalid grouped_topics format")

    claim_topics: Dict[str, List[str]] = {}
    for topic, topic_claims in grouped_topics.items():
      if not isinstance(topic, str):
        raise Exception("Invalid grouped_topics format")
      if not isinstance(topic_claims, list):
        raise Exception("Invalid grouped_topics format")

      for claim in topic_claims:
        if not isinstance(claim, str):
          continue
        topics = claim_topics.setdefault(claim, [])
        if topic not in topics:
          topics.append(topic)

    if claims is not None:
      parsed_results = []
      for claim in claims:
        topics = claim_topics.get(claim)
        if topics is None:
          continue
        parsed_results.append({"claim": claim, "discussion_topics": topics})
      return parsed_results

    return [
      {"claim": claim, "discussion_topics": topics}
      for claim, topics in claim_topics.items()
    ]
  
  raise Exception("Failed extracting grouped_topics")


async def group_claims_by_topic(
  claims: List[str],
) -> List[Dict[str, Any]]:
  if not claims:
    return []

  try:
    chain = llm_model.build_chain(
      prompt=GROUP_CLAIM_PROMPT
    )
  except Exception as e:
    raise Exception("Error building chain")

  try:
    raw_response = await asyncio.to_thread(
      chain.invoke,
      {
        "CLAIMS": json.dumps(claims),
      },
    )
  except Exception as e:
    raise Exception("Failed invoking chain")

  return _parse_results(raw_response, claims=claims)
