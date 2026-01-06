import asyncio
import json
import re
from typing import Any, Dict, List

from src.api.utils import llm_model
from src.config.prompts.group_claim_prompt import GROUP_CLAIM_PROMPT


def _parse_results(raw_response: str) -> List[Dict[str, Any]]:
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

  try:
    results = response["results"]
  except KeyError:
    raise Exception("Failed extracting results")

  if not isinstance(results, list):
    raise Exception("Invalid results format")

  parsed_results = []
  for result in results:
    if not isinstance(result, dict):
      raise Exception("Invalid result format")
    claim = result.get("claim")
    topics = result.get("discussion_topics")
    if not isinstance(claim, str):
      raise Exception("Invalid result format")
    if not isinstance(topics, list):
      raise Exception("Invalid result format")

    cleaned_topics = [topic for topic in topics if isinstance(topic, str)]
    parsed_results.append({"claim": claim, "discussion_topics": cleaned_topics})

  return parsed_results


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

  return _parse_results(raw_response)
