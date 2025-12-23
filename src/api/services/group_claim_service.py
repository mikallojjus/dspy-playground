import asyncio
import json
import math
import re
from typing import Any, Dict, List

from src.api.utils import llm_model
from src.config.prompts.group_claim_prompt import GROUP_CLAIM_PROMPT
from src.infrastructure.embedding_service import EmbeddingService


def _dedupe_tags(tags: List[str]) -> List[str]:
  seen = set()
  deduped = []
  for tag in tags:
    if tag not in seen:
      seen.add(tag)
      deduped.append(tag)
  return deduped


def _parse_relevant_tags(raw_response: str) -> List[str]:
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
    except Exception as e:
      raise Exception("Failed parsing response")

  try:
    relevant_tags = response["relevant_tags"]
  except KeyError:
    raise Exception("Failed extracting relevant_tags")

  if not isinstance(relevant_tags, list):
    raise Exception("Invalid relevant_tags format")

  cleaned_tags = [tag for tag in relevant_tags if isinstance(tag, str)]
  return _dedupe_tags(cleaned_tags)


def _filter_candidate_tags(
  relevant_tags: List[str],
  candidate_tags: List[str],
) -> List[str]:
  candidate_set = set(candidate_tags)
  filtered = [tag for tag in relevant_tags if tag in candidate_set]
  return _dedupe_tags(filtered)


async def _pick_best_tag(
  claim: str,
  tags: List[str],
) -> str:
  embedder = EmbeddingService()
  embeddings = await embedder.embed_texts([claim] + tags)

  claim_embedding = embeddings[0]
  tag_embeddings = embeddings[1:]

  best_tag = None
  best_score = -1.0
  for tag, embedding in zip(tags, tag_embeddings):
    similarity = EmbeddingService.cosine_similarity(claim_embedding, embedding)
    if similarity > best_score or (
      math.isclose(similarity, best_score) and (best_tag is None or tag < best_tag)
    ):
      best_score = similarity
      best_tag = tag

  return best_tag


async def assign_claim_tag(
  claim: str,
  candidate_tags: List[str],
) -> Dict[str, Any]:
  try:
    chain = llm_model.build_chain(
      prompt=GROUP_CLAIM_PROMPT
    )
  except Exception as e:
    print(e)
    raise Exception("Error building chain")

  try:
    raw_response = await asyncio.to_thread(
      chain.invoke,
      {
        "CLAIM": claim,
        "CANDIDATE_TAGS": json.dumps(candidate_tags),
      },
    )
  except Exception as e:
    raise Exception("Failed invoking chain")

  relevant_tags = _parse_relevant_tags(raw_response)
  relevant_tags = _filter_candidate_tags(relevant_tags, candidate_tags)

  assigned_tag = None
  if len(relevant_tags) == 1:
    assigned_tag = relevant_tags[0]
  elif len(relevant_tags) > 1:
    assigned_tag = await _pick_best_tag(claim, relevant_tags)

  return {
    "claim": claim,
    "tags": candidate_tags,
    "assigned_tag": assigned_tag,
  }
