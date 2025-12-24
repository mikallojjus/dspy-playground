import json
from typing import Any, Dict, List
from src.config.prompts.host_extraction_prompt import HOST_EXTRACTION_PROMPT
from src.api.utils import llm_model


def extract_podcast_hosts(
    title: str,
    description: str,
    truncated_transcript: str,
) -> List[Dict[str, Any]]:
  try:
    chain = llm_model.build_chain(
      prompt=HOST_EXTRACTION_PROMPT
    )
  except Exception as e:
    raise Exception("Error building chain")

  try:
    raw_response = chain.invoke({
      "title": title,
      "description": description,
      "truncated_transcript": truncated_transcript,
    })
  except Exception as e:
    raise Exception("Failed invoking chain")

  try:
    response = json.loads(raw_response)
  except Exception as e:
    raise Exception("Failed parsing response")

  try:
    hosts = response["hosts"]
  except KeyError:
    raise Exception("Failed extracting hosts")

  for host in hosts:
    if "name" not in host or "urls" not in host:
      raise Exception("Invalid host format")
    if not isinstance(host["urls"], list):
      raise Exception("Invalid host format")
  return hosts
