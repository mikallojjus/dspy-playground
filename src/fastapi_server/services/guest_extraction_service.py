import json
from typing import Any, Dict, List
from prompts.guest_extraction_prompt import GUEST_EXTRACTION_PROMPT
from utils import llm_model


def extract_podcast_guests(
    title: str,
    description: str,
    transcript: str,
) -> List[Dict[str, Any]]:
  try:
    chain = llm_model.build_chain(
      prompt=GUEST_EXTRACTION_PROMPT
    )
  except Exception as e:
    raise Exception("Error building chain")
  
  try:
    raw_response = chain.invoke({
      "title": title,
      "description": description,
      "transcript": transcript,
    })
  except Exception as e:
    raise Exception("Failed invoking chain")
  
  try:
    response = json.loads(raw_response)
  except Exception as e:
    raise Exception("Failed parsing response")
  
  try:
    guests =  response["guests"]
  except KeyError:
    raise Exception("Failed extracting guests")
  
  for guest in guests:
    if "name" not in guest or "urls" not in guest:
      raise Exception("Invalid guest format")
    if not isinstance(guest["urls"], list):
      raise Exception("Invalid guest format")
  return guests
