import os
from typing import Any, Dict
from config.config import LLMConfig

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI


def _create_chat_model(
  model_name: str = LLMConfig.DEFAULT_MODEL_NAME, 
  temperature: float = LLMConfig.DEFAULT_MODEL_TEMPERATURE,
) -> BaseChatModel:
  return ChatGoogleGenerativeAI(
    model=model_name,
    temperature=temperature,
    api_key=os.getenv("GEMINI_API_KEY"),
  )

def build_chain(
  prompt: str,
  model_name: str = LLMConfig.DEFAULT_MODEL_NAME, 
  temperature: float = LLMConfig.DEFAULT_MODEL_TEMPERATURE,
) -> RunnableSequence[Dict[str, Any], str]:
  model = _create_chat_model(
    model_name=model_name,
    temperature=temperature,
  )
  prompt = ChatPromptTemplate.from_template(prompt)
  return prompt | model | StrOutputParser()