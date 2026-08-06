"""Factory for the shared chat model, pointed at whatever OpenAI-compatible
server is configured via LLM_BASE_URL (local box, SSH-tunneled Barkla node, etc.)."""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from research_pipeline.config import settings


def get_chat_model(temperature: Optional[float] = None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
    )
