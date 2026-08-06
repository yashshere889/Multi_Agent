"""Shared helper for LLM calls that must return a JSON object, with one
repair retry on malformed output. Used by every agent that asks the model for
structured JSON — factored out once a second agent (experiment_planner)
needed the exact same retry logic the hypothesis agent already had.
"""

from __future__ import annotations

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMJSONError(RuntimeError):
    """Raised when the model doesn't return valid JSON even after a repair retry."""


JSON_REPAIR_PROMPT = """Your previous response was not valid JSON matching the requested \
schema. Return ONLY the corrected, valid JSON object — no markdown fences, no \
commentary, nothing before or after it.

Your previous response was:
{previous_response}
"""


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def invoke_json(chat_model: BaseChatModel, system_prompt: str, user_prompt: str) -> dict:
    """Invokes chat_model expecting a JSON object back; on a parse failure,
    retries once with an explicit repair prompt before raising LLMJSONError."""
    messages = [("system", system_prompt), ("human", user_prompt)]
    response = chat_model.invoke(messages)
    text = strip_fences(response.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Model returned invalid JSON (%s) — retrying once with a repair prompt", exc)
        messages.append(("assistant", response.content))
        messages.append(("human", JSON_REPAIR_PROMPT.format(previous_response=response.content[:4000])))
        response = chat_model.invoke(messages)
        text = strip_fences(response.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc2:
            raise LLMJSONError(
                f"Model did not return valid JSON, even after a repair attempt: {exc2}. "
                f"Raw response: {text[:500]!r}"
            ) from exc2
