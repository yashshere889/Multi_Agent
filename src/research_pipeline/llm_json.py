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


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def strip_reasoning(text: str) -> str:
    """Drops a Nemotron-style <think>...</think> reasoning trace, keeping only
    the answer after it.

    Nemotron 3 Nano reasons before answering, and whether the trace reaches us
    depends on how the server was started: with vLLM's nano_v3 reasoning parser
    it's split into `reasoning_content` and never appears here; without it, the
    raw trace is prepended to the content and every json.loads downstream fails.
    We can't control how the server was launched, so we strip here too.

    Only the *final* close tag is treated as the boundary — a trace that
    discusses tags itself still ends at its last </think>. An opening tag with
    no close means the completion was truncated mid-thought (usually
    LLM_MAX_TOKENS too low): everything is trace, so nothing is returned, and
    the caller's parse failure/repair retry handles it rather than us silently
    parsing half a thought as an answer."""
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1].strip()
    if THINK_OPEN in text:
        return text.split(THINK_OPEN, 1)[0].strip()
    return text.strip()


def strip_fences(text: str) -> str:
    text = strip_reasoning(text)
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
        # The repair turn quotes the *stripped* text, not the raw content: with
        # a reasoning model the raw content can be mostly <think> trace, which
        # would crowd the actual malformed JSON out of the 4000-char excerpt
        # and give the model nothing useful to repair.
        previous = strip_reasoning(response.content)
        messages.append(("assistant", previous))
        messages.append(("human", JSON_REPAIR_PROMPT.format(previous_response=previous[:4000])))
        response = chat_model.invoke(messages)
        text = strip_fences(response.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc2:
            raise LLMJSONError(
                f"Model did not return valid JSON, even after a repair attempt: {exc2}. "
                f"Raw response: {text[:500]!r}"
            ) from exc2
