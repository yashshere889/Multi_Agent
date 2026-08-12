"""Shared helper for LLM calls that must return a JSON object, with one
repair retry on malformed output. Used by every agent that asks the model for
structured JSON — factored out once a second agent (experiment_planner)
needed the exact same retry logic the hypothesis agent already had.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMJSONError(RuntimeError):
    """Raised when the model doesn't return valid JSON even after a repair retry."""


JSON_REPAIR_PROMPT = """Your previous response was not valid JSON matching the requested \
schema. Return ONLY the corrected, valid JSON object — no markdown fences, no \
commentary, nothing before or after it.

Fix only the structural problem (e.g. a missing comma, bracket, or quote). Do \
NOT re-escape characters that are already correctly escaped — a backslash \
sequence like \\n, \\t, or \\" in your previous response is already valid JSON \
and must be copied through unchanged. Adding an extra backslash in front of \
one of these (e.g. turning \\n into \\\\n) corrupts the string it appears in \
and is a common mistake to avoid here.

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


_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _fix_invalid_escapes(text: str) -> str:
    """Doubles backslashes that aren't valid JSON escapes.

    Agents that ask the model to return source code as a JSON string value
    routinely get literal code back — e.g. a docstring or regex containing
    "\\d" or "\\_" — rather than the "\\\\d" JSON requires, because the model
    is reproducing code from its training data, not hand-encoding JSON. That
    breaks json.loads immediately and, since the model got it wrong in a
    structural way rather than a one-off typo, a repair turn tends to
    reproduce the same mistake. Fixing it deterministically avoids a wasted
    network round-trip and a second identical failure.
    """
    return _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", text)


def _loads_lenient(text: str) -> dict:
    # strict=False allows literal control characters (raw newlines, tabs) inside
    # JSON string values. Agents that ask for source code as a JSON string
    # routinely get it back with real newlines rather than "\n" — the model is
    # reproducing code from training data, not hand-encoding JSON — which
    # otherwise fails json.loads with "Invalid control character" the same way
    # _fix_invalid_escapes exists for literal backslashes.
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        return json.loads(_fix_invalid_escapes(text), strict=False)


def invoke_json(chat_model: BaseChatModel, system_prompt: str, user_prompt: str) -> dict:
    """Invokes chat_model expecting a JSON object back; on a parse failure,
    retries once with an explicit repair prompt before raising LLMJSONError."""
    messages = [("system", system_prompt), ("human", user_prompt)]
    response = chat_model.invoke(messages)
    text = strip_fences(response.content)
    try:
        return _loads_lenient(text)
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
            return _loads_lenient(text)
        except json.JSONDecodeError as exc2:
            raise LLMJSONError(
                f"Model did not return valid JSON, even after a repair attempt: {exc2}. "
                f"Raw response: {text[:500]!r}"
            ) from exc2
