"""Shared helper for LLM calls that must return a JSON object, with one
repair retry on malformed output. Used by every agent that asks the model for
structured JSON — factored out once a second agent (experiment_planner)
needed the exact same retry logic the hypothesis agent already had.
"""

from __future__ import annotations

import difflib
import json
import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMJSONError(RuntimeError):
    """Raised when the model doesn't return valid JSON even after a repair retry.

    Carries the text that couldn't be parsed, and whether the model was cut off
    by the completion token cap rather than simply returning malformed JSON.
    Both exist for callers that can do better than giving up: a JSON array cut
    off mid-generation is well-formed right up to the cut, so the entries before
    it are recoverable (salvage_json_list), and `truncated` tells a caller the
    problem was the *size* of what it asked for, not the model's syntax.
    """

    def __init__(self, message: str, *, text: str = "", truncated: bool = False) -> None:
        super().__init__(message)
        self.text = text
        self.truncated = truncated


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


def _was_truncated(response: object) -> bool:
    """Did the model stop because it hit the completion cap, or because it was done?

    OpenAI-compatible servers (vLLM, which is what llm.py talks to) report this
    as finish_reason="length"; Anthropic-style clients use
    stop_reason="max_tokens". Both are checked so this keeps working if
    get_chat_model is pointed elsewhere, and an unrecognized shape reads as
    "not truncated", i.e. the repair retry that has always happened.
    """
    metadata = getattr(response, "response_metadata", None) or {}
    if not isinstance(metadata, dict):
        return False
    return metadata.get("finish_reason") == "length" or metadata.get("stop_reason") in ("max_tokens", "length")


_SALVAGE_DECODER = json.JSONDecoder(strict=False)


def salvage_json_list(text: str, key: str) -> list:
    """The complete objects from text's `"<key>": [...]` array, ignoring
    whatever follows the last one that parses.

    For a response the model never finished: everything before the cut is
    well-formed, so the entries that did arrive are real and discarding them
    along with the parse failure throws away work that is already paid for. On
    Barkla job 10279682 a Reviewer hallucination check produced ~70 findings
    and was cut mid-string in the next one; all 70 were lost, and with them the
    entire pipeline run.

    Never raises — it only ever runs on a response that already failed to
    parse — and returns [] when the key isn't present or nothing parses.
    """
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if not match:
        return []
    entries: list = []
    index = match.end()
    while index < len(text):
        while index < len(text) and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text) or text[index] == "]":
            break
        try:
            entry, index = _SALVAGE_DECODER.raw_decode(text, index)
        except ValueError:
            break
        entries.append(entry)
    return entries


def invoke_json(
    chat_model: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict:
    """Invokes chat_model expecting a JSON object back; on a parse failure,
    retries once with an explicit repair prompt before raising LLMJSONError.

    `max_tokens`/`temperature` override, for this call only, whatever the client
    was constructed with (see llm.get_chat_model). Both default to None, and
    when both are None this is byte-for-byte the same invocation as before they
    existed — which is every caller except the Coder Agent, whose prompts can
    grow long enough that a fixed max_tokens overruns the model's context
    window (see coder_agent._bounded_max_tokens)."""
    messages = [("system", system_prompt), ("human", user_prompt)]
    invoke_kwargs: dict = {}
    if max_tokens is not None:
        invoke_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        invoke_kwargs["temperature"] = temperature
    response = chat_model.invoke(messages, **invoke_kwargs)
    text = strip_fences(response.content)
    try:
        return _loads_lenient(text)
    except json.JSONDecodeError as exc:
        if _was_truncated(response):
            # A repair turn cannot fix this, so none is attempted. It re-sends
            # the same prompt under the same max_tokens, so the model produces
            # the same over-long answer and it is cut in the same place: on
            # Barkla job 10279682 both attempts died at "Unterminated string
            # ... (char 74650)", three minutes of generation spent reaching a
            # byte-identical failure. What has to change is how much was asked
            # for — or the caller salvaging what did arrive.
            raise LLMJSONError(
                f"Model's response was cut off by the completion token cap before it finished "
                f"the JSON: {exc}. Raw response: {text[:500]!r}",
                text=text,
                truncated=True,
            ) from exc
        logger.warning("Model returned invalid JSON (%s) — retrying once with a repair prompt", exc)
        # The repair turn quotes the *stripped* text, not the raw content: with
        # a reasoning model the raw content can be mostly <think> trace, which
        # would crowd the actual malformed JSON out of the 4000-char excerpt
        # and give the model nothing useful to repair.
        previous = strip_reasoning(response.content)
        messages.append(("assistant", previous))
        messages.append(("human", JSON_REPAIR_PROMPT.format(previous_response=previous[:4000])))
        response = chat_model.invoke(messages, **invoke_kwargs)
        text = strip_fences(response.content)
        try:
            return _loads_lenient(text)
        except json.JSONDecodeError as exc2:
            raise LLMJSONError(
                f"Model did not return valid JSON, even after a repair attempt: {exc2}. "
                f"Raw response: {text[:500]!r}",
                text=text,
                truncated=_was_truncated(response),
            ) from exc2


# How close a misspelled key must be to an expected one before it is renamed.
# 0.85 admits ordinary single-character slips ("statment", "ratonale", both
# ~0.94) and rejects unrelated field names ("justification" vs "statement" is
# 0.46). It deliberately does NOT admit the doubling degeneration below, which
# scores far lower — that one gets its own exact rule rather than a looser
# threshold, because loosening this far enough to catch it would start renaming
# genuinely different fields onto each other.
_KEY_MATCH_CUTOFF = 0.85


def _collapse_doubled_substring(text: str) -> set[str]:
    """Every string obtainable from `text` by deleting one immediately-repeated
    run of 2+ characters.

    This targets a specific small-model degeneration: the model emits a key with
    an internal syllable repeated — `rationationale` for `rationale`
    ("ration" + "ationale", where "ation" appears twice), or
    `statementstatement` for `statement`. Both are one deletion away from the
    right key, and neither is close enough for a fuzzy ratio to catch without
    setting the threshold so low that unrelated fields start matching.

    Returns a set because a long key can have more than one candidate collapse;
    the caller only acts when exactly one of them is an expected name.
    """
    candidates: set[str] = set()
    length = len(text)
    for run in range(length // 2, 1, -1):
        for start in range(length - 2 * run + 1):
            if text[start : start + run] == text[start + run : start + 2 * run]:
                candidates.add(text[:start] + text[start + run :])
    return candidates


def repair_keys(payload: dict, expected: list[str]) -> dict:
    """Rename near-miss keys in `payload` onto the `expected` names they meant.

    A shallow copy is returned; the input is untouched. Only ever renames an
    **unexpected** key onto an **expected key that is missing**, and only when
    exactly one candidate matches — so a payload that already has all its fields
    is returned unchanged, and a genuinely unrecognised extra field is left
    alone rather than being forced onto whichever expected name it is nearest.

    Why this exists rather than a repair retry: a production run on Barkla
    (job 10334292) lost a whole question — literature, cross-field search and
    hypothesis generation, three minutes of GPU — because the model wrote
    `rationationale` instead of `rationale` on two of three hypotheses. The
    rationales themselves were complete and correct. Asking the model again
    costs a round-trip to fix a defect that is decidable here, and it may well
    reproduce the same degeneration; this is the same "deterministic repair over
    another model call" trade the JSON escape fixes above already make.

    The expected names are the caller's to supply: which keys a response should
    have is domain knowledge belonging to the agent that asked for them, the
    same split llm_sections.py documents for section names.
    """
    if not isinstance(payload, dict):
        return payload

    missing = [name for name in expected if name not in payload]
    if not missing:
        return payload

    unexpected = [key for key in payload if key not in expected]
    if not unexpected:
        return payload

    repaired = dict(payload)
    for name in missing:
        matches = [key for key in unexpected if name in _collapse_doubled_substring(key)]
        if not matches:
            matches = difflib.get_close_matches(name, unexpected, n=2, cutoff=_KEY_MATCH_CUTOFF)
        # Exactly one, or it is a guess: two unexpected keys equally near one
        # expected name means the response is malformed in a way renaming cannot
        # settle, and validation should say so rather than this picking.
        if len(matches) != 1:
            continue
        source = matches[0]
        repaired[name] = repaired.pop(source)
        unexpected.remove(source)
        logger.warning("Repaired malformed response key %r -> %r", source, name)
    return repaired
