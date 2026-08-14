"""Shared helper for LLM calls that must return raw multi-line text — in
practice, generated source code — with one repair retry on malformed output.
The text-shaped counterpart to llm_json.py, which stays the transport for
short/structured fields.

Why a second transport exists at all: asking the model for code inside a JSON
string value forces every newline, quote and backslash in that code through an
escaping round-trip the model has to perform by hand. Small quantized models
get that wrong constantly, and each flavour of "wrong" needs its own
deterministic repair downstream (llm_json._fix_invalid_escapes for literal
backslashes, _loads_lenient's strict=False for literal newlines,
sandbox._strip_redundant_line_continuations for the backslashes that survive
into the rendered file). Those repairs are patches on a transport that is a bad
fit for the payload.

The format below removes the round-trip instead of repairing it:

    ===BEGIN load_data_function===
    def load_data():
        return {"x": 1}
    ===END load_data_function===

Nothing is escaped, on either side. A real newline is a real newline and a real
backslash is a real backslash: parsing is "find the two markers, take the bytes
between them verbatim". The markers themselves are the only reserved text, and
they can't occur in valid Python at the start of a line.

Reasoning traces are handled exactly once, by importing llm_json.strip_reasoning
rather than restating it — the <think> problem is a property of the model, not of
the transport. Markdown fences are deliberately *not* stripped: the markers are
found by search rather than by anchoring to the start of the response, so a
fenced response parses as-is, and stripping fences could corrupt a code section
that legitimately contains one (e.g. inside a README).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.llm_json import strip_reasoning

logger = logging.getLogger(__name__)


class SectionFormatError(ValueError):
    """Raised when a response doesn't carry every requested section.

    `missing` is the list of field names whose BEGIN/END marker pair couldn't be
    found — carried as data (not just prose) so the repair turn can name exactly
    what to fix instead of asking for the whole response again blindly.
    """

    def __init__(self, missing: Sequence[str], detail: str = "") -> None:
        self.missing = list(missing)
        message = f"Response is missing or malformed for section(s): {self.missing}"
        if detail:
            message = f"{message}. {detail}"
        super().__init__(message)


class LLMSectionsError(RuntimeError):
    """Raised when the model doesn't return the delimited format even after a
    repair retry — the text-transport counterpart of LLMJSONError."""


def begin_marker(field: str) -> str:
    return f"===BEGIN {field}==="


def end_marker(field: str) -> str:
    return f"===END {field}==="


def render_section(field: str, content: str) -> str:
    """One field's wire representation. The single definition of the format,
    used by prompts.py to spell the shape out for the model and by the tests to
    build fake responses — so neither can drift from what parse_sections
    accepts."""
    return f"{begin_marker(field)}\n{content}\n{end_marker(field)}"


def render_sections(sections: dict[str, str]) -> str:
    return "\n\n".join(render_section(field, content) for field, content in sections.items())


# A field name is an identifier-ish token: the experiment sections are plain
# identifiers, and the shared-infrastructure call names its sections after the
# files it's generating ("data_utils.py"), so dots/dashes/slashes are allowed
# too. The (?P=name) backreference is what makes a mismatched pair
# (===BEGIN a=== ... ===END b===) a miss rather than a silent mis-parse.
_SECTION_RE_TEMPLATE = r"^[ \t]*===BEGIN[ \t]+{name}[ \t]*===[ \t]*\r?\n(?P<body>.*?)^[ \t]*===END[ \t]+{end_name}[ \t]*==="
_ANY_SECTION_RE = re.compile(
    _SECTION_RE_TEMPLATE.format(name=r"(?P<name>[\w.\-/]+)", end_name=r"(?P=name)"),
    re.DOTALL | re.MULTILINE,
)


def _field_re(field: str) -> re.Pattern[str]:
    escaped = re.escape(field)
    return re.compile(
        _SECTION_RE_TEMPLATE.format(name=escaped, end_name=escaped),
        re.DOTALL | re.MULTILINE,
    )


def _clean_body(body: str) -> str:
    """Drops only the single line break that separates the content from its END
    marker. Everything else — indentation, blank lines, backslashes, trailing
    whitespace inside the block — is kept exactly as the model wrote it, which
    is the entire point of this transport."""
    if body.endswith("\r\n"):
        return body[:-2]
    if body.endswith("\n"):
        return body[:-1]
    return body


def parse_sections(text: str, field_names: Sequence[str] | None = None) -> dict[str, str]:
    """Extracts each `===BEGIN <field>===`/`===END <field>===` block's content,
    verbatim.

    With `field_names`, those fields are required and anything else in the
    response is ignored; a missing or unterminated one raises
    SectionFormatError. With `field_names=None` the field names are *discovered*
    from the response instead, which is what the shared-infrastructure call
    needs — it asks for one section per generated file, and the filenames aren't
    known until the model picks them. Discovery still requires at least one
    section, so an unparseable response is a failure rather than an empty dict.
    """
    if field_names is None:
        found = {
            match.group("name"): _clean_body(match.group("body"))
            for match in _ANY_SECTION_RE.finditer(text)
        }
        if not found:
            raise SectionFormatError(
                ["(any)"],
                "No ===BEGIN <name>===/===END <name>=== section markers were found at all",
            )
        return found

    sections: dict[str, str] = {}
    missing: list[str] = []
    for field in field_names:
        match = _field_re(field).search(text)
        if match is None:
            missing.append(field)
            continue
        sections[field] = _clean_body(match.group("body"))
    if missing:
        raise SectionFormatError(missing)
    return sections


SECTIONS_REPAIR_PROMPT = """Your previous response was not in the required delimited format. \
These section(s) were missing, or had no matching end marker: {missing_fields}.

Return the WHOLE answer again, with every requested section present, each \
wrapped in its own markers exactly like this (one BEGIN line, the raw content, \
one END line):

{expected_format}

Rules for the format itself:
- Never escape anything. Write real line breaks, real quotes and real \
backslashes — this is not JSON, so "\\n" must be an actual new line, not two \
characters.
- Nothing may appear between an END marker and the next BEGIN marker.
- Do not wrap the response in markdown fences or add commentary before or after it.

Your previous response was:
{previous_response}
"""


def _response_text(response: object) -> str:
    """A chat message's `content` is typed `str | list[...]` for multimodal
    replies. Every model this pipeline talks to returns a plain string, and a
    list payload wouldn't carry parseable sections anyway — coerced once here so
    the whole parse path stays str-typed."""
    content = getattr(response, "content", "")
    return content if isinstance(content, str) else str(content)


def _expected_format(field_names: Sequence[str] | None) -> str:
    if not field_names:
        return "===BEGIN <section name>===\n<raw content>\n===END <section name>==="
    return "\n\n".join(render_section(field, f"<{field} content>") for field in field_names)


def invoke_sections(
    chat_model: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    field_names: Sequence[str] | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, str]:
    """Invokes chat_model expecting the delimited format back; on a parse
    failure, retries once with an explicit repair prompt naming the sections
    that were missing before raising LLMSectionsError.

    Mirrors llm_json.invoke_json's shape deliberately — same message sequence,
    same single repair turn quoting the reasoning-stripped previous response,
    same "raise rather than return something half-parsed" contract — so a call
    site swaps between the two transports without changing anything else.

    Every value returned is raw text. Turning a section into a bool or a list
    (`needs_gpu`, `assumptions_made`) is the caller's job, on purpose: this
    module owns the transport, and how a field's text should be interpreted is
    domain knowledge belonging to the agent that asked for it — the same
    "Python decides what the model's output means" split the rest of the
    pipeline uses.

    `max_tokens`/`temperature` override, for this call only, whatever the client
    was constructed with (see llm.get_chat_model) — mirroring invoke_json's own
    two optional parameters. Both default to None, and when both are None this
    is byte-for-byte the same invocation as before they existed.
    """
    messages = [("system", system_prompt), ("human", user_prompt)]
    invoke_kwargs: dict = {}
    if max_tokens is not None:
        invoke_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        invoke_kwargs["temperature"] = temperature
    response = chat_model.invoke(messages, **invoke_kwargs)
    # Stripped once and reused: with a reasoning model the raw content can be
    # mostly <think> trace, and the repair turn must quote the *answer* — a trace
    # would crowd the malformed answer out of the excerpt below and give the
    # model nothing useful to repair. Same reasoning as invoke_json's.
    previous = strip_reasoning(_response_text(response))
    try:
        return parse_sections(previous, field_names)
    except SectionFormatError as exc:
        logger.warning(
            "Model response was not in the required delimited format (%s) — "
            "retrying once with a repair prompt",
            exc,
        )
        messages.append(("assistant", previous))
        messages.append(
            (
                "human",
                SECTIONS_REPAIR_PROMPT.format(
                    missing_fields=", ".join(exc.missing),
                    expected_format=_expected_format(field_names),
                    previous_response=previous[:4000],
                ),
            )
        )
        response = chat_model.invoke(messages, **invoke_kwargs)
        stripped = strip_reasoning(_response_text(response))
        try:
            return parse_sections(stripped, field_names)
        except SectionFormatError as exc2:
            raise LLMSectionsError(
                f"Model did not return the required delimited format, even after a repair "
                f"attempt: {exc2}. Raw response: {stripped[:500]!r}"
            ) from exc2
