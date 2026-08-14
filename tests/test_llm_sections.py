from types import SimpleNamespace

import pytest

from research_pipeline.llm_sections import (
    LLMSectionsError,
    SectionFormatError,
    invoke_sections,
    parse_sections,
    render_section,
    render_sections,
)


class FakeChatModel:
    """Returns queued raw contents in order, recording the messages it was
    given — enough to check what the repair turn actually quotes back. Same
    shape as tests/test_llm_json.py's fake, deliberately."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.call_kwargs = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        return SimpleNamespace(content=self.responses.pop(0))


CODE_FIELDS = ["load_data_function", "needs_gpu"]


def _response(sections: dict[str, str]) -> str:
    return render_sections(sections)


# -- parse_sections: required fields ------------------------------------------------------


def test_parse_sections_extracts_each_field():
    text = _response({"a": "line one\nline two", "b": "true"})
    assert parse_sections(text, ["a", "b"]) == {"a": "line one\nline two", "b": "true"}


def test_parse_sections_keeps_code_byte_for_byte():
    # The reason this transport exists: no escape/unescape step, so a regex
    # escape, a literal tab-looking sequence and a trailing backslash all arrive
    # exactly as written. Through JSON each of these needs hand-doubling that a
    # small quantized model gets wrong.
    code = 'import re\nP = re.compile(r"\\d+\\s")\nS = "a\\tb"\nX = 1 + \\\n    2\n'
    assert parse_sections(_response({"helpers": code}), ["helpers"])["helpers"] == code


def test_parse_sections_accepts_an_empty_section():
    text = "===BEGIN imports===\n===END imports==="
    assert parse_sections(text, ["imports"]) == {"imports": ""}


def test_parse_sections_ignores_text_outside_the_markers():
    text = f"Here you go:\n\n{render_section('a', 'x = 1')}\n\nHope that helps!"
    assert parse_sections(text, ["a"]) == {"a": "x = 1"}


def test_parse_sections_still_parses_a_fenced_response():
    # Markers are found by search rather than by anchoring to the start of the
    # response, so a model that wraps everything in a code fence still parses —
    # and fences are never stripped, which would corrupt a README section that
    # legitimately contains one.
    text = f"```\n{render_section('a', 'x = 1')}\n```"
    assert parse_sections(text, ["a"]) == {"a": "x = 1"}


def test_parse_sections_ignores_sections_that_were_not_asked_for():
    text = _response({"a": "1", "unexpected": "2"})
    assert parse_sections(text, ["a"]) == {"a": "1"}


def test_parse_sections_reports_which_fields_are_missing():
    text = _response({"a": "1"})
    with pytest.raises(SectionFormatError) as excinfo:
        parse_sections(text, ["a", "b", "c"])
    assert excinfo.value.missing == ["b", "c"]
    assert "b" in str(excinfo.value)


def test_parse_sections_treats_an_unterminated_section_as_missing():
    # What a completion truncated mid-answer looks like.
    text = "===BEGIN a===\nx = 1\n"
    with pytest.raises(SectionFormatError) as excinfo:
        parse_sections(text, ["a"])
    assert excinfo.value.missing == ["a"]


def test_parse_sections_rejects_a_mismatched_marker_pair():
    text = "===BEGIN a===\nx = 1\n===END b==="
    with pytest.raises(SectionFormatError):
        parse_sections(text, ["a"])


def test_parse_sections_takes_the_first_of_a_duplicated_section():
    text = f"{render_section('a', 'first')}\n{render_section('a', 'second')}"
    assert parse_sections(text, ["a"])["a"] == "first"


# -- parse_sections: discovery mode (the shared-infrastructure call) ----------------------


def test_parse_sections_discovers_field_names_when_none_are_given():
    text = _response({"data_utils.py": "def load():\n    return 1\n", "README.md": "docs"})
    assert parse_sections(text) == {
        "data_utils.py": "def load():\n    return 1\n",
        "README.md": "docs",
    }


def test_parse_sections_discovery_raises_when_there_are_no_markers_at_all():
    with pytest.raises(SectionFormatError):
        parse_sections('{"files": {"a.py": "..."}}')


# -- invoke_sections ---------------------------------------------------------------------


def test_invoke_sections_parses_a_traced_response():
    # Reasoning traces are stripped by llm_json.strip_reasoning, imported rather
    # than reimplemented here — the <think> problem belongs to the model, not to
    # the transport.
    model = FakeChatModel(["<think>deciding</think>" + _response({"a": "x = 1", "b": "true"})])
    assert invoke_sections(model, "sys", "user", ["a", "b"]) == {"a": "x = 1", "b": "true"}


def test_invoke_sections_repairs_a_missing_section_on_one_retry():
    model = FakeChatModel([_response({"a": "x = 1"}), _response({"a": "x = 1", "b": "true"})])

    assert invoke_sections(model, "sys", "user", ["a", "b"]) == {"a": "x = 1", "b": "true"}

    repair_prompt = model.calls[1][3][1]
    assert "b" in repair_prompt  # names exactly what was missing
    assert "===BEGIN b===" in repair_prompt  # and shows the shape to use
    assert len(model.calls) == 2


def test_invoke_sections_repair_turn_quotes_the_answer_not_the_reasoning_trace():
    trace = "<think>" + "x" * 20000 + "</think>"
    model = FakeChatModel([trace + "no markers here", _response({"a": "x = 1"})])

    assert invoke_sections(model, "sys", "user", ["a"]) == {"a": "x = 1"}

    repair_messages = model.calls[1]
    assert repair_messages[2] == ("assistant", "no markers here")
    assert "x" * 100 not in repair_messages[3][1]
    assert "no markers here" in repair_messages[3][1]


def test_invoke_sections_raises_when_both_attempts_are_unparseable():
    model = FakeChatModel(["no markers", "still no markers"])
    with pytest.raises(LLMSectionsError):
        invoke_sections(model, "sys", "user", ["a"])


def test_invoke_sections_discovers_sections_when_no_field_names_are_given():
    model = FakeChatModel([_response({"utils.py": "def f():\n    return 1\n"})])
    assert invoke_sections(model, "sys", "user") == {"utils.py": "def f():\n    return 1\n"}


def test_invoke_sections_forwards_per_call_max_tokens_and_temperature():
    # Mirrors invoke_json's own overrides: the Coder Agent bounds max_tokens per
    # prompt and pins fix regeneration to temperature 0.
    model = FakeChatModel([_response({"a": "x"})])
    invoke_sections(model, "sys", "user", ["a"], max_tokens=123, temperature=0.0)
    assert model.call_kwargs == [{"max_tokens": 123, "temperature": 0.0}]


def test_invoke_sections_sends_no_extra_kwargs_by_default():
    # Guards the "every other agent is unaffected" contract — with neither
    # override this is exactly the call it was before those parameters existed.
    model = FakeChatModel([_response({"a": "x"})])
    invoke_sections(model, "sys", "user", ["a"])
    assert model.call_kwargs == [{}]


def test_invoke_sections_repair_turn_reuses_the_same_per_call_overrides():
    model = FakeChatModel(["no markers", _response({"a": "x"})])
    invoke_sections(model, "sys", "user", ["a"], max_tokens=123)
    assert model.call_kwargs == [{"max_tokens": 123}, {"max_tokens": 123}]


# -- render_section: the format's single definition ---------------------------------------


def test_render_section_round_trips_through_parse_sections():
    content = 'def f():\n    return {"a": 1}\n'
    assert parse_sections(render_section("f", content), ["f"])["f"] == content
