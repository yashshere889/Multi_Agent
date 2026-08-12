from dataclasses import replace
from types import SimpleNamespace

import pytest

from research_pipeline import llm
from research_pipeline.llm_json import LLMJSONError, invoke_json, strip_fences, strip_reasoning


class FakeChatModel:
    """Returns queued raw contents in order, recording the messages it was
    given — enough to check what the repair turn actually quotes back."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.responses.pop(0))


# -- strip_reasoning: Nemotron <think> traces --------------------------------------------


def test_strip_reasoning_drops_a_complete_think_block():
    assert strip_reasoning('<think>let me plan this out</think>{"a": 1}') == '{"a": 1}'


def test_strip_reasoning_handles_a_trace_the_chat_template_already_opened():
    # vLLM can emit the closing tag only, when the template pre-opens the turn.
    assert strip_reasoning('reasoning about it</think>{"a": 1}') == '{"a": 1}'


def test_strip_reasoning_splits_on_the_final_close_tag():
    raw = "<think>the answer uses a </think> tag as an example</think>DONE"
    assert strip_reasoning(raw) == "DONE"


def test_strip_reasoning_returns_nothing_for_a_truncated_trace():
    # No close tag = the completion ran out of tokens mid-thought; there is no
    # answer to salvage, and returning half a thought would be worse than failing.
    assert strip_reasoning("<think>still thinking and then the tokens ran ou") == ""


def test_strip_reasoning_leaves_untraced_output_alone():
    assert strip_reasoning('  {"a": 1}  ') == '{"a": 1}'


def test_strip_fences_removes_a_trace_and_the_fences_around_the_answer():
    assert strip_fences('<think>plan</think>```json\n{"a": 1}\n```') == '{"a": 1}'


# -- invoke_json -------------------------------------------------------------------------


def test_invoke_json_parses_a_traced_response():
    model = FakeChatModel(['<think>deciding</think>{"hypotheses": []}'])
    assert invoke_json(model, "sys", "user") == {"hypotheses": []}


def test_invoke_json_repair_turn_quotes_the_answer_not_the_reasoning_trace():
    trace = "<think>" + "x" * 20000 + "</think>"
    model = FakeChatModel([trace + "{oops not json}", '{"ok": true}'])

    assert invoke_json(model, "sys", "user") == {"ok": True}

    repair_messages = model.calls[1]
    assert repair_messages[2] == ("assistant", "{oops not json}")
    assert "x" * 100 not in repair_messages[3][1]
    assert "{oops not json}" in repair_messages[3][1]


def test_invoke_json_raises_when_both_attempts_are_unparseable():
    model = FakeChatModel(["not json", "still not json"])
    with pytest.raises(LLMJSONError):
        invoke_json(model, "sys", "user")


def test_invoke_json_repairs_stray_backslashes_from_embedded_code_without_a_retry():
    # A generated-code string containing a literal regex escape like "\d"
    # isn't valid JSON (only \" \\ \/ \b \f \n \r \t \u are), but it's a
    # deterministic mistake the model will just repeat on a repair turn — so
    # this must be fixed locally, in a single model call.
    model = FakeChatModel([r'{"files": {"a.py": "import re\nPATTERN = \d+"}}'])
    assert invoke_json(model, "sys", "user") == {
        "files": {"a.py": "import re\nPATTERN = \\d+"}
    }
    assert len(model.calls) == 1


# -- llm.get_chat_model: Nemotron request fields -----------------------------------------


def test_get_chat_model_sends_the_configured_thinking_mode(monkeypatch):
    # Pinned rather than read from the ambient .env, so the assertion doesn't
    # flip with whoever's LLM_ENABLE_THINKING is loaded.
    monkeypatch.setattr(llm, "settings", replace(llm.settings, llm_enable_thinking=False))
    assert llm.get_chat_model().extra_body == {"chat_template_kwargs": {"enable_thinking": False}}

    monkeypatch.setattr(llm, "settings", replace(llm.settings, llm_enable_thinking=True))
    assert llm.get_chat_model().extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_get_chat_model_sends_reasoning_budget_only_when_thinking_is_on(monkeypatch):
    # Settings is a frozen dataclass, so swap in a whole replaced copy.
    monkeypatch.setattr(llm, "settings", replace(llm.settings, llm_reasoning_budget=512))

    assert "reasoning_budget" not in llm.get_chat_model(enable_thinking=False).extra_body
    thinking = llm.get_chat_model(enable_thinking=True)
    assert thinking.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_budget": 512,
    }


def test_get_chat_model_applies_configured_sampling_and_token_budget():
    model = llm.get_chat_model(temperature=0.1)
    assert model.temperature == 0.1
    assert model.top_p == llm.settings.llm_top_p
    assert model.max_tokens == llm.settings.llm_max_tokens


def test_get_chat_model_uses_the_configured_model_id():
    assert llm.get_chat_model().model_name == llm.settings.llm_model
