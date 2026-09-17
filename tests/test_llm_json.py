from dataclasses import replace
from types import SimpleNamespace

import pytest

from research_pipeline import llm
from research_pipeline.llm_json import LLMJSONError, invoke_json, salvage_json_list, strip_fences, strip_reasoning


class FakeChatModel:
    """Returns queued raw contents in order, recording the messages it was
    given — enough to check what the repair turn actually quotes back."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.call_kwargs = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        response = self.responses.pop(0)
        # A queued (content, metadata) pair stands in for a response the server
        # annotated — finish_reason="length" is how an OpenAI-compatible server
        # says it cut the model off at max_tokens.
        content, metadata = response if isinstance(response, tuple) else (response, {})
        return SimpleNamespace(content=content, response_metadata=metadata)


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


def test_invoke_json_repair_prompt_warns_against_re_escaping():
    # The repair turn re-quotes the model's own already-escaped JSON, which
    # invites the model to "fix" a correctly-escaped \n into a doubled \\n —
    # decoding to a literal backslash+'n' instead of a newline once the JSON
    # is parsed. The repair prompt must warn against this explicitly.
    model = FakeChatModel(["{oops not json}", '{"ok": true}'])
    invoke_json(model, "sys", "user")

    repair_prompt_text = model.calls[1][3][1]
    assert "re-escape" in repair_prompt_text
    assert r"\n" in repair_prompt_text


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
    assert invoke_json(model, "sys", "user") == {"files": {"a.py": "import re\nPATTERN = \\d+"}}
    assert len(model.calls) == 1


def test_invoke_json_forwards_per_call_max_tokens_and_temperature():
    # The Coder Agent bounds max_tokens per prompt (its fix prompts can otherwise
    # push prompt+completion past the model's context window) and pins fix
    # regeneration to temperature 0.
    model = FakeChatModel(['{"a": 1}'])
    invoke_json(model, "sys", "user", max_tokens=123, temperature=0.0)
    assert model.call_kwargs == [{"max_tokens": 123, "temperature": 0.0}]


def test_invoke_json_sends_no_extra_kwargs_by_default():
    # Guards the "every other agent is unaffected" contract: with neither
    # override the call is exactly what it was before those parameters existed,
    # so the client's constructor-time settings still apply untouched.
    model = FakeChatModel(['{"a": 1}'])
    invoke_json(model, "sys", "user")
    assert model.call_kwargs == [{}]


def test_invoke_json_repair_turn_reuses_the_same_per_call_overrides():
    # The retry must be bounded the same way the first call was — a repair turn
    # that dropped the cap would 400 for exactly the reason the cap exists.
    model = FakeChatModel(["not json", '{"a": 1}'])
    invoke_json(model, "sys", "user", max_tokens=123)
    assert model.call_kwargs == [{"max_tokens": 123}, {"max_tokens": 123}]


# -- llm.get_chat_model: provider request fields -----------------------------------------


def test_get_chat_model_sends_no_thinking_kwarg_when_thinking_is_off(monkeypatch):
    """Nothing is sent in the default configuration.

    This asserted `{"enable_thinking": False}` while the pipeline served
    Nemotron 3 Nano, where that kwarg exists. Qwen3-Coder-Instruct is not a
    reasoning model and its chat template has no such variable, so sending the
    field is a request field with no reader — and on some servers a template
    render error rather than a harmless no-op. The kwarg now goes out only when
    thinking is actually asked for, which keeps the reasoning-model path
    working without putting a dead field on every request.
    """
    # Pinned rather than read from the ambient .env, so the assertion doesn't
    # flip with whoever's LLM_ENABLE_THINKING is loaded.
    monkeypatch.setattr(llm, "settings", replace(llm.settings, llm_enable_thinking=False))
    assert llm.get_chat_model().extra_body == {}

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


# -- truncated responses -----------------------------------------------------------------


def test_invoke_json_does_not_spend_a_repair_turn_on_a_truncated_response():
    """Regression test for Barkla job 10279682: a Reviewer hallucination check
    came back cut off at the completion cap, and the repair turn — same prompt,
    same max_tokens — regenerated the same over-long answer and was cut at the
    byte-identical place, three minutes later. The retry cannot succeed, so it
    is not attempted."""
    cut_off = '{"hallucinations": [{"claim": "a", "issue": "cut off mid-str'
    model = FakeChatModel([(cut_off, {"finish_reason": "length"})])

    with pytest.raises(LLMJSONError) as exc_info:
        invoke_json(model, "sys", "user")

    assert len(model.calls) == 1
    assert exc_info.value.truncated is True
    assert exc_info.value.text == cut_off


def test_invoke_json_still_repairs_a_response_that_is_merely_malformed():
    # The completion finished; it was just bad JSON. That is what the repair
    # turn has always been for, and a stop_reason check must not disable it.
    model = FakeChatModel([("{oops not json}", {"finish_reason": "stop"}), '{"ok": true}'])

    assert invoke_json(model, "sys", "user") == {"ok": True}
    assert len(model.calls) == 2


def test_invoke_json_error_carries_the_text_a_failed_repair_returned():
    model = FakeChatModel(["{oops", "{still oops"])

    with pytest.raises(LLMJSONError) as exc_info:
        invoke_json(model, "sys", "user")

    assert exc_info.value.text == "{still oops"
    assert exc_info.value.truncated is False


# -- salvage_json_list -------------------------------------------------------------------


def test_salvage_json_list_keeps_every_entry_before_the_cut():
    text = (
        '{\n  "hallucinations": [\n'
        '    {"claim": "a", "issue": "x", "grounded": false},\n'
        '    {"claim": "b", "issue": "y", "nested": {"k": [1, 2]}},\n'
        '    {"claim": "c", "issue": "cut off mid-str'
    )
    salvaged = salvage_json_list(text, "hallucinations")
    assert [entry["claim"] for entry in salvaged] == ["a", "b"]


def test_salvage_json_list_reads_a_complete_array_too():
    assert salvage_json_list('{"framing_issues": [{"hypothesis_id": "H1"}]}', "framing_issues") == [{"hypothesis_id": "H1"}]


def test_salvage_json_list_returns_nothing_for_a_missing_or_empty_key():
    text = '{"hallucinations": []}'
    assert salvage_json_list(text, "hallucinations") == []
    assert salvage_json_list(text, "framing_issues") == []
    assert salvage_json_list("not json at all", "hallucinations") == []


def test_salvage_json_list_stops_at_the_first_entry_it_cannot_read():
    # Everything after the damage is unreliable, so nothing past it is taken —
    # a salvage that guesses is worse than one that stops.
    text = '{"hallucinations": [{"claim": "a"}, {"claim": broken}, {"claim": "c"}]}'
    assert salvage_json_list(text, "hallucinations") == [{"claim": "a"}]
