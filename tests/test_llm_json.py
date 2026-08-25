from dataclasses import replace
from types import SimpleNamespace

import pytest

from research_pipeline import llm, llm_json
from research_pipeline.llm_json import LLMJSONError, invoke_json, strip_fences, strip_reasoning


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


# -- repair_keys -------------------------------------------------------------

HYPOTHESIS_KEYS = [
    "id",
    "statement",
    "rationale",
    "related_gaps",
    "related_methods",
    "suggested_variables",
]


def test_a_doubled_syllable_is_repaired():
    """The Barkla job 10334292 regression: the model wrote `rationationale`
    ("ration" + "ationale") on two of three hypotheses. The rationales were
    complete; only the key was mangled, and the whole question was lost."""
    repaired = llm_json.repair_keys(
        {"id": "H2", "statement": "s", "rationationale": "the real rationale text"},
        HYPOTHESIS_KEYS,
    )

    assert repaired["rationale"] == "the real rationale text"
    assert "rationationale" not in repaired


def test_a_wholly_doubled_key_is_repaired():
    repaired = llm_json.repair_keys({"statementstatement": "s"}, HYPOTHESIS_KEYS)

    assert repaired == {"statement": "s"}


def test_an_ordinary_typo_is_repaired():
    assert llm_json.repair_keys({"statment": "s"}, HYPOTHESIS_KEYS) == {"statement": "s"}
    assert llm_json.repair_keys({"ratonale": "r"}, HYPOTHESIS_KEYS) == {"rationale": "r"}


def test_an_unrelated_extra_field_is_left_alone():
    # Not forced onto whichever expected name it happens to be nearest.
    payload = {"id": "H1", "notes": "n", "justification": "j"}

    assert llm_json.repair_keys(payload, HYPOTHESIS_KEYS) == payload


def test_a_complete_payload_is_returned_unchanged():
    payload = {key: "v" for key in HYPOTHESIS_KEYS}

    assert llm_json.repair_keys(payload, HYPOTHESIS_KEYS) == payload


def test_the_input_is_not_mutated():
    payload = {"id": "H1", "rationationale": "r"}
    llm_json.repair_keys(payload, HYPOTHESIS_KEYS)

    assert "rationationale" in payload, "repair_keys must return a copy"


def test_an_ambiguous_rename_is_declined():
    # Two unexpected keys equally near one missing name is a response malformed
    # in a way renaming cannot settle — validation should say so, not this.
    repaired = llm_json.repair_keys({"statment": "a", "statemnt": "b"}, HYPOTHESIS_KEYS)

    assert "statement" not in repaired


def test_a_present_field_is_never_overwritten():
    # `rationale` is already there, so the doubled key is not a repair target.
    payload = {"rationale": "the good one", "rationationale": "the mangled one"}

    assert llm_json.repair_keys(payload, HYPOTHESIS_KEYS)["rationale"] == "the good one"


def test_a_non_dict_payload_passes_through():
    assert llm_json.repair_keys("not a dict", HYPOTHESIS_KEYS) == "not a dict"
