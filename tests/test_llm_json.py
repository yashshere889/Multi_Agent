from dataclasses import replace
from types import SimpleNamespace

import pytest

from research_pipeline import config, llm
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


@pytest.fixture
def openai_backend(monkeypatch):
    """Pins the openai backend so these assertions don't depend on whatever
    LLM_BACKEND the developer's ambient .env happens to set — under
    huggingface every one of them would be constructing a real model."""
    monkeypatch.setattr(llm, "settings", replace(llm.settings, llm_backend="openai"))


def test_get_chat_model_sends_the_configured_thinking_mode(monkeypatch, openai_backend):
    # Pinned rather than read from the ambient .env, so the assertion doesn't
    # flip with whoever's LLM_ENABLE_THINKING is loaded.
    monkeypatch.setattr(
        llm, "settings", replace(llm.settings, llm_backend="openai", llm_enable_thinking=False)
    )
    assert llm.get_chat_model().extra_body == {"chat_template_kwargs": {"enable_thinking": False}}

    monkeypatch.setattr(
        llm, "settings", replace(llm.settings, llm_backend="openai", llm_enable_thinking=True)
    )
    assert llm.get_chat_model().extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_get_chat_model_sends_reasoning_budget_only_when_thinking_is_on(monkeypatch):
    # Settings is a frozen dataclass, so swap in a whole replaced copy.
    monkeypatch.setattr(
        llm, "settings", replace(llm.settings, llm_backend="openai", llm_reasoning_budget=512)
    )

    assert "reasoning_budget" not in llm.get_chat_model(enable_thinking=False).extra_body
    thinking = llm.get_chat_model(enable_thinking=True)
    assert thinking.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_budget": 512,
    }


def test_get_chat_model_applies_configured_sampling_and_token_budget(openai_backend):
    model = llm.get_chat_model(temperature=0.1)
    assert model.temperature == 0.1
    assert model.top_p == llm.settings.llm_top_p
    assert model.max_tokens == llm.settings.llm_max_tokens


def test_get_chat_model_uses_the_configured_model_id(openai_backend):
    assert llm.get_chat_model().model_name == llm.settings.llm_model


# -- llm.get_chat_model: huggingface backend ---------------------------------------------


@pytest.fixture
def fake_huggingface(monkeypatch):
    """Stands in for transformers and langchain_huggingface, which are an
    optional extra and would otherwise download and load a real 12-30B model.
    Records the kwargs the factory builds so the transformers-specific
    translation can be checked without any of it installed.

    Also resets llm's module-level model/tokenizer cache (see llm.py's
    _cached_hf_model_and_tokenizer): that cache is what makes every agent
    reuse one loaded model instead of each reloading its own, so it must
    start empty each test or a later test would silently skip
    from_pretrained and see a previous test's recorded kwargs."""
    recorded = {}

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            recorded["model_id"] = model_id
            recorded["model_kwargs"] = kwargs
            return "fake-model"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return SimpleNamespace(pad_token_id="fake-pad-token", eos_token_id="fake-eos-token")

    def fake_pipeline(*, task, model, tokenizer, **pipeline_kwargs):
        recorded["task"] = task
        recorded["pipeline_kwargs"] = pipeline_kwargs
        return "fake-pipeline"

    class FakeHuggingFacePipeline:
        def __init__(self, pipeline):
            self.pipeline = pipeline

    class FakeChatHuggingFace:
        def __init__(self, llm):
            self.llm = llm

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            pipeline=fake_pipeline,
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "langchain_huggingface",
        SimpleNamespace(
            ChatHuggingFace=FakeChatHuggingFace,
            HuggingFacePipeline=FakeHuggingFacePipeline,
        ),
    )
    monkeypatch.setattr(llm, "settings", replace(llm.settings, llm_backend="huggingface"))
    monkeypatch.setattr(llm, "_hf_model_and_tokenizer", None)
    return recorded


def test_huggingface_backend_is_selected_by_the_configured_backend(fake_huggingface):
    model = llm.get_chat_model()
    assert type(model).__name__ == "FakeChatHuggingFace"
    assert fake_huggingface["model_id"] == llm.settings.llm_model
    assert fake_huggingface["task"] == "text-generation"


def test_huggingface_backend_enables_sampling_only_above_zero_temperature(fake_huggingface):
    # do_sample=False makes transformers decode greedily and ignore temperature
    # / top_p outright, so passing them together would look configured while
    # doing nothing. Above zero they must travel together.
    llm.get_chat_model(temperature=0.2)
    assert fake_huggingface["pipeline_kwargs"]["do_sample"] is True
    assert fake_huggingface["pipeline_kwargs"]["temperature"] == 0.2
    assert fake_huggingface["pipeline_kwargs"]["top_p"] == llm.settings.llm_top_p


def test_huggingface_backend_uses_greedy_decoding_at_zero_temperature(fake_huggingface):
    llm.get_chat_model(temperature=0.0)
    kwargs = fake_huggingface["pipeline_kwargs"]
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_huggingface_backend_translates_the_token_budget_and_suppresses_the_prompt_echo(
    fake_huggingface,
):
    # max_tokens (prompt+completion) has no transformers equivalent;
    # max_new_tokens bounds generation only. return_full_text=False matters
    # just as much: the pipeline otherwise prepends the whole prompt, which
    # would break every strip_fences/json.loads downstream.
    llm.get_chat_model()
    kwargs = fake_huggingface["pipeline_kwargs"]
    assert kwargs["max_new_tokens"] == llm.settings.llm_max_tokens
    assert kwargs["return_full_text"] is False


def test_unknown_backend_is_rejected_at_load_time(monkeypatch):
    # Validated when settings load rather than when a model is first built:
    # otherwise a typo'd backend surfaces dozens of LLM calls into a run, on
    # whichever agent happened to construct a model first.
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        config._env_backend()


@pytest.mark.parametrize(
    "raw,expected",
    [("", "openai"), ("  ", "openai"), ("HuggingFace", "huggingface"), (" openai ", "openai")],
)
def test_backend_is_normalized_and_defaults_to_openai(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_BACKEND", raw)
    assert config._env_backend() == expected


def test_huggingface_backend_ignores_thinking_rather_than_failing(fake_huggingface, caplog):
    # enable_thinking rides on an OpenAI wire field that doesn't exist here.
    # Turning it on must degrade to a warning (strip_reasoning still guards
    # the parse), never an error.
    with caplog.at_level("WARNING"):
        llm.get_chat_model(enable_thinking=True)
    assert "enable_thinking" in caplog.text
