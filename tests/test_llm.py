import dataclasses
import sys

import pytest

from research_pipeline import llm


def _patch_settings(monkeypatch, **overrides):
    """Settings is a frozen dataclass, so swap in a copy rather than
    mutating — mirrors tests/test_coder_agent.py's helper of the same name."""
    monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, **overrides))


def test_get_chat_model_openai_backend_uses_default_model(monkeypatch):
    _patch_settings(monkeypatch, llm_backend="openai", llm_model="default-model")
    model = llm.get_chat_model(temperature=0.1)
    assert model.model_name == "default-model"


def test_get_chat_model_model_override_is_per_call_only(monkeypatch):
    # Confirms the override doesn't leak into a later unrelated call — every
    # other agent must keep getting the pipeline-wide default.
    _patch_settings(monkeypatch, llm_backend="openai", llm_model="default-model")

    overridden = llm.get_chat_model(temperature=0.1, backend="openai", model="qwen3-8b")
    default = llm.get_chat_model(temperature=0.1)

    assert overridden.model_name == "qwen3-8b"
    assert default.model_name == "default-model"


def test_get_chat_model_max_tokens_override_is_per_call_only(monkeypatch):
    _patch_settings(monkeypatch, llm_backend="openai", llm_model="m", llm_max_tokens=8192)

    overridden = llm.get_chat_model(temperature=0.1, max_tokens=2048)
    default = llm.get_chat_model(temperature=0.1)

    assert overridden.max_tokens == 2048
    assert default.max_tokens == 8192


def test_check_min_gpus_is_a_noop_at_the_default(monkeypatch):
    # LLM_HF_MIN_GPUS defaults to 1 — must return without even importing
    # torch, so the default openai-only install (no 'huggingface' extra) is
    # never affected by this check.
    _patch_settings(monkeypatch, llm_hf_min_gpus=1)
    llm._check_min_gpus("some/model")  # does not raise, does not import torch


def test_check_min_gpus_raises_a_clear_error_when_torch_reports_too_few(monkeypatch):
    _patch_settings(monkeypatch, llm_hf_min_gpus=2)

    class _FakeCuda:
        @staticmethod
        def device_count():
            return 1

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    with pytest.raises(RuntimeError, match="LLM_HF_MIN_GPUS=2"):
        llm._check_min_gpus("Qwen/Qwen3-8B")
