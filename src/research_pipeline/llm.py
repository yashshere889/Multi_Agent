"""Factory for the shared chat model, pointed at whatever OpenAI-compatible
server is configured via LLM_BASE_URL (local box, SSH-tunneled Barkla node, etc.).

The pipeline targets NVIDIA Nemotron 3 Nano
(https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), served
through vLLM's OpenAI-compatible endpoint — see scripts/slurm/run_llm_server.sbatch.
Nemotron is a *reasoning* model: left alone it emits a <think>...</think> trace
before its answer, which would land in `response.content` and break every
agent's JSON parse on any server started without the reasoning parser plugin.
Two independent guards handle that, because either one alone can be defeated
by a server misconfiguration:

1. Here: `chat_template_kwargs.enable_thinking` is sent on every request, so
   the chat template doesn't open a reasoning turn in the first place.
2. In llm_json.strip_reasoning: any trace that shows up anyway is stripped
   before parsing.

Everything below is still plain OpenAI-compatible request fields, so pointing
LLM_BASE_URL/LLM_MODEL at a non-Nemotron server keeps working — servers that
don't know `chat_template_kwargs` ignore it.
"""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from research_pipeline.config import settings


def _extra_body(enable_thinking: bool) -> dict:
    body: dict = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    # Nemotron's own knob for capping how many tokens it spends deliberating;
    # meaningless (and omitted) when thinking is off.
    if enable_thinking and settings.llm_reasoning_budget is not None:
        body["reasoning_budget"] = settings.llm_reasoning_budget
    return body


def get_chat_model(
    temperature: Optional[float] = None,
    *,
    enable_thinking: Optional[bool] = None,
) -> ChatOpenAI:
    """The one place a chat model is constructed. Callers pass `temperature`
    to trade determinism against variety per agent; `enable_thinking`
    overrides LLM_ENABLE_THINKING for an agent that genuinely wants Nemotron
    to deliberate (nothing does today)."""
    thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        extra_body=_extra_body(thinking),
    )
