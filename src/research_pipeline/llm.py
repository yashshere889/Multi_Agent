"""Factory for the shared chat model. Single backend: any OpenAI-compatible
server reached over LLM_BASE_URL — vLLM on a Barkla GPU node, LM Studio,
llama-server, a local box. See scripts/slurm/run_llm_server.sbatch. Every
agent calls get_chat_model() and gets back a BaseChatModel; no agent
constructs its own client.

The pipeline targets NVIDIA Nemotron 3 Nano
(https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), a
*reasoning* model: left alone it emits a <think>...</think> trace before its
answer, which would land in `response.content` and break every agent's JSON
parse. Two independent guards handle that, because either one alone can be
defeated by a misconfiguration:

1. `chat_template_kwargs.enable_thinking` is sent on every request, so the chat
   template doesn't open a reasoning turn in the first place.
2. In llm_json.strip_reasoning: any trace that shows up anyway is stripped
   before parsing.

Guard 2 is what actually saves a run when guard 1 is defeated — e.g. a vLLM
server started without the `nano_v3` reasoning parser leaves the trace in
`content` regardless of what was requested.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.config import settings

logger = logging.getLogger(__name__)


def _extra_body(enable_thinking: bool) -> dict:
    body: dict = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    # Nemotron's own knob for capping how many tokens it spends deliberating;
    # meaningless (and omitted) when thinking is off.
    if enable_thinking and settings.llm_reasoning_budget is not None:
        body["reasoning_budget"] = settings.llm_reasoning_budget
    return body


def _openai_chat_model(temperature: float, thinking: bool, streaming: bool) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
        extra_body=_extra_body(thinking),
        # The default of 2 is tuned for a stable local/cluster endpoint. Some
        # deployments of this backend (e.g. a Cloudflare quick tunnel fronting
        # a Kaggle notebook — see scripts/kaggle/tunnel.py) are inherently
        # flaky infrastructure, and a bare APIConnectionError here has no
        # caller-side retry: it aborts the whole orchestrator run. A higher
        # ceiling costs nothing on a healthy endpoint.
        max_retries=6,
        timeout=180,
        streaming=streaming,
    )


def get_chat_model(
    temperature: Optional[float] = None,
    *,
    enable_thinking: Optional[bool] = None,
    streaming: bool = False,
) -> BaseChatModel:
    """The one place a chat model is constructed. Callers pass `temperature`
    to trade determinism against variety per agent; `enable_thinking`
    overrides LLM_ENABLE_THINKING for an agent that genuinely wants the model
    to deliberate (nothing does today).

    `streaming` is opt-in, not a blanket default: a quick tunnel's edge proxy
    524s a request that sends zero bytes back within ~100s, which a large
    non-streamed completion can exceed — the Coder Agent's generated code
    routinely runs 1000s of tokens, and it calls sequentially, so streaming
    there only ever helps. But experiment_planner (and any other agent that
    fans out concurrent calls, by design — see CLAUDE.md) fires several
    simultaneous requests, and several concurrent long-lived SSE streams
    through a free quick tunnel fail immediately with a connection error
    where the equivalent non-streamed calls succeed. invoke() aggregates a
    stream into one AIMessage either way, so nothing downstream needs to know
    which mode is in effect.

    Every agent shares one LLM_BASE_URL/LLM_MODEL — there is no per-agent
    override anymore; all seven agents run against the same vLLM-served
    Nemotron 3 Nano 30B endpoint."""
    thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking
    resolved_temperature = settings.llm_temperature if temperature is None else temperature
    return _openai_chat_model(resolved_temperature, thinking, streaming)
