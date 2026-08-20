"""Factory for the shared chat model. Single backend: any OpenAI-compatible
server reached over LLM_BASE_URL — vLLM on a Barkla GPU node, LM Studio,
llama-server, a local box. See scripts/slurm/run_llm_server.sbatch. Every
agent calls get_chat_model() and gets back a BaseChatModel; no agent
constructs its own client.

The pipeline targets Qwen3-Coder 30B-A3B-Instruct
(https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct), a sparse-MoE
coding model with ~3B active parameters per token. It replaced NVIDIA Nemotron
3 Nano, and the two differ in one way that matters here: Qwen3-Coder-Instruct
is *not* a reasoning model. It emits no <think> trace and its chat template
has no `enable_thinking` kwarg, so sending one is at best ignored and at worst
a template error.

Hence `_extra_body` sends `chat_template_kwargs` only when thinking is actually
requested, which by default it is not. The setting survives rather than being
deleted so that pointing LLM_MODEL back at a reasoning model needs no code
change, and llm_json.strip_reasoning still strips any trace that arrives —
free insurance, and the guard that saves a run when a server is started
without the matching reasoning parser.

Sampling: `temperature` and `top_p` are sent explicitly and win. Qwen3-Coder's
own generation config additionally sets top_k=20 and repetition_penalty=1.05,
which this pipeline does not override — both are reasonable for code, and vLLM
logs that it is applying them at startup.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.config import settings

logger = logging.getLogger(__name__)


def _extra_body(enable_thinking: bool) -> dict:
    """Provider-specific request fields, sent only when they mean something.

    Nothing is sent in the default configuration. `enable_thinking` is a chat
    template kwarg that exists on reasoning models (Nemotron 3 Nano) and not on
    the coding model this pipeline now serves; passing `enable_thinking=false`
    to a template with no such variable is a request field with no reader, and
    on some servers a template render error rather than a harmless no-op.

    So the kwarg goes out only when thinking is actually being asked for, which
    keeps the reasoning-model path working unchanged without putting a dead
    field on every request in the common case.
    """
    if not enable_thinking:
        return {}
    body: dict = {"chat_template_kwargs": {"enable_thinking": True}}
    # A reasoning model's own knob for capping how many tokens it spends
    # deliberating; meaningless (and omitted) when thinking is off.
    if settings.llm_reasoning_budget is not None:
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
    Qwen3-Coder 30B-A3B endpoint."""
    thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking
    resolved_temperature = settings.llm_temperature if temperature is None else temperature
    return _openai_chat_model(resolved_temperature, thinking, streaming)
