"""Factory for the shared chat model. Two backends, selected by LLM_BACKEND:

- **openai** (default): any OpenAI-compatible server reached over LLM_BASE_URL
  — vLLM on a Barkla GPU node, LM Studio, llama-server, a local box.
  See scripts/slurm/run_llm_server.sbatch.
- **huggingface**: the model is loaded *in-process* with transformers via
  langchain-huggingface. No server, no port, no health check. On Barkla this
  is the path that matches the cluster's own `nemotron/nano-12b-v2` module,
  which ships exactly this stack with the weights pre-cached.

Both return a BaseChatModel, so no agent knows or cares which is in use.

The pipeline targets NVIDIA Nemotron
(https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), a
*reasoning* model: left alone it emits a <think>...</think> trace before its
answer, which would land in `response.content` and break every agent's JSON
parse. Two independent guards handle that, because either one alone can be
defeated by a misconfiguration:

1. `chat_template_kwargs.enable_thinking` is sent on every request, so the chat
   template doesn't open a reasoning turn in the first place.
2. In llm_json.strip_reasoning: any trace that shows up anyway is stripped
   before parsing.

Guard 1 is an OpenAI *wire protocol* field and therefore exists only on the
openai backend — there is no request body to attach it to when the model runs
in-process, and langchain-huggingface offers no hook into the tokenizer's
apply_chat_template call. Under the huggingface backend guard 2 is thus the
only thing standing between a reasoning trace and a failed parse, which is
precisely the case it was written for. LLM_ENABLE_THINKING is honoured where
it can be, and logged and ignored where it can't.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.config import settings

logger = logging.getLogger(__name__)

# get_chat_model() is called independently by every agent (hypothesis,
# experiment_planner, coder, writer, reviewer, literature) — see each
# agent's __init__. For the openai backend that's cheap (just a new HTTP
# client), but for huggingface it used to mean re-running
# AutoModelForCausalLM.from_pretrained and reloading the whole checkpoint
# onto the GPU on every single call. The first instance's memory was still
# resident when the second load started, so accelerate silently offloaded
# part of the second model onto the CPU — and that split CPU/GPU placement
# disables Mamba's fused CUDA kernel path, falling back to the
# quadratic-memory pure-PyTorch implementation, which then OOMs on a single
# batch. Loading once and reusing the same model/tokenizer for every agent
# fixes all three problems at once.
_hf_model_and_tokenizer: Optional[tuple] = None
_hf_model_lock = threading.Lock()


def _extra_body(enable_thinking: bool) -> dict:
    body: dict = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    # Nemotron's own knob for capping how many tokens it spends deliberating;
    # meaningless (and omitted) when thinking is off.
    if enable_thinking and settings.llm_reasoning_budget is not None:
        body["reasoning_budget"] = settings.llm_reasoning_budget
    return body


def _openai_chat_model(temperature: float, thinking: bool, streaming: bool) -> BaseChatModel:
    # Imported here rather than at module scope purely for symmetry with the
    # huggingface branch, whose import is genuinely expensive.
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


def _load_hf_model_and_tokenizer() -> tuple:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading %s onto the GPU (once per process)...", settings.llm_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model,
        device_map=settings.llm_hf_device_map,
        dtype=settings.llm_hf_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.llm_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _cached_hf_model_and_tokenizer() -> tuple:
    global _hf_model_and_tokenizer
    if _hf_model_and_tokenizer is None:
        with _hf_model_lock:
            if _hf_model_and_tokenizer is None:
                _hf_model_and_tokenizer = _load_hf_model_and_tokenizer()
    return _hf_model_and_tokenizer


def _huggingface_chat_model(temperature: float, thinking: bool) -> BaseChatModel:
    # Imported lazily: langchain-huggingface pulls in torch/transformers, which
    # is several GB and thousands of inodes. It's an optional extra
    # (`uv sync --extra huggingface`) so the default openai backend stays light.
    try:
        from transformers import pipeline as transformers_pipeline

        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "LLM_BACKEND=huggingface requires the 'huggingface' extra: "
            "`uv sync --extra huggingface`."
        ) from exc

    if thinking:
        logger.warning(
            "LLM_ENABLE_THINKING is set, but the huggingface backend has no way to "
            "forward chat_template_kwargs.enable_thinking, so the model decides for "
            "itself. Any reasoning trace is still stripped by llm_json.strip_reasoning."
        )

    # do_sample gates sampling entirely: with the transformers default of False
    # the model decodes greedily and temperature/top_p are silently ignored —
    # so passing them alongside do_sample=False would look configured while
    # doing nothing. temperature=0 is the one case where greedy is what was
    # actually asked for.
    do_sample = temperature > 0

    pipeline_kwargs = {
        # max_new_tokens, not max_tokens: transformers counts *generated*
        # tokens, whereas the OpenAI field bounds prompt+completion together.
        "max_new_tokens": settings.llm_max_tokens,
        "do_sample": do_sample,
        # Without this the pipeline echoes the whole prompt back before the
        # answer, and every strip_fences/json.loads downstream would choke.
        "return_full_text": False,
    }
    if do_sample:
        pipeline_kwargs["temperature"] = temperature
        pipeline_kwargs["top_p"] = settings.llm_top_p

    model, tokenizer = _cached_hf_model_and_tokenizer()
    # Building a fresh transformers pipeline around the cached model/tokenizer
    # is cheap (no weights are loaded) — it's just wiring in this call's own
    # generation kwargs (temperature/do_sample differ per agent).
    text_generation_pipeline = transformers_pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs,
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return ChatHuggingFace(llm=llm)


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

    `streaming` (openai backend only) is opt-in, not a blanket default: a
    quick tunnel's edge proxy 524s a request that sends zero bytes back
    within ~100s, which a large non-streamed completion can exceed — the
    Coder Agent's generated code routinely runs 1000s of tokens, and it calls
    sequentially, so streaming there only ever helps. But experiment_planner
    (and any other agent that fans out concurrent calls, by design — see
    CLAUDE.md) fires several simultaneous requests, and several concurrent
    long-lived SSE streams through a free quick tunnel fail immediately with
    a connection error where the equivalent non-streamed calls succeed.
    invoke() aggregates a stream into one AIMessage either way, so nothing
    downstream needs to know which mode is in effect."""
    thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking
    resolved_temperature = settings.llm_temperature if temperature is None else temperature

    if settings.llm_backend == "huggingface":
        return _huggingface_chat_model(resolved_temperature, thinking)
    return _openai_chat_model(resolved_temperature, thinking, streaming)
