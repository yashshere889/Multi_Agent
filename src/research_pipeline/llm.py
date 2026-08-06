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


def _openai_chat_model(temperature: float, thinking: bool) -> BaseChatModel:
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
    )


def _huggingface_chat_model(temperature: float, thinking: bool) -> BaseChatModel:
    # Imported lazily: langchain-huggingface pulls in torch/transformers, which
    # is several GB and thousands of inodes. It's an optional extra
    # (`uv sync --extra huggingface`) so the default openai backend stays light.
    try:
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

    llm = HuggingFacePipeline.from_model_id(
        model_id=settings.llm_model,
        task="text-generation",
        pipeline_kwargs=pipeline_kwargs,
        model_kwargs={
            "device_map": settings.llm_hf_device_map,
            "dtype": settings.llm_hf_dtype,
            "trust_remote_code": True,
        },
    )
    return ChatHuggingFace(llm=llm)


def get_chat_model(
    temperature: Optional[float] = None,
    *,
    enable_thinking: Optional[bool] = None,
) -> BaseChatModel:
    """The one place a chat model is constructed. Callers pass `temperature`
    to trade determinism against variety per agent; `enable_thinking`
    overrides LLM_ENABLE_THINKING for an agent that genuinely wants the model
    to deliberate (nothing does today)."""
    thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking
    resolved_temperature = settings.llm_temperature if temperature is None else temperature

    if settings.llm_backend == "huggingface":
        return _huggingface_chat_model(resolved_temperature, thinking)
    return _openai_chat_model(resolved_temperature, thinking)
