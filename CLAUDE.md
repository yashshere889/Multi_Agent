# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-agent research pipeline: Literature → Hypothesis → Experiment Planner → Coder → Writer ⇄ Reviewer. Each stage searches papers, synthesizes hypotheses, plans experiments, generates+runs code, drafts a NeurIPS-style paper PDF, and reviews it against upstream ground truth. Built on [LangGraph](https://github.com/langchain-ai/langgraph) for the one agent that genuinely needs graph fan-out; every other agent is a plain Python callable.

Read [README.md](README.md) first — it documents the full architecture, per-agent I/O contracts, and CLI usage in detail. Don't duplicate that content in explanations; point there instead.

## Commands

```bash
uv sync                    # install deps
cp .env.example .env       # then fill in LLM_BASE_URL / SEMANTIC_SCHOLAR_API_KEY

uv run pytest                          # run all tests
uv run pytest tests/test_hypothesis_agent.py            # single file
uv run pytest tests/test_hypothesis_agent.py::test_name # single test

uv run research-pipeline <agent> ...   # run a pipeline stage, e.g.:
uv run research-pipeline literature "research question" --max-results 5 --download-dir papers
```

There is no configured linter/formatter (no ruff config in `pyproject.toml`, ruff isn't a declared dependency) — don't assume `ruff check` works.

## Architecture

Six agents live under `src/research_pipeline/agents/<name>/`, each self-contained with its own `schema.py` (output contract + `validate_output()`), `prompts.py`, and `<name>_agent.py` (class + `run_<name>_agent()` entry point). Non-LLM logic (HTTP clients, batching, sandboxing, PDF I/O, citation resolution) is factored into separate modules within each agent package specifically so it's unit-testable without hitting an LLM.

Two structural patterns, both valid — pick whichever fits new work:
- **Graph agent** (`literature/`): `state.py` + `nodes.py` + `graph.py`, used only where work genuinely branches/parallelizes (arXiv + Semantic Scholar queried concurrently).
- **Callable agent** (`hypothesis/`, `experiment_planner/`, `coder/`, `writer/`, `reviewer/`): a class with `.run(...)` plus a module-level `run_<name>_agent()` function. Use this for anything that's fundamentally a sequence (map-then-reduce, section-by-section drafting) — simpler to read/test/call than a graph.

Agents chain by **data shape, not by a shared orchestrator**: each `run_<name>_agent()` takes a plain dict/list shaped like the *previous agent's output* (not internal state), and validates it against that upstream agent's own `validate_output()` before doing anything else. This means every link works identically in-process or decoupled via disk (read a previous run's JSON straight from `outputs/`). See README's "Chaining agents individually" section for the full code example.

Two orchestrators sit above the agents, neither an agent itself:
- **Writer/Reviewer loop** (`writer_reviewer_loop.py`): draft → review → revise → re-review, until the Reviewer reports zero issues and every quality score clears the threshold, or `max_iterations` is hit. It keeps every iteration's PDF and review for traceability, not just the final one.
- **Pipeline orchestrator** (`orchestrator/`, Graph-agent pattern): runs all six agents end to end as one LangGraph, with the Writer/Reviewer cycle as a conditional edge routing on `overall_pass`. Its nodes are thin wrappers over the same `run_<name>_agent()` entry points — no agent logic is reimplemented there, and it reuses the loop's `route_feedback_to_sections`/`_consolidate_unresolved` rather than duplicating them. Non-serializable dependencies (preconfigured `writer`/`reviewer` agents) go in `config["configurable"]`, never in state, which is checkpointed.

Key shared modules at the top level:
- `config.py` — all env-var settings (LLM endpoint, API keys, per-agent output dirs/defaults), loaded via `.env`.
- `llm.py` — single `get_chat_model()` factory used by every LLM-calling agent; swapping providers (e.g. to `langchain_anthropic.ChatAnthropic`) happens here only.
- `llm_json.py` — shared "invoke model, parse JSON, retry once with a repair prompt on bad JSON" helper, reused by any agent that wants structured output.
- `cli.py` — argparse entry point; adding a new agent means adding a subparser here following an existing one.

### Design principles specific to this codebase

- **Determinism over model judgment wherever verifiable.** Hypothesis outcomes (supported/refuted/inconclusive) are computed in Python from the Coder Agent's own `meets_success_criteria`/`status` fields, never left for the model to decide. Citation validity is enforced by resolving `[[cite:PAPER_ID]]`/`[[citet:PAPER_ID]]` markers only against papers the Literature Agent actually found — any other id is stripped and flagged, never printed. The Reviewer re-derives its own ground truth independently rather than trusting the Writer's summary, and checks citation/results/coverage deterministically (regex/dict-diff, no LLM) — only hallucination nuance and framing/tone honesty go to the LLM.
- **The Coder Agent never submits SLURM jobs itself.** `high`-complexity or GPU-needing plans get generated code plus a `run.sbatch` template for a human to review and submit; only `low`/`medium` complexity plans without a GPU requirement are executed synchronously (with a bounded timeout — see `sandbox.py:TIMEOUT_SECONDS_BY_COMPLEXITY`), and only after runtime probing of network/GPU availability (never hardcoded assumptions, since the same code runs both locally and on a Barkla compute node).
- **Template splicing, not `str.format()`.** The Coder Agent's `run.py.template` is filled via plain string replacement because LLM-generated code routinely contains literal `{`/`}` that would break format-string substitution.
- **One model, one factory, provider-agnostic transport.** Every agent is LLM-powered and every one of them goes through `llm.get_chat_model()`, which targets [NVIDIA Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) over a plain OpenAI-compatible endpoint (`LLM_BASE_URL`) — no agent constructs its own client, and nothing assumes llama.cpp/Kaggle/a specific provider. Nemotron is a reasoning model, so `<think>` traces are guarded twice: `chat_template_kwargs.enable_thinking` is sent on every request (`LLM_ENABLE_THINKING`, default off), *and* `llm_json.strip_reasoning` strips any trace that arrives anyway — a server started without vLLM's `nano_v3` reasoning parser leaves the trace in `content`, which would break every JSON parse. Anything parsing raw model output must go through `strip_fences`/`invoke_json` rather than doing its own fence regex.

### Adding a new agent

Copy either structural pattern above into `agents/<name>/`, then register a subcommand in `cli.py`. If it needs LLM calls returning structured JSON, reuse `llm_json.py` rather than reimplementing retry/repair logic.
