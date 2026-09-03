# research-pipeline

A multi-agent research pipeline. Currently ships seven agents, each usable
standalone and chained by data shape rather than by coupling to each other
(see "Chaining agents individually" below); a LangGraph orchestrator runs all
seven end to end in one call (see "Running the whole pipeline"):

- **literature** — searches arXiv + Semantic Scholar + CORE, dedupes, downloads
  PDFs. Built as a [LangGraph](https://github.com/langchain-ai/langgraph)
  `StateGraph` because it genuinely benefits from graph fan-out/fan-in (three
  independent HTTP APIs queried in parallel).
- **interdisciplinary-literature** — takes the literature agent's papers,
  identifies up to `INTERDISCIPLINARY_MAX_FIELDS` *adjacent* fields whose
  methods could inform the same problem, searches each of them with the same
  arXiv/Semantic Scholar/CORE clients (one `Send` branch per field), and returns the
  merged, deduped paper pool plus **bridge insights** — concrete "this method
  from field X could inform this problem because Y" entries, each citing the
  cross-field papers it came from. Its `papers` key is the same shape the
  hypothesis agent already consumes, so it drops into the chain without either
  neighbour changing. Deduplication uses the literature agent's own
  doi/normalized-title key (not model judgment), and a `supporting_paper_ids`
  entry naming a paper that isn't in the merged pool is dropped and logged
  rather than passed downstream. See its docstring
  ([interdisciplinary_literature_agent.py](src/research_pipeline/agents/interdisciplinary_literature/interdisciplinary_literature_agent.py))
  for the full input/output contract.
- **hypothesis** — takes the upstream papers (from either literature agent),
  synthesizes the body of literature, produces exactly 3 grounded, testable
  hypotheses as JSON, and **ranks** them against each other on feasibility /
  testability / grounding, naming a single `selected_hypothesis_id`. All 3 are
  always generated and always returned — ranking is additive and decides
  nothing on its own; the winner is derived in Python from whichever ranking
  entry holds rank 1, and the schema enforces that the ranks are a permutation
  of 1..3 over exactly the generated ids. When the interdisciplinary agent ran
  upstream, its bridge insights are shown to the model as explicitly-labelled
  cross-field inspiration for both the generation and the ranking call. See its
  docstring
  ([hypothesis_agent.py](src/research_pipeline/agents/hypothesis/hypothesis_agent.py))
  for the full input/output contract.
- **experiment-planner** — takes the hypothesis agent's hypotheses and turns
  each into an implementation-ready experiment plan (feasibility, design,
  data requirements, methods, evaluation, step-by-step implementation plan),
  plus shared-infrastructure notes and a priority order across them. An
  optional `hypothesis_ids` argument narrows *which* hypotheses get planned —
  the orchestrator passes the hypothesis agent's `selected_hypothesis_id`, so a
  full run executes and writes up one hypothesis rather than three. It changes
  nothing about the input contract: the complete 3-hypothesis output is still
  required and still validated in full. Also a
  plain callable, for the same reason as hypothesis: planning per hypothesis
  is independent work (run concurrently, one LLM call each) followed by one
  cross-cutting synthesis call — a graph wouldn't add anything a thread pool
  doesn't already give it. See its docstring
  ([experiment_planner_agent.py](src/research_pipeline/agents/experiment_planner/experiment_planner_agent.py))
  for the full input/output contract.
- **coder** — takes the experiment planner's plans and, per feasible plan
  (in `priority_order`, skipping `feasible: false` ones), generates a real
  runnable experiment (a single `run.py` rendered from a template, plus
  `README.md`/`requirements.txt`) and, when the estimated complexity and this
  environment's detected resources allow it, actually executes it and
  captures structured results. A plain callable like the other two, processing
  experiments
  **sequentially** rather than concurrently — unlike LLM-only synthesis
  calls, these calls execute real code that consumes real (possibly shared,
  possibly GPU) compute, so running them one at a time is the safer default,
  not a missed optimization. See its docstring
  ([coder_agent.py](src/research_pipeline/agents/coder/coder_agent.py)) for
  the full input/output contract and execution model.
- **writer** — takes the combined output of the agents above and drafts
  a full academic paper (PDF), styled after a NeurIPS submission — title,
  abstract, numbered sections (introduction, related work, hypotheses,
  per-experiment methods/results/discussion, limitations, future work), and a
  references list built only from papers the literature agent actually found.
  A plain callable, same reasoning as hypothesis/experiment-planner: it's a
  fixed sequence of section-drafting calls (some independent enough to run
  concurrently), not something that benefits from graph fan-out. Every claim
  a section makes is grounded in the specific upstream data handed to that
  section's prompt — hypothesis outcomes (supported/refuted/inconclusive) are
  computed deterministically in Python from the coder agent's own
  `meets_success_criteria`/`status` fields, never left to the model to judge,
  and in-text citations use `[[cite:PAPER_ID]]`/`[[citet:PAPER_ID]]` markers
  (NeurIPS's default author-year style, e.g. "(Smith et al., 2020)") that are
  mechanically resolved against — and rejected if not present in — the
  literature agent's actual papers, so a fabricated citation can never reach
  the printed PDF. See its docstring
  ([writer_agent.py](src/research_pipeline/agents/writer/writer_agent.py)) for
  the full input/output contract.
- **reviewer** — takes the Writer agent's paper (PDF + JSON summary) and
  checks it against the *same* upstream ground truth the writer used — not
  just internal consistency. Where a check can be done exactly, it is
  (agents.reviewer.checks, no LLM calls): every citation actually matches a
  real paper by surname+year, every reported metric's value appears
  (in some plausible textual form) in the Coder Agent's own data, no
  `skipped`/`code_generated_not_run` experiment's Results text reads as
  completed, every hypothesis has a Results/Discussion subsection, and a
  keyword pre-check flags the unambiguous cases where Discussion text
  contradicts the true supported/refuted/inconclusive verdict. What's left —
  hallucination nuance and honesty of framing/tone/flow — genuinely needs
  judgment, so that part *is* LLM-powered: one call per section plus one
  holistic quality-scoring call, each grounded in only the ground-truth slice
  relevant to what it's checking. A plain callable, same reasoning as the
  others. See its docstring
  ([reviewer_agent.py](src/research_pipeline/agents/reviewer/reviewer_agent.py))
  for the full input/output contract and exactly which checks are
  deterministic vs. LLM-based.

The **Writer/Reviewer feedback loop**
([writer_reviewer_loop.py](src/research_pipeline/writer_reviewer_loop.py)) is
a small orchestrator, not itself an agent: draft → review → if issues, revise
against the Reviewer's structured feedback → re-review → repeat, until the
Reviewer finds zero issues and every quality score clears the threshold, or
`max_iterations` is hit. See "Run" below for its CLI usage. The orchestrator
graph expresses that same loop as a conditional edge, and reuses this module's
feedback-routing logic rather than duplicating it.

All seven are valid patterns for a new agent — pick whichever fits the
agent's control flow, not necessarily the literature agent's graph shape.

## Layout

```
src/research_pipeline/
├── config.py              # env-var settings (LLM endpoint, API keys, defaults)
├── llm.py                 # shared ChatOpenAI factory
├── llm_json.py            # shared "call the model, expect JSON, retry once on bad JSON" helper
├── llm_sections.py        # same, for responses carrying source code: a delimited section format, no escaping
├── checkpointer.py        # shared LangGraph checkpointer factory (memory/sqlite/postgres) + shared node cache
├── cli.py                 # `research-pipeline <agent> ...` entry point
├── batch.py               # runs `orchestrate` over a file of questions unattended
├── writer_reviewer_loop.py # orchestrator: draft -> review -> revise -> re-review loop (not an agent itself)
├── orchestrator/          # LangGraph orchestrator running all seven agents end to end (not an agent itself)
│   ├── state.py            # PipelineState — one key per stage's full output dict
│   ├── nodes.py            # thin nodes wrapping each agent's existing entry point
│   └── graph.py            # StateGraph wiring, incl. the writer/reviewer cycle + compile()
├── webapp/                # optional web UI (`--extra webapp`); reads the graph, never modifies it
│   ├── events.py           # append-only events.jsonl — the runner's only channel to the server
│   ├── runs.py             # RunStore: run directories, run.json, launching + cancelling runners
│   ├── stages.py           # state deltas -> display rows (reuses should_continue_revising)
│   ├── runner.py           # the subprocess that streams one graph run
│   ├── app.py              # FastAPI routes; renders fragments the browser polls
│   ├── templates/          # Jinja2 pages + polled fragments
│   └── static/             # app.css and a ~60-line poller (no JS framework, no CDN)
└── agents/
    ├── literature/         # LangGraph StateGraph agent
    │   ├── state.py         # graph state schema (TypedDict)
    │   ├── clients.py        # arXiv / Semantic Scholar / CORE HTTP clients (no LangGraph coupling)
    │   ├── nodes.py          # LangGraph node functions
    │   └── graph.py          # StateGraph wiring + compile()
    ├── interdisciplinary_literature/  # LangGraph StateGraph agent
    │   ├── schema.py          # output contract (TypedDicts) + validate_output()
    │   ├── state.py           # graph state schema (TypedDict)
    │   ├── prompts.py         # prompt templates (adjacent-field identification, bridge synthesis)
    │   ├── graph.py           # StateGraph wiring — one Send branch per adjacent field + compile()
    │   └── interdisciplinary_literature_agent.py # InterdisciplinaryLiteratureAgent class + run_interdisciplinary_literature_agent() entry point
    ├── hypothesis/         # plain-callable agent
    │   ├── schema.py          # output contract (TypedDicts) + validate_output(), incl. the ranking/selection rules
    │   ├── papers.py          # paper normalization + batching (no LLM calls — unit-testable)
    │   ├── prompts.py         # prompt templates
    │   └── hypothesis_agent.py # HypothesisAgent class + run_hypothesis_agent() entry point
    ├── experiment_planner/  # plain-callable agent
    │   ├── schema.py          # output contract (TypedDicts) + validate_output()
    │   ├── prompts.py         # prompt templates (compute/data feasibility assumptions live here)
    │   └── experiment_planner_agent.py # ExperimentPlannerAgent class + run_experiment_planner_agent() entry point
    ├── coder/               # plain-callable agent
    │   ├── schema.py          # output contract (TypedDicts) + validate_output()
    │   ├── sandbox.py          # execution mechanics: network/GPU probing, isolated venvs, subprocess execution, syntax checking (no LLM calls — unit-testable)
    │   ├── prompts.py         # prompt templates (calling convention + results.json contract live here)
    │   └── coder_agent.py      # CoderAgent class + run_coder_agent() entry point
    ├── writer/              # plain-callable agent
    │   ├── schema.py          # output contract (JSON summary shape) + validate_output()
    │   ├── citations.py        # paper indexing (same id scheme as hypothesis/papers.py) + [[cite:ID]]/[[citet:ID]] marker resolution into NeurIPS-style author-year citations (no LLM calls — unit-testable)
    │   ├── pdf_builder.py       # reportlab PDF rendering — pure Python, no LibreOffice/Node/pandoc (no LLM calls — unit-testable)
    │   ├── pdf_reader.py        # reads this pipeline's own rendered PDFs back into section/subsection text (no LLM calls — unit-testable) — used by revise() and by the Reviewer Agent
    │   ├── prompts.py         # prompt templates (grounding + honesty rules live here)
    │   └── writer_agent.py     # WriterAgent class + run_writer_agent() entry point; .revise() for feedback-driven redrafts
    └── reviewer/            # plain-callable agent
        ├── schema.py          # output contract (TypedDicts) + validate_output()
        ├── checks.py          # deterministic citation/results-accuracy/hypothesis-coverage checks (no LLM calls — unit-testable)
        ├── prompts.py         # prompt templates for the LLM-based checks (hallucination nuance, framing honesty, quality scoring)
        └── reviewer_agent.py   # ReviewerAgent class + run_reviewer_agent() entry point
```

`llm_json.py` exists because hypothesis and experiment_planner both needed
identical "invoke the model, parse JSON, retry once with a repair prompt on
failure" logic — factored out once a second agent needed it, so it doesn't
drift between copies. Reuse it for any new agent that asks the model for JSON.

`llm_sections.py` is its counterpart for responses that carry **source code**,
and exists because JSON is the wrong transport for that. A JSON string value
forces every newline, quote and backslash in generated code through an escaping
round-trip the model performs by hand, and a small quantized model gets it wrong
constantly — a literal `\d` in a regex, a real newline instead of `\n`, a
stray trailing backslash. Instead of adding a repair for each flavour of
corruption, code is transported unescaped between markers:

```
===BEGIN load_data_function===
def load_data():
    return pd.read_csv(PATH)   # a real newline is a real newline
===END load_data_function===
```

`invoke_sections` mirrors `invoke_json`'s shape exactly (one repair retry, then
raise), so a call site swaps between them without other changes. The Coder Agent
uses it for every code-bearing call; short structured responses (its self-review
verdict, every other agent) stay on `llm_json.py`.

`checkpointer.py` is the same "one factory, one place" idea as `llm.py`: all
eight compiled graphs call `get_checkpointer()`, so the whole pipeline switches
between in-process and durable checkpointing through one env var rather than
eight edits. See [Checkpointing and caching](#checkpointing-and-caching).

### Adding a new agent

Two patterns exist in this repo, both fine to copy:

- **Graph agent** (like `literature/`): copy its shape into `agents/<name>/`
  — `state.py`, `nodes.py`, `graph.py` (+ `clients.py` for external calls).
  Use this when the work genuinely branches/parallelizes.
- **Callable agent** (like `hypothesis/`): a class with a `.run(...)` method
  and a module-level `run_<name>_agent(...)` convenience function, following
  `hypothesis_agent.py`. Use this when the flow is a straightforward
  sequence (e.g. map-then-reduce) — it's simpler to read, test, and call
  from other Python code than a graph would be.

Either way: register a subcommand for it in `cli.py`.

### Running the whole pipeline (orchestrator)

`orchestrator/` wires all seven agents into a single LangGraph — Literature →
Interdisciplinary Literature → Hypothesis → Experiment Planner → Coder →
Writer ⇄ Reviewer → finalize — so a full run is one call instead of seven
hand-chained ones:

```python
import uuid
from research_pipeline.orchestrator import build_pipeline_graph

state = build_pipeline_graph().invoke(
    {"research_question": "does retrieval augmentation help small models?"},
    config={"configurable": {"thread_id": str(uuid.uuid4())}},
)
state["final_result"]  # {final_paper_path, iterations_run, converged, unresolved_issues, ...}
```

The Writer/Reviewer cycle is a real conditional edge, not a Python loop:
`review` routes back to `draft_or_revise` until `overall_pass` is True or
`max_iterations` is spent — the same stop conditions, output files
(`v1.pdf`, `v2.pdf`, …, `review_log.json`) and result shape as
`run_writer_reviewer_loop`, which stays available for running just that loop
against existing upstream output. The routing reads a value the Reviewer Agent
already derives deterministically; the orchestrator adds no model judgment of
its own. Each node calls the same `run_<name>_agent()` entry point documented
below, so every stage still writes its own outputs and validates its own input.

One stage narrows the run: the Experiment Planner is asked to plan only the
hypothesis the Hypothesis Agent ranked first, so the Coder and Writer stages
spend their effort on one hypothesis instead of three. All 3 hypotheses (with
their ranking and the justification for the winner) still reach the Writer and
Reviewer, which is what lets the paper say which hypotheses were considered and
why one was taken forward. Running `experiment-planner` standalone from the CLI
or from disk still plans all 3, exactly as before.

The Writer and Reviewer are both handed the *merged* paper pool (in-domain +
cross-field), not just the Literature Agent's own output — both already accept
"a dict with a `papers` key", which is exactly the interdisciplinary agent's
output shape. Without that, a hypothesis grounded in a cross-field paper could
never be cited: the marker would be stripped as unresolvable and reported as a
citation issue the Writer had no way to fix. They must be given the same set as
each other, or the Reviewer flags citations the Writer was entitled to make.

Optional inputs, all defaulting to the agents' own settings:
`max_results_per_query`, `download_dir`, `output_dir`, `max_iterations`,
`quality_threshold`. Pass preconfigured `writer`/`reviewer` agents (e.g.
sharing one `chat_model`) via `config["configurable"]` rather than the state,
which is checkpointed and so must stay serializable.

### Chaining agents individually (Literature → Interdisciplinary Literature → Hypothesis → Experiment Planner → Coder → Writer ⇄ Reviewer)

Each agent's entry point takes a plain `dict`/`list[dict]` shaped like the
previous agent's *output*, never the previous agent's internal state — so
every link in the chain works identically in-process or decoupled via disk:

```python
# in-process: no serialization round-trip
lit_result = literature_agent.invoke({...})
ida_result = run_interdisciplinary_literature_agent(lit_result["merged_papers"], research_question=lit_result["research_question"])
hyp_result = run_hypothesis_agent(
    ida_result["papers"],                                    # in-domain + cross-field, deduped
    research_question=ida_result["research_question"],
    interdisciplinary_context=ida_result["bridge_insights"],  # optional; omit to skip the cross-field agent entirely
)
# plan every hypothesis, or narrow to the one the ranking picked:
plan_result = run_experiment_planner_agent(hyp_result, hypothesis_ids=[hyp_result["selected_hypothesis_id"]])
code_result = run_coder_agent(plan_result)
paper_result = run_writer_agent(lit_result, hyp_result, plan_result, code_result)
review_result = run_reviewer_agent(paper_result["paper_path"], paper_result, lit_result, hyp_result, plan_result, code_result)

# or run the writer + reviewer as a single feedback loop (draft -> review -> revise -> re-review):
from research_pipeline.writer_reviewer_loop import run_writer_reviewer_loop
loop_result = run_writer_reviewer_loop(lit_result, hyp_result, plan_result, code_result)

# or decoupled: read a previous run's output straight from disk, anywhere in the chain
import json
plan_data = json.loads(Path("outputs/experiment_plan_<timestamp>.json").read_text())
code_result = run_coder_agent(plan_data)
```

`run_experiment_planner_agent`, `run_coder_agent`, `run_writer_agent`, and
`run_reviewer_agent` all validate their input against the *previous* agent(s)'
own `validate_output` before doing anything else — a malformed/outdated file
fails immediately with a clear error instead of confusing the model or
generating a paper/review against garbage. The writer agent is the
results-analysis stage: it reads the coder agent's `experiments` list itself
and only reports metrics for entries with `status == "completed"` —
`code_generated_not_run`/`skipped` entries are stated as such in the paper's
Results/Limitations sections, never described as completed work (see
[writer_agent.py](src/research_pipeline/agents/writer/writer_agent.py)'s
docstring for exactly how). The reviewer agent re-derives that same
ground truth independently (it doesn't trust the writer's summary as fact)
and checks the *rendered* paper against it — see
[reviewer_agent.py](src/research_pipeline/agents/reviewer/reviewer_agent.py)'s
docstring for which checks are deterministic vs. LLM-based.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in LLM_BASE_URL / SEMANTIC_SCHOLAR_API_KEY / CORE_API_KEY
```

Optional, if you're going to change code:

```bash
uv run pre-commit install          # one-time, wires the git hook
uv run pre-commit run --all-files  # manual pass: ruff --fix, ruff format, mypy
```

Lint/format/type-check rules live in `pyproject.toml`; `.pre-commit-config.yaml`
currently scopes them to the coder agent only (see
[its AGENTS.md](src/research_pipeline/agents/coder/AGENTS.md)).

### LLM backend

Every agent is LLM-powered and they all share one model, built in one place:
[`llm.get_chat_model()`](src/research_pipeline/llm.py). There is a single
backend — HTTP to any OpenAI-compatible server via `LLM_BASE_URL`/`LLM_MODEL`
— and every agent shares the same endpoint; there is no per-agent override.
The pipeline is wired specifically for that endpoint to be vLLM serving
[Qwen3-Coder 30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
(`Qwen/Qwen3-Coder-30B-A3B-Instruct`), but nothing is vLLM-specific —
any OpenAI-compatible server works by changing `LLM_BASE_URL`/`LLM_MODEL`.
`LLM_MODEL` must match the id the server advertises: the SLURM script passes
no `--served-model-name`, so that's the full HF repo id; the model card's own
example aliases it to `model`, in which case set `LLM_MODEL=model`.

> An earlier version of this codebase also supported an in-process
> `LLM_BACKEND=huggingface` path (no server, model loaded directly with
> `transformers`) for smaller models like Nemotron Nano 12B v2 on hardware too
> small for the 30B. That backend has been removed from `llm.py`/`config.py`
> now that the pipeline targets vLLM + the 30B exclusively —
> `scripts/slurm/run_pipeline_hf.sbatch`/`run_pipeline_hf_container.sbatch`
> and `notebooks/kaggle_nemotron_nano_v2_pipeline.ipynb` still reference it and
> no longer work; use `run_pipeline.sbatch` (below) instead.

**Reasoning mode.** The pipeline no longer serves a reasoning model — Qwen3-Coder-Instruct
answers directly. The machinery below is retained so that pointing `LLM_MODEL` at a
reasoning model needs no code change. Nemotron 3 Nano thought before answering, emitting a
`<think>...</think>` trace. `LLM_ENABLE_THINKING` (default `false`) controls
it, sent as the model's documented `chat_template_kwargs.enable_thinking` on
every request. It's off because no stage consumes reasoning traces and a full
run makes dozens of calls. Independently of that flag,
[`llm_json.strip_reasoning`](src/research_pipeline/llm_json.py) strips any trace
that arrives anyway — a server started *without* vLLM's `nano_v3` reasoning
parser puts the trace in `content`, where it would otherwise break every
agent's JSON parse. Turning thinking on is therefore safe either way; raise
`LLM_TEMPERATURE`/`LLM_TOP_P` to the card's recommended `1.0`/`1.0` and give
`LLM_MAX_TOKENS` room if you do.

**On Barkla:** the vLLM path needs a container first, since there is no vLLM
module. Build it once on a *viz* node (never the login node — long tasks there
are killed without warning):

```bash
bash scripts/slurm/build_vllm_sif.sh
```

Then run the whole pipeline — server and all seven agents — as a single job:

```bash
# Qwen3-Coder 30B-A3B-Instruct (~57GiB) — vLLM served
sbatch scripts/slurm/run_pipeline.sbatch "your research question"
```

[`run_pipeline.sbatch`](scripts/slurm/run_pipeline.sbatch) starts vLLM on the
allocated GPU, waits for `/v1/models` to answer, then runs `orchestrate`
against `localhost` on the same node. The pipeline is pure CPU and the GPU
nodes have 96–168 cores, so co-locating costs nothing and avoids both the SSH
tunnel and any long-running process on the login node.

Defaults to `gpu-h100` with **two** GPUs (`--gres=gpu:2`, `TP=2`): the 30B
BF16 weights are ~60GB, which fits a single 80GB H100 or A100 on their own
(not a 48GB L40S, and not a V100 at all — 16GB, and Volta has no bfloat16) —
the second GPU here isn't for fitting the weights, it's so they shard across
both cards and leave real KV-cache headroom for `--max-model-len`, which
targets a measured 131072-token window (Barkla job 10274103 reported a
150,272-token KV cache at TP=1)
ceiling (see [`_vllm_serve.sh`](scripts/slurm/_vllm_serve.sh)'s comment for
the reasoning and what to do if that doesn't fit Barkla's real KV budget).
Drop back to one
GPU (and a smaller `--max-model-len`/`LLM_CONTEXT_WINDOW`) if the extra
context headroom isn't worth doubling the job's footprint on a shared
partition. Both scripts derive their port from the job id, since GPU nodes are
shared.

To keep a long-lived server for interactive work instead, use
[`run_llm_server.sbatch`](scripts/slurm/run_llm_server.sbatch); it prints the
node, port, and the exact tunnel command on startup.

Model weights and the `.sif` live on `fastscratch` (`HF_HOME`), not `home` —
home is capped at 75GB/100k inodes and the weights alone exceed it.

**On a Kaggle GPU:** [`notebooks/kaggle_gemma4_pipeline.ipynb`](notebooks/kaggle_gemma4_pipeline.ipynb)
runs the whole pipeline in a Kaggle notebook against a 4-bit
[Gemma 4 12B](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) GGUF served
locally by `llama-server`. It changes no pipeline code: `llama-server` speaks
the OpenAI API, so it's reached the same way any other server is — just point
`LLM_BASE_URL` at loopback. Quantization is what makes it fit — ~7GB at
`Q4_K_M` against ~24GB at BF16, on a 16GB T4. (This notebook predates the
pipeline narrowing to the 30B model above; it's kept as-is for smaller
hardware, but no longer reflects the pipeline's primary target.)

[`scripts/kaggle/gguf_server.py`](scripts/kaggle/gguf_server.py) holds the
deployment plumbing (build, download, launch, health-check) and imports nothing
from `research_pipeline`, so it works from a plain terminal or any GPU box too:

```bash
python scripts/kaggle/gguf_server.py --foreground &
LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=gemma-4-local \
  research-pipeline orchestrate "your research question"
```

It builds llama.cpp from source rather than fetching a binary because llama.cpp
publishes no prebuilt *Linux CUDA* archive (the CUDA builds are Windows-only),
and a CPU build would make a run take hours. The build is narrowed to the
`llama-server` target and to the GPU's own compute capability — probed with
`nvidia-smi`, since Kaggle hands out T4s, P100s and L4s and compiling for all of
them multiplies build time for nothing. Two Kaggle-specific gotchas worth
knowing: `settings` is frozen at import, so every `LLM_*` variable must be set
**before** the first `import research_pipeline`; and `uv` must be on the kernel's
`PATH`, or the Coder Agent can't provision an isolated environment for generated
experiment code and reports `code_generated_not_run` instead of results.

**Kaggle as just the backend, browser UI running locally:** the notebook's
section 3c (skip it if you're running the whole notebook end to end) opens a
[Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
in front of `llama-server`, since a Kaggle notebook accepts no inbound
connections and `LLM_BASE_URL=http://127.0.0.1:8000/v1` is only reachable from
inside the kernel. [`scripts/kaggle/tunnel.py`](scripts/kaggle/tunnel.py) holds
that plumbing the same way `gguf_server.py` holds the model server's — stdlib
only, no pipeline import, downloads and caches the `cloudflared` binary itself:

```python
import tunnel
llm_tunnel = tunnel.start_tunnel(gguf_server.DEFAULT_PORT)
print(llm_tunnel.public_url)   # https://<random-words>.trycloudflare.com
```

Then, on your own machine (not the Kaggle kernel):

```bash
uv sync --extra webapp
LLM_BASE_URL=https://<random-words>.trycloudflare.com/v1 \
  LLM_MODEL=gemma-4-local LLM_API_KEY=not-needed \
  uv run research-pipeline serve
```

Open <http://127.0.0.1:8000> and start a run as usual (see "Watching a run in
a browser" below) — every LLM call now crosses the tunnel to the Kaggle GPU,
while arXiv/Semantic Scholar/CORE are still queried directly from your machine.
The tunnel URL is **unauthenticated** — anyone who has it can call the model —
and only live as long as the notebook's tunnel cell and Kaggle session stay up,
so treat it as throwaway: don't post it anywhere public, and stop it (notebook
section 7) once you're done.

### Semantic Scholar

Get a key at https://www.semanticscholar.org/product/api#api-key and set
`SEMANTIC_SCHOLAR_API_KEY` in `.env`. Without it, Semantic Scholar search is
skipped (logged as a warning) rather than failing the whole run — unauthenticated
requests to that API are aggressively rate-limited / rejected with 403s.

### CORE

Get a free key at https://core.ac.uk/services/api and set `CORE_API_KEY` in
`.env`. CORE has no unauthenticated tier, so without a key CORE search is
skipped entirely (logged as a warning) rather than failing the whole run.

### Checkpointing and caching

Every agent graph and the orchestrator compile with a LangGraph checkpointer, so
a run's progress is recorded node by node under its `thread_id`. All eight build
theirs from one factory,
[`checkpointer.get_checkpointer()`](src/research_pipeline/checkpointer.py), which
`CHECKPOINTER_BACKEND` selects:

| Backend | What it does | Needs |
| --- | --- | --- |
| `memory` (default) | Checkpoints live in the process and die with it. | — |
| `sqlite` | One file at `CHECKPOINTER_SQLITE_PATH` (default `checkpoints/pipeline.db`), created on first use. | `uv sync --extra checkpoint-sqlite` |
| `postgres` | `CHECKPOINTER_POSTGRES_URI`, for a shared or multi-host setup. | `uv sync --extra checkpoint-postgres` |

`memory` is the default because it needs no extra dependency and reproduces what
every graph did before this setting existed. Switch to `sqlite` wherever the
process itself is at risk — a pre-empted SLURM job on Barkla, a restarted Kaggle
kernel, a crashed web-app runner — since that's exactly where in-process
checkpoints are worth nothing. One file is shared by every graph in the process:
checkpoints are keyed by `(thread_id, checkpoint_ns)` and every call site already
mints its own `thread_id`, so there's nothing to collide.

Durability is what makes resuming possible, and both unattended entry points use
it. A web-app run that failed or was cancelled gets a **Resume** button, which
continues it from the stage it died in because its `thread_id` has always been
the run id; `orchestrate-batch` records each question's `thread_id` in the
manifest, so a resubmitted job resumes a question that was pre-empted half-way
instead of only skipping the ones that finished. Neither does anything under the
default `memory` backend — the checkpoints died with the process being resumed —
so this is the payoff for turning `sqlite` on, and the Resume button is hidden
entirely under `memory` rather than offering a restart that calls itself a
resume.

Resuming replays from the checkpoint, which means it uses the **checkpointed**
inputs. Editing a run's parameters and relaunching will not change them; start a
new run for that.

Two smaller reliability settings sit alongside it:

- **Node retries.** LLM- and search-calling nodes carry a
  `RetryPolicy(max_attempts=2)` as an outer safety net over the retries that
  already exist further down (`llm.py`'s client-level `max_retries`,
  `clients.py`'s `_request_with_retry`). It uses LangGraph's default `retry_on`,
  which deliberately does *not* retry `ValueError`/`RuntimeError` and friends —
  a schema failure that survived `llm_json.py`'s repair round-trip is a
  persistent problem, not a flaky one. Nodes with non-idempotent side effects are
  excluded on purpose: the Coder's `attempt`/`snapshot_and_regenerate` fix loop
  (retrying would re-execute generated code or re-provision an environment) and
  `download_papers` (already partial-success tolerant).
- **Paper-search caching.** `ENABLE_PAPER_SEARCH_CACHE` (default `true`) caches
  the arXiv / Semantic Scholar / CORE search nodes and the interdisciplinary
  per-field search on their inputs, for `PAPER_SEARCH_CACHE_TTL_SECONDS`
  (default 3600). LangGraph's only cache backend is in-memory, so this pays off
  inside one long-lived process — the web app across runs, an `orchestrate-batch`
  sweep across a question list — and does nothing for a one-shot CLI call, which
  always starts with an empty cache.

## Run

To run every stage end to end in one command:

```bash
uv run research-pipeline orchestrate "recent approaches to reducing hallucination in RAG systems" \
    --max-results 5 \
    --output-dir outputs/paper
```

This runs the orchestrator graph described above and prints the final paper
path, how many Writer/Reviewer iterations ran, whether it converged, and any
unresolved issues. Every stage still writes its own outputs along the way, so
a run can be inspected (or picked up from disk) stage by stage.

### Watching a run in a browser

`orchestrate` blocks for tens of minutes and prints nothing until it finishes.
The web UI runs the same graph and shows each stage completing as it completes:

```bash
uv sync --extra webapp
uv run research-pipeline serve
```

Then open <http://127.0.0.1:8000>, enter a research question, and watch. Each
stage reports what it actually produced — papers found, the adjacent fields
explored and the bridge insights they yielded, the three hypotheses with their
ranking and which one was selected, per-plan feasibility and complexity,
per-experiment status and fix attempts,
then the Writer/Reviewer iterations with their quality scores and issue counts.
The pipeline's own log lines stream underneath, every draft's PDF is openable
the moment it's written (not just the final one), and a run can be cancelled.

Three processes, not one. The server never runs a pipeline itself: starting a
run spawns `python -m research_pipeline.webapp.runner <run_dir>`, and the
server thereafter only reads files that subprocess writes. That is what makes a
run cancellable at all — a thread sitting inside `graph.invoke()` cannot be
interrupted, but a subprocess takes `SIGTERM` — and it keeps a crash in the
Coder Agent's sandbox from taking the server down, lets the server be restarted
mid-run, and gives each run its own `CODER_EXPERIMENTS_DIR`, which is the one
output location the graph can't route per run.

Progress comes from `graph.stream(..., stream_mode="updates")`, which already
yields `{node_name: delta}` at each node boundary. No agent and no orchestrator
code is modified or reimplemented for the UI; everything it displays is read
out of an agent's own validated output.

Each run is one self-contained, rsync-able directory under `WEBAPP_RUNS_DIR`
(default `runs/`):

```
runs/<run_id>/
    run.json        question, params, status, pid, timings, final result
    events.jsonl    the progress + log stream the browser polls
    stdout.log      the runner's own stdout/stderr
    outputs/        v1.pdf, v1_summary.json, ..., review_log.json
    papers/         downloaded PDFs + metadata.json
    experiments/    generated experiment code
```

Three things worth knowing before running it anywhere shared:

- **There is no authentication.** The app can start jobs and read files, so it
  binds `127.0.0.1` by default and should stay there on a shared machine.
  `serve` warns if you bind anything else.
- **On Barkla it has to live inside the SLURM allocation**, alongside vLLM, the
  same way [`run_pipeline.sbatch`](scripts/slurm/run_pipeline.sbatch) co-locates
  the pipeline with the model server — the runner needs `localhost` access to
  `LLM_BASE_URL`, so a login-node server is no use. Reach it with a tunnel:
  `ssh -L 8000:<compute-node>:8000 <barkla-host>`.
  [`run_webapp.sbatch`](scripts/slurm/run_webapp.sbatch) sets this up (vLLM +
  `serve` on one node, `CHECKPOINTER_BACKEND=sqlite` so a pre-empted job stays
  resumable) and prints the exact tunnel command — see
  [BARKLA.md](docs/BARKLA.md) §7.
- **A killed run can only be resumed with a durable checkpointer.** The default
  `CHECKPOINTER_BACKEND=memory` is per-process, so a run whose process dies has
  nothing to resume to and restarting starts it over. With `sqlite` or
  `postgres` (see `.env.example`), the UI offers a Resume button on a
  failed/cancelled run instead. Either way, a run
  whose process dies is detected and reported as failed rather than left
  showing as running forever.

Concurrency is capped at `WEBAPP_MAX_CONCURRENT_RUNS` (default 1): the pipeline
talks to a single LLM endpoint, so a second simultaneous run competes with the
first for it rather than finishing sooner. A refused start says so instead of
queueing silently.

### Many questions at once

To sweep a whole list of research questions unattended — for example to collect
a corpus of runs:

```bash
uv run research-pipeline orchestrate-batch \
    --questions-file questions.txt \
    --output-dir outputs/batch
```

`questions.txt` is one research question per line (blank lines and `#` comments
are skipped). Each question is an independent run with its own output
subdirectory, so nothing leaks between them.

Two things make this survive a long unattended sweep. A `batch_manifest.json`
is rewritten after **every** question, recording status, output dir, paper
path, and any error — so re-running the same command skips questions already
marked completed and picks up where a killed job left off (`--no-resume` forces
a full redo). And one question's failure is caught and recorded rather than
ending the run; a `--max-consecutive-failures` circuit breaker (default 5)
stops the sweep only when failures are consecutive, which is the signature of
something systemic like a dead model server. Questions never reached stay
`pending`, so a later re-run still attempts them.

For Barkla, [`run_pipeline_batch.sbatch`](scripts/slurm/run_pipeline_batch.sbatch)
runs this as a job array, each task taking a contiguous slice of the question
list and writing into one shared manifest:

```bash
# 10 array tasks, at most 3 running at once
sbatch --array=0-9%3 scripts/slurm/run_pipeline_batch.sbatch questions.txt 10
```

An array rather than one long job because 100 questions can exceed even
`gpu-h100`'s 3-day cap, and a single job failing loses the whole sweep; a vLLM
server per task rather than one shared server job because a shared server is a
single point of failure for every task at once, and the model load amortizes
fine across a slice of 10-20 questions. Resubmitting the same command after a
pre-emption resumes from the manifest.

The per-agent subcommands below run a single stage — useful for iterating on
one agent, or resuming from a previous run's JSON.

```bash
uv run research-pipeline literature "recent approaches to reducing hallucination in RAG systems" \
    --max-results 5 \
    --download-dir papers
```

This searches arXiv + Semantic Scholar + CORE, dedupes by DOI/title, downloads
available PDFs into `papers/`, and writes `papers/metadata.json`.

```bash
uv run research-pipeline interdisciplinary-literature --from-file papers/metadata.json --output-dir outputs
```

This asks the model which adjacent fields could inform the same problem (up to
`INTERDISCIPLINARY_MAX_FIELDS`, default 3), searches each of them on arXiv +
Semantic Scholar + CORE with that field's own generated queries, merges and dedupes
what it finds against the in-domain papers, synthesizes bridge insights tying
the cross-field work back to the core problem, and writes
`outputs/interdisciplinary_<UTC timestamp>.json` — see
[interdisciplinary_literature_agent.py](src/research_pipeline/agents/interdisciplinary_literature/interdisciplinary_literature_agent.py)'s
docstring for the exact output schema. Its `papers` key is a drop-in for the
`hypothesis` subcommand's `--from-file`, which also picks up its
`bridge_insights` automatically.

```bash
uv run research-pipeline hypothesis --from-file outputs/interdisciplinary_<timestamp>.json --output-dir outputs
```

This synthesizes the paper set, extracts methods/gaps, generates exactly 3
hypotheses, ranks them and names a `selected_hypothesis_id`, then writes
`outputs/hypotheses_<UTC timestamp>.json` — see
[hypothesis_agent.py](src/research_pipeline/agents/hypothesis/hypothesis_agent.py)'s
docstring for the exact output schema. `--from-file` accepts a Literature Agent
`metadata.json`, an Interdisciplinary Literature Agent output, or a bare JSON
list of papers.

```bash
uv run research-pipeline experiment-planner --from-file outputs/hypotheses_<timestamp>.json --output-dir outputs
```

This judges feasibility per hypothesis, produces a full implementation-ready
plan for each (objective, variables, design, data requirements, methods,
evaluation, ordered implementation steps, complexity, risks), notes any
shared infrastructure across the plans, ranks a priority order, and writes
`outputs/experiment_plan_<UTC timestamp>.json` — see
[experiment_planner_agent.py](src/research_pipeline/agents/experiment_planner/experiment_planner_agent.py)'s
docstring for the exact output schema and its compute/data feasibility
assumptions (a shared SLURM/GPU HPC cluster; source papers assumed minable
for full-text detail).

```bash
uv run research-pipeline coder --from-file outputs/experiment_plan_<timestamp>.json --output-dir outputs
```

This processes experiment plans in `priority_order`, skips `feasible: false`
ones (logged, not silently dropped), generates a runnable project per
feasible plan under `experiments/<hypothesis_id>/`, and — for `low`/`medium`
`estimated_complexity` plans that don't need a GPU this environment lacks —
actually runs it and captures `results.json`. GPU-needing plans with no GPU
detected get code plus a `run.sbatch` template instead of being run
synchronously, since there's nothing to run them on locally; **by default the
agent does not submit SLURM jobs itself**, review the script first (see
[Autonomous fixing](#autonomous-fixing-and-slurm-auto-submission) to change
that). `high`-complexity plans get the same sbatch-only treatment by default —
set `CODER_RUN_HIGH_COMPLEXITY_WHEN_GPU_AVAILABLE=true` to instead run them
synchronously like `low`/`medium` whenever a GPU is actually detected in this
process (a Kaggle notebook, a Barkla node reached via `run_pipeline.sbatch`),
bounded by `CODER_HIGH_COMPLEXITY_TIMEOUT_SECONDS` (default 1800s) rather than
the low/medium timeout table — there's no shared cluster queue to protect when
the compute is already dedicated to this pipeline. Writes
`outputs/coder_agent_summary_<UTC timestamp>.json` — see
[coder_agent.py](src/research_pipeline/agents/coder/coder_agent.py)'s
docstring for the exact output schema and execution model (confirmed with
the pipeline owner, not assumed).

#### Real data instead of invented data

Before generating each experiment, the agent looks up one real dataset matching
the plan's `data_requirements` — Hub search for candidates, then the
[Hugging Face Dataset Viewer](https://huggingface.co/docs/dataset-viewer/index)
to confirm the candidate is actually servable and to read its real column names
and first rows. The match (id, config, split, schema, sample rows) plus the exact
`/rows` REST URL goes into the codegen prompt, so `load_data()` can pull real
records with `requests` — no `datasets` package in the experiment's throwaway
venv, and no dataset cache on shared scratch. The model is told to ignore the
dataset if it doesn't genuinely fit the plan, and to say so in
`assumptions_made`.

This is an enhancement to a prompt, never a dependency: it's only attempted when
the runtime network probe succeeds, and every failure (no network, rate limit, a
dataset the viewer can't serve) degrades silently to generating exactly as
before. `CODER_ENABLE_HF_DATASET_SEARCH=false` turns it off for fully offline or
reproducible runs; `HUGGINGFACE_API_TOKEN` is optional and only raises rate
limits.

Whether or not a dataset was found, `load_data()` must not *assume* its data is
there. `sandbox.check_data_fallback` parses the generated `load_data` and flags
any `open`/`pandas.read_*`/`numpy.load` that isn't inside a `try` block, routing
it back through the fix loop as `missing_data_fallback`. This is AST-based rather
than regex precisely because the question is "is this read guarded?", and it
exists because a real run produced code that assumed a `survey_data.csv` would
be present for a plan that required collecting new data — the prompt had always
asked for a synthesized fallback, and nothing checked.

#### Autonomous fixing and SLURM auto-submission

When generated code fails a check, the agent doesn't give up on it — it feeds
the actual error back to the model and regenerates, up to
`CODER_MAX_FIX_ATTEMPTS` times (default 3). The error is whatever the failing
check produced: a compile error, a static-lint finding, an unguarded data read,
the stdout/stderr tail from the run, or the traceback `run.py` recorded in
`results.json["error"]`.
Every stop condition is a real check's verdict rather than a model's opinion
about whether the code looks right. Env-provisioning failures are deliberately
not retried — regenerating code can't install a missing package. Each failed
attempt is preserved under `experiments/<hypothesis_id>/fix_attempts/attempt_<n>/`
and summarized in the output's `fix_history`, which is also the useful artifact
if you're mining runs for training data: failing code, its error, and the fix.

Plans deferred to `run.sbatch` (GPU-needing with no GPU present, or
`high`-complexity without `CODER_RUN_HIGH_COMPLEXITY_WHEN_GPU_AVAILABLE`)
can't be run locally, so there's no real error to
learn from. Setting `CODER_AUTO_SUBMIT_SLURM=true` (default **false**) lets the
agent submit those to the cluster itself during unattended sweeps. It still
refuses unless every gate passes:

| Gate | What it does |
|---|---|
| `sandbox.static_safety_check` | Refuses code containing `eval`/`exec`, `shell=True`, `os.system`, `rmtree`, raw sockets, credential-shaped env lookups, and similar |
| LLM pre-flight review | Reads the code against the plan; concerns route back through the same fix loop |
| `CODER_MAX_CONCURRENT_SLURM_JOBS` | Checked against `squeue`, so it holds across every process in a sweep. A failed probe blocks rather than waves through |
| `CODER_MAX_SLURM_JOBS_PER_RUN` | Per-question ceiling, so one plan set can't flood the queue |

Anything that trips a gate falls back to writing `run.sbatch` for manual
review, with a `reason` naming which gate stopped it — an auto-submit that
didn't happen never looks like an ordinary "needs a human" result.
Successfully submitted experiments get `status: "submitted_to_slurm"` and a
`slurm_job_id`; their results aren't part of that run, since the job outlives
it.

```bash
uv run research-pipeline writer \
    --literature-file papers/metadata.json \
    --hypothesis-file outputs/hypotheses_<timestamp>.json \
    --planner-file outputs/experiment_plan_<timestamp>.json \
    --coder-file outputs/coder_agent_summary_<timestamp>.json \
    --output-dir outputs
```

This drafts a full paper section by section (each LLM call sees only the
slice of upstream data it needs, so the full pipeline output is never
crammed into one context window), resolves every `[[cite:PAPER_ID]]`/
`[[citet:PAPER_ID]]` marker the model wrote into a NeurIPS-style author-year
citation (e.g. "(Smith et al., 2020)") against the literature agent's actual
papers (dropping and flagging any that don't match — see `notes_for_review`
in the output), and renders it to `outputs/paper_<UTC timestamp>.pdf` — Times
font, numbered sections, a references list sorted alphabetically by author —
via [reportlab](https://www.reportlab.com/) (pure Python — no LibreOffice/Node
required, so it runs unmodified on Barkla). Set `WRITER_PAPER_AUTHORS` /
`WRITER_PAPER_AFFILIATION` in `.env` to replace the default anonymized-submission
placeholder ("Anonymous Author(s)" / "Anonymous Institution") with real
names. Also writes `outputs/paper_summary_<UTC timestamp>.json` — see
[writer_agent.py](src/research_pipeline/agents/writer/writer_agent.py)'s
docstring for the exact output schema.

```bash
uv run research-pipeline reviewer \
    --literature-file papers/metadata.json \
    --hypothesis-file outputs/hypotheses_<timestamp>.json \
    --planner-file outputs/experiment_plan_<timestamp>.json \
    --coder-file outputs/coder_agent_summary_<timestamp>.json \
    --paper-summary-file outputs/paper_summary_<timestamp>.json \
    --output-dir outputs
```

This reads the PDF back (via
[pdf_reader.py](src/research_pipeline/agents/writer/pdf_reader.py), the same
module `writer.revise()` uses), runs the deterministic citation/results/
coverage checks plus the LLM-based hallucination/framing/quality checks
against the *same* upstream ground truth the writer used, and writes
`outputs/review_<UTC timestamp>.json` matching
[reviewer_agent.py](src/research_pipeline/agents/reviewer/reviewer_agent.py)'s
`ReviewOutput` schema — `overall_pass`, per-category issue lists, 1-5 quality
scores, and a consolidated `feedback_for_writer`.

```bash
uv run research-pipeline writer-reviewer-loop \
    --literature-file papers/metadata.json \
    --hypothesis-file outputs/hypotheses_<timestamp>.json \
    --planner-file outputs/experiment_plan_<timestamp>.json \
    --coder-file outputs/coder_agent_summary_<timestamp>.json \
    --max-iterations 3 --quality-threshold 4 \
    --output-dir outputs/paper
```

Runs the full draft → review → revise → re-review loop
([writer_reviewer_loop.py](src/research_pipeline/writer_reviewer_loop.py)),
writing every iteration's PDF (`outputs/paper/v1.pdf`, `v2.pdf`, ...) and
summary, plus a consolidated `outputs/paper/review_log.json` with every
iteration's full review — so the whole revision history is auditable even
after the loop finishes, not just the final verdict. Stops early once the
Reviewer reports zero issues and every quality score clears the threshold;
otherwise runs to `max_iterations` and returns the last draft alongside
`unresolved_issues` rather than silently treating it as done.

### LLM used by the interdisciplinary-literature / hypothesis / experiment-planner / coder / writer / reviewer agents

All six reuse the same `research_pipeline.llm.get_chat_model()` client as
the literature agent (Nemotron 3 Nano, i.e. whatever `LLM_BASE_URL`/`LLM_MODEL`
in `.env` points at) rather than a separate client — interdisciplinary-literature/hypothesis/experiment-planner/coder/reviewer
at a lower temperature (0.1) suited to grounded extraction/planning/codegen/judgment,
writer at 0.2 suited to grounded prose. Each agent also accepts a `chat_model`
argument, so one shared client (or a fake, in tests) can be threaded through
the whole pipeline. Every agent is LLM-driven — including the literature agent,
which uses the model to expand a research question into search queries (falling
back to the raw question if the server is unreachable), and the reviewer, whose
verifiable checks are deliberately deterministic with only hallucination nuance
and framing/tone honesty going to the model. If you specifically want Anthropic's
Claude models here, point `LLM_BASE_URL`/`LLM_MODEL` at an OpenAI-compatible
Claude endpoint, or swap `get_chat_model()` for
`langchain_anthropic.ChatAnthropic` — nothing elsewhere in any of these
agents assumes a particular provider.

### What the coder agent actually does to your machine

Worth knowing before running it unattended:
- Writes files under `experiments/<hypothesis_id>/` and `experiments/_shared/`
  (both gitignored by default).
- For plans it decides to run, probes network access (a quick connection
  attempt to pypi.org:443) and GPU presence (`nvidia-smi` on PATH) at
  **runtime**, not from a hardcoded assumption — so the same code adapts
  whether it's run locally or on a Barkla compute node.
- If a generated `requirements.txt` needs a package that isn't already
  importable, it creates an **isolated `uv venv` inside that experiment's own
  directory** and installs into it — it never touches this pipeline's own
  `.venv` — and only attempts this if network access was detected.
- Runs `python run.py` as a subprocess with a bounded timeout (120s for
  `low`, 300s for `medium` by default — `CODER_LOW_COMPLEXITY_TIMEOUT_SECONDS`
  / `CODER_MEDIUM_COMPLEXITY_TIMEOUT_SECONDS`; `high` is never run
  synchronously at all unless
  `CODER_RUN_HIGH_COMPLEXITY_WHEN_GPU_AVAILABLE` is on).
- Usually runs it **twice**: first a shrunken copy (`run_smoke.py`, every known
  cost knob pinned to its floor, `CODER_SMOKE_TIMEOUT_SECONDS`), deleted along
  with the results.json it wrote as soon as it finishes, and then the real run.
  The point is that a defect anywhere in the program surfaces in seconds rather
  than after the full timeout — so a fix-loop round costs seconds too. The smoke
  run can only end an attempt early; it can never let one skip the real
  execution. `CODER_ENABLE_SMOKE_RUN=false` turns it off.
- Never submits anything to SLURM itself; `run.sbatch` files are generated
  for a human to review and submit.

## Tests

```bash
uv run pytest
```

## Notes on this version vs. the original notebook

- LLM/Barkla config is env-driven (`.env`) instead of hardcoded/Kaggle-specific.
- Semantic Scholar / arXiv / CORE requests retry transient failures (429/5xx) with backoff.
- Query generation falls back to the raw research question on *any* LLM failure
  (not just bad JSON), and de-dupes near-identical generated queries.
- Downloaded files are checked against their `Content-Type` / PDF magic bytes
  before being saved, so a paywall/error page can't silently masquerade as a PDF.
- Download filenames include a paper ID suffix to avoid collisions between
  same-titled papers.
- PDF downloads run concurrently (thread pool) instead of one at a time.
- The graph is compiled with an in-memory LangGraph checkpointer, keyed by a
  per-run `thread_id`, so a crash mid-run has a resume point.
- Added the **hypothesis** agent: synthesizes a literature agent's paper set
  into a summary, methods overview, gaps, and 3 grounded/cited hypotheses,
  returned as validated JSON and written to `outputs/`. Batches large paper
  sets to stay within context, retries once on malformed JSON, and never
  silently returns output that fails schema validation.
- Added the **interdisciplinary-literature** agent, between literature and
  hypothesis: identifies adjacent fields whose methods could inform the same
  problem, searches each one concurrently (a `Send` branch per field, reusing
  the literature agent's own arXiv/Semantic Scholar/CORE clients rather than a
  second copy of them), and produces bridge insights connecting what it found
  back to the core question. Merging is deterministic — the literature agent's
  own doi/normalized-title dedupe key, not the model deciding which papers are
  the same — and a bridge insight citing a paper that isn't in the merged pool
  has that id stripped and logged, the same rule the writer agent applies to
  citations.
- Added **ranking** to the hypothesis agent: it still generates and returns
  exactly 3 hypotheses, but now also scores them against each other and names
  a `selected_hypothesis_id`. The winner is derived in Python from whichever
  ranking entry holds rank 1 — the model is never asked to assert it
  separately — and the schema rejects a ranking that isn't a permutation of
  1..3 over exactly the generated ids. The orchestrator uses it to plan and
  execute only the winner, cutting the coder/writer cost of a run from three
  hypotheses to one, while the writer and reviewer still see all 3.
- Added the **experiment-planner** agent: turns each of the hypothesis
  agent's hypotheses into a full, implementation-ready experiment plan
  (concrete, coder-actionable steps — not "analyze the data"), judges
  feasibility against a stated compute/data envelope rather than assuming
  unlimited resources, plans per hypothesis concurrently (they're
  independent), and validates that every input hypothesis id has a
  corresponding output plan before returning.
- Added the **coder** agent: generates a real runnable experiment per
  feasible plan and, resources permitting, actually executes it. Each
  experiment is a single `run.py` rendered from
  `agents/coder/templates/run.py.template` — a fixed metadata block and
  orchestration footer (timing, exception handling, `results.json` writing)
  are NOT model-generated; only `load_data`/`build_model`/`run_experiment`/
  `evaluate` (plus imports/configuration/helpers) are, spliced into the
  template with plain string replacement rather than `str.format()` (LLM
  code routinely contains literal `{`/`}` that would break format-string
  substitution). This removes an entire failure mode the earlier
  four-separate-files design had: there's no cross-file calling convention
  for the model to get slightly wrong on one experiment out of several,
  because the wiring isn't model-generated at all. Also has a syntax-check
  gate before any execution is attempted, and network/GPU detection done at
  runtime rather than assumed. Shared infrastructure across experiments is
  generated once, not per experiment.
- Added the **writer** agent: drafts a full academic paper (PDF), styled
  after a NeurIPS submission (Times font, numbered sections/subsections,
  author-year citations), from the combined output of the upstream agents.
  Every section's prose is model-written, section by section, each call
  grounded in only the relevant slice of upstream data. Two things are
  deliberately *not* left to the model: whether a hypothesis was
  supported/refuted/inconclusive (computed in Python from the coder agent's
  own `meets_success_criteria`/`status`, then handed to the model as a fact
  to write consistently with) and whether a citation is legitimate (the model
  must cite via a `[[cite:PAPER_ID]]`/`[[citet:PAPER_ID]]` marker restricted
  to ids actually in the literature agent's output; a marker naming any other
  id is stripped before rendering and reported in `notes_for_review`, never
  silently printed). Same-author-same-year collisions are disambiguated
  ("2020a"/"2020b") only after every section is drafted, since that's the
  first point the full cited-paper set is known — see
  [citations.py](src/research_pipeline/agents/writer/citations.py)'s module
  docstring. A validation pass after drafting checks every hypothesis appears
  in Results/Discussion and that no `skipped`/`code_generated_not_run`
  experiment's Results text reads as completed work. Renders to PDF via
  `reportlab` — pure Python, so it needs nothing beyond `uv sync` to run on
  Barkla.
- Added the **reviewer** agent and the **writer/reviewer feedback loop**:
  the reviewer checks a drafted paper against the same upstream ground truth
  the writer used, not just for internal consistency. It re-derives its own
  copy of that ground truth (hypothesis verdicts, the literature paper index)
  independently, rather than trusting the writer's JSON summary as fact, and
  reads the *rendered* PDF back in
  ([pdf_reader.py](src/research_pipeline/agents/writer/pdf_reader.py)) to
  check what was actually printed. Citation validity, results-figure
  accuracy, and hypothesis-coverage are checked deterministically — regex/
  dict-diff against the Coder/Literature Agent's own data, no LLM judgment
  involved — while hallucination nuance and honesty of framing/tone/flow go
  to the LLM, one call per section plus one holistic quality-scoring call,
  each grounded in only the ground-truth slice relevant to what it's
  checking. The loop orchestrator
  ([writer_reviewer_loop.py](src/research_pipeline/writer_reviewer_loop.py))
  keeps both agents decoupled from each other — neither knows the other's
  schema — and is the only place that translates a review's issues into
  section-routed revision feedback for `WriterAgent.revise()`. Every
  iteration's PDF and review are kept, not just the final ones, so a run that
  hits `max_iterations` without converging still returns full traceability
  (`review_log.json`) alongside the flattened `unresolved_issues` list —
  never silently accepted as done.
