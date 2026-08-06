# research-pipeline

A multi-agent research pipeline. Currently ships six agents, each usable
standalone and chained by data shape rather than by coupling to each other
(see "Chaining agents individually" below); a LangGraph orchestrator runs all
six end to end in one call (see "Running the whole pipeline"):

- **literature** — searches arXiv + Semantic Scholar, dedupes, downloads PDFs.
  Built as a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`
  because it genuinely benefits from graph fan-out/fan-in (two independent
  HTTP APIs queried in parallel).
- **hypothesis** — takes the literature agent's papers, synthesizes the body
  of literature, and produces exactly 3 grounded, testable hypotheses as JSON.
  Built as a plain Python class/function instead of a graph — its "map over
  batches, then reduce" flow is naturally sequential, so a graph would add
  ceremony without buying anything. See its docstring
  ([hypothesis_agent.py](src/research_pipeline/agents/hypothesis/hypothesis_agent.py))
  for the full input/output contract.
- **experiment-planner** — takes the hypothesis agent's 3 hypotheses and turns
  each into an implementation-ready experiment plan (feasibility, design,
  data requirements, methods, evaluation, step-by-step implementation plan),
  plus shared-infrastructure notes and a priority order across the 3. Also a
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
- **writer** — takes the combined output of all four agents above and drafts
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

All six are valid patterns for a new agent — pick whichever fits the
agent's control flow, not necessarily the literature agent's graph shape.

## Layout

```
src/research_pipeline/
├── config.py              # env-var settings (LLM endpoint, API keys, defaults)
├── llm.py                 # shared ChatOpenAI factory
├── llm_json.py            # shared "call the model, expect JSON, retry once on bad JSON" helper
├── cli.py                 # `research-pipeline <agent> ...` entry point
├── writer_reviewer_loop.py # orchestrator: draft -> review -> revise -> re-review loop (not an agent itself)
├── orchestrator/          # LangGraph orchestrator running all six agents end to end (not an agent itself)
│   ├── state.py            # PipelineState — one key per stage's full output dict
│   ├── nodes.py            # thin nodes wrapping each agent's existing entry point
│   └── graph.py            # StateGraph wiring, incl. the writer/reviewer cycle + compile()
└── agents/
    ├── literature/         # LangGraph StateGraph agent
    │   ├── state.py         # graph state schema (TypedDict)
    │   ├── clients.py        # arXiv / Semantic Scholar HTTP clients (no LangGraph coupling)
    │   ├── nodes.py          # LangGraph node functions
    │   └── graph.py          # StateGraph wiring + compile()
    ├── hypothesis/         # plain-callable agent
    │   ├── schema.py          # output contract (TypedDicts) + validate_output()
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

`orchestrator/` wires all six agents into a single LangGraph — Literature →
Hypothesis → Experiment Planner → Coder → Writer ⇄ Reviewer → finalize — so a
full run is one call instead of six hand-chained ones:

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

Optional inputs, all defaulting to the agents' own settings:
`max_results_per_query`, `download_dir`, `output_dir`, `max_iterations`,
`quality_threshold`. Pass preconfigured `writer`/`reviewer` agents (e.g.
sharing one `chat_model`) via `config["configurable"]` rather than the state,
which is checkpointed and so must stay serializable.

### Chaining agents individually (Literature → Hypothesis → Experiment Planner → Coder → Writer ⇄ Reviewer)

Each agent's entry point takes a plain `dict`/`list[dict]` shaped like the
previous agent's *output*, never the previous agent's internal state — so
every link in the chain works identically in-process or decoupled via disk:

```python
# in-process: no serialization round-trip
lit_result = literature_agent.invoke({...})
hyp_result = run_hypothesis_agent(lit_result["merged_papers"], research_question=lit_result["research_question"])
plan_result = run_experiment_planner_agent(hyp_result)
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
cp .env.example .env   # then fill in LLM_BASE_URL / SEMANTIC_SCHOLAR_API_KEY
```

### LLM server

Every agent is LLM-powered, and they all share one model:
[NVIDIA Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
(`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`), served by vLLM. The code
reaches it as a plain OpenAI-compatible chat endpoint via `LLM_BASE_URL`, so
any other such server (LM Studio, Ollama's OpenAI-compat mode, `llama-server`)
still works by changing `LLM_BASE_URL`/`LLM_MODEL` — nothing is Kaggle- or
llama.cpp-specific.

`LLM_MODEL` must match the id the server advertises. The SLURM template passes
no `--served-model-name`, so that's the full HF repo id; the model card's own
example aliases it to `model`, in which case set `LLM_MODEL=model`.

**Reasoning mode.** Nemotron 3 Nano thinks before answering, emitting a
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

**On Barkla:** launch the server as a SLURM job (see
[`scripts/slurm/run_llm_server.sbatch`](scripts/slurm/run_llm_server.sbatch) —
adjust the partition/module TODOs for your account first), find the compute
node it lands on with `squeue -u $USER`, then SSH-tunnel to it:

```bash
ssh -L 8080:<compute-node-hostname>:8080 <user>@barkla.liverpool.ac.uk
```

and leave `LLM_BASE_URL=http://localhost:8080/v1` in `.env`.

### Semantic Scholar

Get a key at https://www.semanticscholar.org/product/api#api-key and set
`SEMANTIC_SCHOLAR_API_KEY` in `.env`. Without it, Semantic Scholar search is
skipped (logged as a warning) rather than failing the whole run — unauthenticated
requests to that API are aggressively rate-limited / rejected with 403s.

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

The per-agent subcommands below run a single stage — useful for iterating on
one agent, or resuming from a previous run's JSON.

```bash
uv run research-pipeline literature "recent approaches to reducing hallucination in RAG systems" \
    --max-results 5 \
    --download-dir papers
```

This searches arXiv + Semantic Scholar, dedupes by DOI/title, downloads
available PDFs into `papers/`, and writes `papers/metadata.json`.

```bash
uv run research-pipeline hypothesis --from-file papers/metadata.json --output-dir outputs
```

This synthesizes the paper set, extracts methods/gaps, generates exactly 3
hypotheses, and writes `outputs/hypotheses_<UTC timestamp>.json` — see
[hypothesis_agent.py](src/research_pipeline/agents/hypothesis/hypothesis_agent.py)'s
docstring for the exact output schema.

```bash
uv run research-pipeline experiment-planner --from-file outputs/hypotheses_<timestamp>.json --output-dir outputs
```

This judges feasibility per hypothesis, produces a full implementation-ready
plan for each (objective, variables, design, data requirements, methods,
evaluation, ordered implementation steps, complexity, risks), notes any
shared infrastructure across the 3 plans, ranks a priority order, and writes
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
actually runs it and captures `results.json`. `high`-complexity or
GPU-needing plans get code plus a `run.sbatch` template instead of being run
synchronously; **the agent never submits SLURM jobs itself**, review the
script first. Writes `outputs/coder_agent_summary_<UTC timestamp>.json` — see
[coder_agent.py](src/research_pipeline/agents/coder/coder_agent.py)'s
docstring for the exact output schema and execution model (confirmed with
the pipeline owner, not assumed).

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

### LLM used by the hypothesis / experiment-planner / coder / writer / reviewer agents

All five reuse the same `research_pipeline.llm.get_chat_model()` client as
the literature agent (Nemotron 3 Nano, i.e. whatever `LLM_BASE_URL`/`LLM_MODEL`
in `.env` points at) rather than a separate client — hypothesis/experiment-planner/coder/reviewer
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
  `low`, 300s for `medium`; `high` is never run synchronously at all — see
  `sandbox.py:TIMEOUT_SECONDS_BY_COMPLEXITY`).
- Never submits anything to SLURM itself; `run.sbatch` files are generated
  for a human to review and submit.

## Tests

```bash
uv run pytest
```

## Notes on this version vs. the original notebook

- LLM/Barkla config is env-driven (`.env`) instead of hardcoded/Kaggle-specific.
- Semantic Scholar / arXiv requests retry transient failures (429/5xx) with backoff.
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
- Added the **experiment-planner** agent: turns each of the hypothesis
  agent's 3 hypotheses into a full, implementation-ready experiment plan
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
  author-year citations), from the combined output of the other four agents.
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
