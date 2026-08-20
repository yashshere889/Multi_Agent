# AGENTS.md — Coder Agent

Instructions for whoever (human or AI) works inside `src/research_pipeline/agents/coder/`.
Pipeline-wide context lives in the repo root [`CLAUDE.md`](../../../../CLAUDE.md) and
[`README.md`](../../../../README.md) — read those for how the six agents chain, and don't
restate them here. This file covers only what's specific to this package.

## What this agent does

Takes the Experiment Planner's output dict, generates one runnable `run.py` per feasible
plan by splicing model-written functions into `templates/run.py.template`, executes what it
safely can here (or writes a `run.sbatch` for a cluster), diagnoses and regenerates code that
fails, and returns a `CoderAgentOutput` the Writer consumes. The full execution model —
which plans run locally vs. get deferred, and why — is in `coder_agent.py`'s module docstring
(`coder_agent.py:1-97`); read it before changing any branch in `_attempt_once`.

## File map

| File | What's in it |
|---|---|
| `coder_agent.py` | `CoderAgent` class: graph-node wrappers (`_node_*`), the private helpers they delegate to, LLM calls, persistence. Plus `run_coder_agent()`. |
| `graph.py` | `build_coder_graph(agent)` — wires the two cycles. Module docstring explains why they're cycles and not `Send` fan-out. |
| `state.py` | `CoderState` TypedDict. Its docstring documents what is deliberately *not* in state. |
| `schema.py` | Output contract + `validate_output()`. Dependency-free, no LLM. |
| `sandbox.py` | Execution primitives: env probes, `uv venv` provisioning, subprocess running, `compile_check`, `static_safety_check`, `check_data_fallback`, `check_required_function_names`, template rendering. No LLM calls, no settings reads — unit-testable anywhere. |
| `huggingface_client.py` | Hub search + Dataset Viewer REST lookup, so a generated experiment can read real rows instead of inventing data. Every failure degrades to `None`; never raises. |
| `slurm_submit.py` | `squeue`/`sbatch` shell-outs. Split from `sandbox.py` because those binaries only exist on a cluster; `sandbox.py` must stay runnable on a laptop. |
| `diagnose.py` | `classify_execution_failure` — what *kind* of failure a non-zero exit was. Pure text in, route out; no LLM, no filesystem, no network. |
| `repair.py` | The repairs that need no model call: `downscale` (halve cost knobs) and `install_for` (install a missing import). Reads no settings, same rule as `sandbox.py`. |
| `provenance.py` | Resolves each declared data input to real/surrogate, and withholds the hypothesis verdict when any is synthetic. No LLM. |
| `prompts.py` | All prompt templates. |
| `starters.py` | The pre-validated starter-program library: `STARTERS` (one hand-authored, stdlib-only worked example per ML/NLP task shape) and `select_starter(plan)`, a deterministic keyword match with no LLM call. |
| `templates/run.py.template` | The fixed experiment scaffold — metadata block + orchestration footer that write `results.json`. Not model-generated. |
| `templates/run.sbatch.template` | Barkla-shaped SLURM script. |
| `templates/starters/*.sections` | The starter library's content, one `.sections` file per archetype, in `llm_sections.py`'s own delimited format — parsed by the same `parse_sections` the model's responses are parsed by. |

## Conventions

- **Constructor injection, not monkeypatching.** `chat_model`, `experiments_dir`, `output_dir`,
  `network_check`, `gpu_check`, `max_fix_attempts`, `huggingface_lookup_fn` are all constructor
  args. Tests pass `network_check=lambda: False, gpu_check=lambda: False` rather than patching
  `sandbox.has_network_access`/`has_gpu`, and a fake `huggingface_lookup_fn` rather than faking
  four HTTP endpoints. Keep any new environment dependency injectable the same way.
- **Code-bearing model responses use `llm_sections.py`, not `llm_json.py`.** `_call_sections`
  (delimited `===BEGIN <field>===` blocks, nothing escaped) is for anything returning source:
  the experiment codegen/fix calls and shared-infrastructure generation. `_call_json` is left
  for short structured responses — currently only the self-review verdict. Never move generated
  code back into a JSON string value; the whole reason that transport was dropped is that
  hand-escaping multi-line Python is something this model reliably gets wrong. Section *names*
  are fixed for the experiment calls (`prompts.EXPERIMENT_FIELD_NAMES`) and discovered from the
  response for shared infra, which names its sections after the files it chose to write.
- **Prompt shapes are generated from `llm_sections.render_section`, not typed out.**
  `prompts.EXPERIMENT_SECTION_PLACEHOLDERS` is the one list of fields; the format example the
  model sees is built from it, so the shape the prompt asks for cannot drift from the shape the
  parser accepts. Add a field there, not in prose.
- **Graph nodes are thin.** Every `_node_*(state) -> dict` returns only the keys it produces and
  delegates to a private helper of the shape it had before the graph existed (`_attempt_once`,
  `_generate_experiment_files`, `_setup_shared_infrastructure`, …). Put new behaviour in a
  helper and call it from a node; don't inline logic into the node.
- **`Settings` is a frozen dataclass.** Tests swap in a copy via
  `dataclasses.replace` + `monkeypatch.setattr` (`tests/test_coder_agent.py:839`). Note it
  patches `coder_agent.settings` specifically — `sandbox.py` and `slurm_submit.py` deliberately
  read no settings at all, so all config reads must stay in `coder_agent.py` or that helper
  stops working.
- **`graph.py` is imported inside `run()`** (`coder_agent.py:214`), not at module scope:
  `graph.py` imports `CoderAgent` for typing, so a top-level import is circular.

## Architecture gotchas

- **The two loops are sequential cycles on purpose — never convert them to `Send` fan-out.**
  `CODER_MAX_SLURM_JOBS_PER_RUN` is gated against a counter each plan must read *after* every
  earlier plan has finished submitting (`coder_agent.py:717`, `:736`). Concurrent plans would
  both read the same pre-submission count and both submit. Full reasoning in `graph.py:16-25`.
- **`state["slurm_jobs_submitted"]` is a trace mirror, not the gate.** The authority is
  `self._slurm_jobs_submitted` on the instance, reset per `run()` call (`coder_agent.py:221`) and
  mirrored into state after each attempt (`coder_agent.py:350`). That's only safe because the
  loop is sequential — see `state.py:31-38`.
- **`run.py`'s content, its requirements path, and the plan's complexity are deliberately kept
  out of `CoderState`** (`state.py:21-29`). They're locals in `_attempt_once`, which the
  `attempt` node calls whole, precisely so the deferred-to-sbatch branch
  (`_handle_unrunnable_locally`: self-review → per-run cap → concurrent cap → submit) stays
  untouched. Don't split that helper up to thread them through state.
- **Two different templating mechanisms, and they are not interchangeable.**
  `run.py.template` is filled by `str.replace()` on `__TOKEN__` markers
  (`sandbox.py:251`, `:274`) because model-generated Python routinely contains literal `{`/`}`
  that `.format()` would choke on. `run.sbatch.template` *does* use `.format()`
  (`sandbox.py:225`), safely, because nothing model-generated is spliced into it. Most of
  `prompts.py` also uses `.format()`, so any literal brace inside a `.format()`-ed template must
  be escaped `{{`/`}}` — adding an unescaped one raises at call time, not at import. The
  delimiter shapes have no braces at all (that's one less thing to get wrong than the JSON
  examples they replaced), and `HF_DATASET_USAGE_NOTE` keeps its literal JSON braces *unescaped*
  precisely because it is appended by `_hf_dataset_block` and never passed through `.format()`.
- **`static_safety_check` is regex-based, not AST-based** — intentional, since it runs against
  code the fix loop rewrites repeatedly and the pattern list is cheap to extend. It is the
  **only** gate on the SLURM auto-submit path, where nothing ever executes locally first. Treat
  additions to `DANGEROUS_PATTERNS` as a security change.
- **`check_data_fallback` in the same file *is* AST-based, and that isn't inconsistency.** Its
  question is "is this read inside a `try` body?", which is structural; a regex can find
  `pd.read_csv(` but can't tell a guarded read from an unguarded one, and flagging every read
  would fail correct code. It is scoped to `load_data`'s own source (a read inside `helpers`
  parses in isolation and would look unguarded even when its caller wraps it), and a function
  that fetches from the Dataset Viewer host is exempt as a whole. Read its docstring before
  widening either rule — a false positive here burns fix attempts on code that was fine.
- **A `CoderAgentError` from generation must never escape a node.** Both
  `_node_generate_experiment_code` and `_node_snapshot_and_regenerate` catch it and convert it to
  `{"generation_error": ...}`, which `_attempt_once` turns into a normal `invalid_format` outcome
  that counts against the fix budget. One unparseable model response must not kill a multi-plan
  run — see commit `558d0d5` and its two regression tests. (That error source was called
  `invalid_json` until generated code moved off the JSON transport.)
- **The dataset lookup happens in its own node, before generation.** `process_current_plan` sets
  up the plan; `search_hf_dataset` looks a dataset up once per plan and parks it in
  `current_hf_dataset`; `generate_experiment_code` writes the first candidate. The result is
  threaded into both the codegen *and* the fix prompt from state, so a three-attempt fix loop
  doesn't re-search. Don't fold the lookup back into the generation call — a run whose
  experiments silently stopped getting real data should be visible in the trace.
- **Starter selection is a pure function, not a node.** Unlike the HF dataset lookup above (a
  real network call with its own cache/retry policy), `starters.select_starter` is a
  deterministic keyword match with no LLM call and no side effect, so it's called directly inside
  `process_current_plan` and stored as `current_starter_id` — no dedicated graph node, no
  recursion-limit bump. `""` means "general" (nothing matched, no worked example shown); a real
  id means `_starter_block` renders that starter's sections into both the codegen and fix
  prompts, same threading pattern as `current_hf_dataset`. The chosen id is also recorded on the
  finished `ExperimentResult` as `starter_used`, for the same traceability reason `fix_history`
  exists.
- **Env-provisioning failures are not retried through the fix loop.** A missing package or
  unreachable index isn't something regenerating code can fix, so it returns a terminal
  `code_generated_not_run` result directly.
- **An execution failure is classified before it is repaired, and two kinds never reach the
  model.** `_attempt_once` loops around `sandbox.run_experiment`: a `missing_dependency` is
  installed and the *unchanged* code re-run, and a `resource_limit` has its own cost knobs halved
  by `repair.downscale` and is re-run. Neither spends a fix attempt — `CODER_MAX_ENV_REPAIRS`
  bounds installs separately — because neither was a defect in the generated source. This is the
  fix for a production run that spent all three fix attempts regenerating code against
  `ModuleNotFoundError: No module named 'pandas'`.
- **`diagnose.IMPORT_TO_PACKAGE` and `diagnose.DEAD_IMPORTS` must not be merged.** The first is
  aliases, where installing the distribution satisfies the import (`sklearn` → `scikit-learn`).
  The second is successors, where nothing can (`pymc3` → `pymc`): installing there reports success
  and changes nothing, and the identical ImportError returns on the re-run. That is why dead
  imports route to regeneration carrying the replacement API, and aliases route to the installer.
- **Two streak functions, and they are not interchangeable.** `_consecutive_error_streak` compares
  `error_source` and feeds `_stuck_block`, which escalates the *prompt* at 2. `_identical_failure_streak`
  compares a normalized `error_summary` too and *stops the plan* at 3. The stricter one is what
  the stop uses on purpose: three different bugs all surface as `run_experiment`, and a model
  fixing one bug into the next is making progress even though the source never changes.
- **The provenance gate returns the string `"unknown"`, never `False`.** `writer_agent.compute_hypothesis_verdict`
  maps `False` to **"refuted"** and `"unknown"` to "inconclusive", so returning `False` for a run on
  synthesized data would have the paper publish a refutation of a hypothesis that was never tested.
  A test asserts this at the Writer end, not just the Coder's field.
- **A Hugging Face dataset counts as a real input only when the rendered `run.py` names it.** It is
  offered, not imposed — the model may decline it and say why in `assumptions_made`, which
  `check_hf_dataset_usage` accepts — and treating an offered-but-declined dataset as evidence is
  exactly the over-claim the gate exists to prevent.
- **`VALID_ERROR_SOURCES` (`schema.py`) and `_ERROR_STAGE_ORDER` (`coder_agent.py`) must stay in
  sync.** Same sixteen members; the list additionally encodes check *order*, which
  `_cleared_previous_error` uses to decide whether a regeneration made progress. A new failure
  path needs an entry in both, in the right position. `test_error_source_lists_stay_in_sync` now
  enforces this rather than leaving it to be remembered.
- **Adding or removing a graph node means updating the recursion-limit constants**
  (`_FIXED_STEPS` / `_STEPS_PER_PLAN` / `_STEPS_PER_FIX_ATTEMPT` in `coder_agent.py`).
  LangGraph's default limit of 25 super-steps is the wrong unit here because both loops are real
  cycles; the limit is derived per run from plan count × fix budget.
- **`count_running_jobs` returns a huge sentinel (`_UNKNOWN_QUEUE_DEPTH`) when `squeue` can't be
  read** (`slurm_submit.py:29`), so a failed probe blocks submission instead of waving it
  through. Don't "fix" it to return 0.

## Testing

`tests/test_coder_agent.py` and `tests/test_slurm_submit.py`. No test hits a real model, a real
cluster, or the network.

- `FakeChatModel` — canned responses keyed by a substring of the prompt.
- `ScriptedChatModel` — responses per *prompt kind*, each kind a list consumed in order
  (last entry repeats). Kinds are detected by marker substrings in `KINDS`, which are literal
  excerpts of `prompts.py` — reword a prompt's opening line and these tests misroute.
- `_agent(tmp_path, model, **kwargs)` — the standard agent under test: tmp dirs, and no network
  or GPU unless a test overrides `network_check`/`gpu_check`.
- `GOOD_SECTIONS` / `_codegen_response()` / `_plan()` / `_planner_output()` — valid fixtures to
  mutate rather than rebuild. `_codegen_response()` builds the *delimited* response format via
  `llm_sections.render_sections`, and is the seam nearly every agent test goes through, so the
  transport is defined in one place; `_unparseable_sections_response()` is its failure-path twin.
- `HF_DATASET_MATCH` + `_recording_lookup()` — a fake `huggingface_lookup_fn` and the queries it
  was asked. `_fake_hf()` routes `huggingface_client`'s `requests.get` by URL substring for the
  client's own unit tests. No test touches the real network.
- `_patch_settings` and the `auto_submit` fixture — the fixture flips
  `CODER_AUTO_SUBMIT_SLURM` on and stubs `slurm_submit.count_running_jobs`/`submit_job`, and
  yields the list of what would have been submitted. Never let a test reach real `sbatch`.

## Dev tooling

This directory (plus its two test files) is the **only** part of the repo currently covered by
ruff + mypy; see `.pre-commit-config.yaml` at the repo root and the `[tool.ruff]`/`[tool.mypy]`
sections in `pyproject.toml`.

```bash
uv run pre-commit run --files src/research_pipeline/agents/coder/*.py   # manual pass
uv run mypy --config-file=pyproject.toml src/research_pipeline/agents/coder
```

- The hooks auto-fix (`ruff --fix`, `ruff format`) and mypy blocks on real type errors. Code is
  formatted to 100 columns; `E501` is off so long explanatory comments aren't force-rewrapped.
- mypy is gradual (`check_untyped_defs`, not `strict`) and `follow_imports = "silent"` keeps it
  from chasing imports into the still-unannotated sibling agents.
- The mypy hook runs in pre-commit's own isolated venv, so this module's third-party imports are
  listed under `additional_dependencies` in `.pre-commit-config.yaml`. Add to that list if this
  package ever imports something new, or the hook will silently type it as `Any` and pass code
  that `uv run mypy` rejects.
