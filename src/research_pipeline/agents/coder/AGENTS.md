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
| `sandbox.py` | Execution primitives: env probes, `uv venv` provisioning, subprocess running, `compile_check`, `static_safety_check`, template rendering. No LLM calls — unit-testable anywhere. |
| `slurm_submit.py` | `squeue`/`sbatch` shell-outs. Split from `sandbox.py` because those binaries only exist on a cluster; `sandbox.py` must stay runnable on a laptop. |
| `prompts.py` | All prompt templates. |
| `templates/run.py.template` | The fixed experiment scaffold — metadata block + orchestration footer that write `results.json`. Not model-generated. |
| `templates/run.sbatch.template` | Barkla-shaped SLURM script. |

## Conventions

- **Constructor injection, not monkeypatching.** `chat_model`, `experiments_dir`, `output_dir`,
  `network_check`, `gpu_check`, `max_fix_attempts` are all constructor args
  (`coder_agent.py:178-201`). Tests pass `network_check=lambda: False, gpu_check=lambda: False`
  rather than patching `sandbox.has_network_access`/`has_gpu`. Keep any new environment
  dependency injectable the same way.
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
  (`sandbox.py:225`), safely, because nothing model-generated is spliced into it. `prompts.py`
  also uses `.format()`, so every literal brace in a prompt's JSON example is escaped `{{`/`}}`
  — adding an unescaped one raises at call time, not at import.
- **`static_safety_check` is regex-based, not AST-based** (`sandbox.py:92-120`) — intentional,
  since it runs against code the fix loop rewrites repeatedly and the pattern list is cheap to
  extend. It is the **only** gate on the SLURM auto-submit path, where nothing ever executes
  locally first. Treat additions to `DANGEROUS_PATTERNS` as a security change.
- **A `CoderAgentError` from generation must never escape a node.** Both
  `_node_process_current_plan` (`coder_agent.py:323`) and `_node_snapshot_and_regenerate`
  (`coder_agent.py:414`) catch it and convert it to `{"generation_error": ...}`, which
  `_attempt_once` (`coder_agent.py:538`) turns into a normal `invalid_json` outcome that counts
  against the fix budget. One unparseable model response must not kill a multi-plan run — see
  commit `558d0d5` and its two regression tests.
- **Env-provisioning failures are not retried through the fix loop** (`coder_agent.py:617-632`).
  A missing package or unreachable index isn't something regenerating code can fix, so it
  returns a terminal `code_generated_not_run` result directly.
- **`VALID_ERROR_SOURCES` (`schema.py:9`) and `_ERROR_STAGE_ORDER` (`coder_agent.py:131`) must
  stay in sync.** Same seven members; the list additionally encodes check *order*, which
  `_cleared_previous_error` (`coder_agent.py:750`) uses to decide whether a regeneration made
  progress. A new failure path needs an entry in both, in the right position.
- **Adding or removing a graph node means updating the recursion-limit constants**
  (`_FIXED_STEPS` / `_STEPS_PER_PLAN` / `_STEPS_PER_FIX_ATTEMPT`, `coder_agent.py:147-149`).
  LangGraph's default limit of 25 super-steps is the wrong unit here because both loops are real
  cycles; the limit is derived per run from plan count × fix budget.
- **`count_running_jobs` returns a huge sentinel (`_UNKNOWN_QUEUE_DEPTH`) when `squeue` can't be
  read** (`slurm_submit.py:29`), so a failed probe blocks submission instead of waving it
  through. Don't "fix" it to return 0.

## Testing

`tests/test_coder_agent.py` and `tests/test_slurm_submit.py`. No test hits a real model, a real
cluster, or the network.

- `FakeChatModel` (`:159`) — canned responses keyed by a substring of the prompt.
- `ScriptedChatModel` (`:596`) — responses per *prompt kind*, each kind a list consumed in order
  (last entry repeats). Kinds are detected by marker substrings in `KINDS`, which are literal
  excerpts of `prompts.py` — reword a prompt's opening line and these tests misroute.
- `_agent(tmp_path, model, **kwargs)` (`:638`) — the standard agent under test: tmp dirs, no
  network, no GPU.
- `GOOD_SECTIONS` / `_codegen_response()` / `_plan()` / `_planner_output()` (`:178-239`) — valid
  fixtures to mutate rather than rebuild.
- `_patch_settings` (`:839`) and the `auto_submit` fixture (`:851`) — the fixture flips
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
