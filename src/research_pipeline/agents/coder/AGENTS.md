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
| `sandbox.py` | Execution primitives: env probes, `uv venv` provisioning, subprocess running, `compile_check`, `check_undefined_names`, `static_safety_check`, `check_data_fallback`, `check_required_function_names`, template rendering (`render_experiment_with_spans` and its line map). No LLM calls, no settings reads — unit-testable anywhere. |
| `huggingface_client.py` | Hub search + Dataset Viewer REST lookup, so a generated experiment can read real rows instead of inventing data. Every failure degrades to `None`; never raises. |
| `slurm_submit.py` | `squeue`/`sbatch`/`sacct` shell-outs. Split from `sandbox.py` because those binaries only exist on a cluster; `sandbox.py` must stay runnable on a laptop. |
| `reconcile.py` | The other half of submission: asks `sacct` what became of the jobs a previous run recorded, and imports a finished job's `results.json` back into its summary (`submitted_to_slurm` -> `completed`/`slurm_job_failed`). A separate pass, not a wait — see its module docstring. No LLM. |
| `diagnose.py` | `classify_execution_failure` — what *kind* of failure a non-zero exit was. Pure text in, route out; no LLM, no filesystem, no network. |
| `repair.py` | The repairs that need no model call: `downscale` (halve cost knobs), `smoke_variant` (pin them to the floor for the pre-run) and `install_for` (install a missing import). Owns the `PRECISION_KNOBS`/`MEASUREMENT_KNOBS` split that decides whether a downscaled run keeps its verdict. Reads no settings, same rule as `sandbox.py`. |
| `provenance.py` | Resolves each declared data input to real/surrogate, and withholds the hypothesis verdict when any is synthetic. No LLM. |
| `compute_provenance.py` | The same withholding, asked of the compute instead of the inputs: records which cost knobs `repair.downscale` had to shrink, and withholds the verdict when shrinking one changed what the experiment measures. No LLM. |
| `prompts.py` | All prompt templates. |
| `starters.py` | The pre-validated starter-program library: `STARTERS` (one hand-authored, stdlib-only worked example per ML/NLP task shape) and `select_starter(plan)`, a deterministic keyword match with no LLM call. |
| `templates/run.py.template` | The fixed experiment scaffold — metadata, the runtime-support block (`logger`, `log_progress`, `begin_checkpoint`/`finish_checkpoint`/`resume_checkpoint`, the SIGTERM handler) and the orchestration footer that writes `results.json`. Not model-generated. |
| `templates/run.sbatch.template` | Barkla-shaped SLURM script — `--requeue`, `--open-mode=append`, and `python run.py --resume`, so a preempted job continues rather than restarting. |
| `templates/starters/*.sections` | The starter library's content, one `.sections` file per archetype, in `llm_sections.py`'s own delimited format — parsed by the same `parse_sections` the model's responses are parsed by. |

## Conventions

- **Constructor injection, not monkeypatching.** `chat_model`, `experiments_dir`, `output_dir`,
  `network_check`, `gpu_check`, `max_fix_attempts`, `max_structural_retries`,
  `huggingface_lookup_fn` are all constructor args. Tests pass `network_check=lambda: False, gpu_check=lambda: False` rather than patching
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
- **`render_experiment_template` delegates to `render_experiment_with_spans`.**
  There is one splicing implementation on purpose: the span map describes the
  file the renderer returns, and two implementations agreeing only by inspection
  would drift. The metadata tokens are substituted first, over the whole
  template, and always to a `repr()` — always a single line — so they can never
  shift a line number; only the seven agent tokens (each alone on its own line in
  the template, which is what makes the spans exact) are spliced line-wise. A test
  asserts the two entry points return identical text.
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
- **The template hands generated code four things it must not reimplement.**
  `log_progress(**fields)` (one flushed JSON line per epoch into `progress.jsonl` — the only
  trace a killed job leaves, since `results.json` is deliberately never written for an
  incomplete run), and `begin_checkpoint()`/`finish_checkpoint(tmp)`/`resume_checkpoint()`.
  The checkpoint trio is split three ways for a reason: only the generated code knows how to
  serialize its own model (torch.save, np.save, joblib), and only the template can guarantee a
  job killed mid-write doesn't destroy the last good checkpoint — so the template hands out a
  `.tmp` path and does the `os.replace`. It cannot use `pickle` itself: `pickle.loads?` is in
  `DANGEROUS_PATTERNS` and `static_safety_check` runs over the whole *rendered* run.py, template
  included, so a pickling template would fail its own safety gate on every experiment.
  `resume_checkpoint()` returns None unless `--resume` was passed *and* a checkpoint exists,
  which is what makes `python run.py --resume` correct as the only line in `run.sbatch`.
- **`build_model` takes the data.** `REQUIRED_FUNCTIONS` in `sandbox.py` carries an arity
  alongside each name and `_accepts_positional` enforces it, because `def build_model():`
  compiles, defines the right name, passes every other check, and only dies on a TypeError once
  a venv has been provisioned. Extra parameters with defaults and `*args` pass — the question is
  "can the orchestration call this", not "does the signature match exactly".
- **Three static checks run on the rendered `run.py`, in order, and they answer
  different questions.** `lenient_compile_check` asks "does it parse";
  `check_undefined_names` asks "does it bind every name it uses" (pyflakes, not a
  hand-rolled AST walk — scope handling is exactly where a home-grown version
  produces the false positives that would spend fix attempts on correct code, and
  only `UndefinedName`/`UndefinedLocal` are reported, never style findings);
  `static_safety_check` asks "does it do anything it shouldn't". The pyflakes
  import is guarded so a venv synced before it was declared loses this one check
  rather than the whole agent.
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
- **The first execution of an experiment is usually a shrunken one.**
  `_smoke_failure` writes `run_smoke.py` — the same rendered code with every
  `repair.DOWNSCALE_KNOBS` entry pinned to its floor — and runs it under
  `CODER_SMOKE_TIMEOUT_SECONDS` before the real run. The asymmetry is the design:
  it can end an attempt early but can never let one through, so an experiment
  that passes it is still executed at full size and the smoke run's own
  `results.json` is deleted rather than read. Three guards keep it from failing
  correct code — it is skipped when `smoke_variant` found nothing to shrink (the
  smoke run would then *be* the real run); an env or resource failure falls
  through to the execution loop that owns those repairs; and a code failure is
  only believed when `diagnose.is_scale_independent` says the exception could not
  have been produced by the shrinking itself. Keep `SCALE_INDEPENDENT_EXCEPTIONS`
  conservative: `KeyError`/`IndexError`/`ValueError` are all things a small sample
  provokes, and a member added there turns a correct experiment into a spent fix
  attempt.
- **A failing check names the section it came from wherever it can, and the
  regeneration asks only for that.** `_target_sections` returns either a subset
  of `prompts.RUN_PY_SECTION_NAMES` or `None` for "all of them" (the behaviour
  every fix had before this existed). Three sources feed it: a static table
  (`_SECTIONS_BY_ERROR_SOURCE`, for checks that run against one section's own
  source), the finding text (`_sections_mentioned_in`, for the two structural
  checks that open each finding with its section name), and a line number mapped
  through `sandbox.render_experiment_with_spans`'s span map (`compile_check`,
  `undefined_name`). `_ALWAYS_REGENERATED` — imports/configuration/helpers — is
  added to every targeted set, because a fix routinely needs a new import and
  those three are short. A set covering every code section collapses back to
  `None` rather than being spelled out. The merge happens in
  `_assemble_generation(sections, previous=...)`: fields the model wasn't asked
  for come from the previous generation, and with `previous=None` that path is
  byte-for-byte what it was.
- **`give_up` reports the furthest attempt, not the last one.** A fix loop is not
  monotonic, and `_best_candidate` ranks candidates by `_ERROR_STAGE_ORDER` — the
  same ordering `_cleared_previous_error` already uses, deliberately not a second
  definition of "better". When an earlier attempt outranks the final one,
  `_restore_attempt` puts that snapshot's `run.py`/`requirements.txt` back and its
  `results.json` (or removes the final attempt's, which was written by code no
  longer on disk), and the result carries `reported_attempt`. Ties go to the final
  attempt, so a run that never regressed is untouched.
- **There are two regeneration budgets, and which one a failure draws on is
  decided by whether anything was learned.** `_STRUCTURAL_ERROR_SOURCES` —
  `invalid_format`, `missing_sections`, `missing_required_function`,
  `empty_body` — is the model failing to return a program at all: nothing was
  rendered, provisioned or executed, so those draw on
  `CODER_MAX_STRUCTURAL_RETRIES` instead of `max_fix_attempts`, which exists for
  debugging code. `compile_check` is deliberately **not** in the set: a syntax
  error is a real defect in a real answer, and it is the failure most likely to
  repeat. Exhausting the structural budget **ends the plan** — it does not fall
  through to the fix budget. Both readings were implemented independently (see
  `barkla-wip/coder-format-retries`, which bounded `invalid_format` alone) and
  this is the reconciliation: the identical-failure stop cannot bound a
  fall-through, because an `invalid_format` summary embeds
  `Raw response: <500 chars of the model's own output>` and `_failure_signature`
  normalises numbers, paths and addresses but not prose — so malformed responses
  that differ read as different failures and the streak never fires. Falling
  through would spend `max_structural_retries + max_fix_attempts` regenerations
  producing nothing. `0` means off, not "give up immediately": with no separate
  budget a structural failure costs a fix attempt exactly as it did before the
  split. Two more consequences to keep in mind: `fix_history` entries are
  numbered by their own ordinal (`len(fix_history) + 1`), never by
  `current_attempt`, or two entries and two snapshot directories would collide
  once the counters diverge; and `_recursion_limit_for` takes the structural
  budget too, or a plan that spends it stops on the recursion limit instead of
  on a check's verdict.
- **Never `Path.resolve()` an experiment interpreter — use `_interpreter_path`.** A venv's
  `bin/python` is a symlink to the base interpreter, and resolving it hands back that base
  interpreter, which cannot see anything installed into the venv. It is a venv-destroying operation
  dressed up as path normalisation, and it is what ended Barkla jobs 10410771 and 10410847:
  `uv pip install --python <venv>/bin/python numpy` reports success, the package really is in the
  venv's site-packages, and the next `import numpy` by the *resolved* path raises ModuleNotFoundError.
  Measured on Barkla: `./.venv/bin/python -c "import numpy"` → OK, `$(resolve …) -c "import numpy"` →
  ModuleNotFoundError. `os.path.abspath` gives the absolute path the subprocesses need (their `cwd`
  is the experiment dir, and `CODER_EXPERIMENTS_DIR` defaults to a relative `experiments`) without
  following symlinks. `module_importable` and `run_experiment` both go through the helper and must
  keep agreeing — when they disagree, a repair that worked reads as "installed but still not
  importable" and ends the attempt.
- **Env-provisioning failures are not retried through the fix loop.** A missing package or
  unreachable index isn't something regenerating code can fix, so it returns a terminal
  `code_generated_not_run` result directly.
- **An execution failure is classified before it is repaired, and three kinds never reach the
  model.** `_attempt_once` loops around `sandbox.run_experiment`: a `missing_dependency` is
  installed and the *unchanged* code re-run, a `resource_limit` has its own cost knobs halved
  by `repair.downscale` and is re-run, and one narrow shape of `obsolete_dependency` — pandas 3's
  removed `.fillna(method=)` — is rewritten by `repair.patch_removed_pandas_fillna` and re-run,
  bounded by `_MAX_API_PATCHES`. None spends a fix attempt — `CODER_MAX_ENV_REPAIRS` bounds
  installs separately — because none was a defect in the generated source. This is the fix for a
  production run that spent all three fix attempts regenerating code against
  `ModuleNotFoundError: No module named 'pandas'`.
- **The removed-API patch is keyed on `error_source`, not on a route, and that is deliberate.**
  Most of `obsolete_dependency` — a dead import, `DataFrame.append` — genuinely needs a model to
  rewrite, so the category's route stays `ROUTE_REGENERATE`; the patcher is the narrow
  deterministic case *inside* it, exactly as `downscale` is the narrow case inside
  `resource_limit`. `patch_removed_pandas_fillna` covers two shapes and refuses everything else:
  the expression form is a pure syntax swap, while the inplace form is rewritten only when the
  whole line is a bare `<dotted.name>.fillna(..., inplace=True)`, because reassigning anything
  else could turn a mutation into a silent no-op — worse than the TypeError it replaces, in a
  pipeline whose entire point is not reporting numbers that were never computed.
- **`_smoke_failure` must fall through for anything the execution loop can repair itself.**
  It already does for `ROUTE_ENV` and `ROUTE_DOWNSCALE`; `diagnose.removed_api` is the third and
  is not optional bookkeeping. A removed-API failure raises `TypeError`, which
  `is_scale_independent` treats as a real defect, so without that line the smoke run reports it,
  the fix loop takes over, and the patcher above becomes unreachable in the one case it was
  written for. A test asserts the execution order (`run_smoke.py`, `run.py`, `run.py`) and fails
  if the fall-through is removed.
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
  The stop additionally skips when the newest `fix_history` entry is `resolved`: that list
  describes the attempts *before* the current one, so its streak is stale by exactly one, and a
  regeneration that just cleared the failure the streak is made of has converged whatever the
  three before it did. Only reachable since structural failures got their own budget — until then
  the fix budget ran out on the very same step, so the staleness never showed.
- **The provenance gate returns the string `"unknown"`, never `False`.** `writer_agent.compute_hypothesis_verdict`
  maps `False` to **"refuted"** and `"unknown"` to "inconclusive", so returning `False` for a run on
  synthesized data would have the paper publish a refutation of a hypothesis that was never tested.
  A test asserts this at the Writer end, not just the Coder's field.
- **A Hugging Face dataset counts as a real input only when the rendered `run.py` names it.** It is
  offered, not imposed — the model may decline it and say why in `assumptions_made`, which
  `check_hf_dataset_usage` accepts — and treating an offered-but-declined dataset as evidence is
  exactly the over-claim the gate exists to prevent. `_reads_dataset` matches the raw id *and* the
  percent-encoded one, because the prompt hands over a rows URL that encodes the namespace slash;
  it must keep agreeing with `check_hf_dataset_usage`, which checks the same two forms. The id key
  is `dataset_id` — it was `dataset` here once, which meant the branch never fired and a genuinely
  fetched dataset was still scored a surrogate.
- **`_provenance_for(run_py=None)` means "prompt time", and that is not the same as `run_py=""`.**
  With no code yet, an offered dataset is the input the model is *being asked* to read and is listed
  as real. Calling it a surrogate there is not caution but a contradiction: `prompt_block` would
  order a `synthesize_` generator in the same prompt where `_hf_dataset_block` introduces the
  dataset as real, and the model does as it is told. Once `run.py` exists the question becomes
  whether the code that got written actually reads it.
- **The two post-run provenance passes run in this order and not the other.**
  `verify_downloads_used` first — downgrade any declared download the code never fetches — then
  `supersede_unresolved`, which lets what remains answer requirements no source could be found for.
  Superseding first would let an *unfetched* declaration stand in for one. `supersede_unresolved`
  touches only `unresolved` entries: a restricted or credentialed source names real data that
  specifically was not obtained, and no amount of other data answers it.
- **`_rows_url` has one definition for a reason.** The prompt block hands that URL to the model and
  `_provenance_for` records it as the input's `uri`, which `verify_downloads_used` matches on by
  *host* — a uri built differently, or left empty, silently stops vouching for a dataset the code
  really did fetch.
- **`CODER_REQUIRE_REAL_DATA` (default false) is a policy gate, not a repair.** It skips a plan whose
  every input would be a surrogate before any codegen call, rather than generating, running and
  reporting it inconclusive. Routed after `search_hf_dataset` because the lookup is the last thing
  that can turn a surrogate into a real input, and `skip_no_real_data` is a per-plan exit like
  `finalize`/`give_up` — it costs fewer super-steps than the path it replaces, so the recursion
  limit is unchanged.
- **`VALID_ERROR_SOURCES` (`schema.py`) and `_ERROR_STAGE_ORDER` (`coder_agent.py`) must stay in
  sync.** Same seventeen members; the list additionally encodes check *order*, which
  `_cleared_previous_error` uses to decide whether a regeneration made progress — and which
  `_best_candidate` uses to decide which attempt to leave on disk. A new failure
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

- `pyflakes` is a base dependency (not an extra) because `check_undefined_names` uses it as a
  library; it is listed in the mypy hook's `additional_dependencies` too.
- The hooks auto-fix (`ruff --fix`, `ruff format`) and mypy blocks on real type errors. Code is
  formatted to 100 columns; `E501` is off so long explanatory comments aren't force-rewrapped.
- mypy is gradual (`check_untyped_defs`, not `strict`) and `follow_imports = "silent"` keeps it
  from chasing imports into the still-unannotated sibling agents.
- The mypy hook runs in pre-commit's own isolated venv, so this module's third-party imports are
  listed under `additional_dependencies` in `.pre-commit-config.yaml`. Add to that list if this
  package ever imports something new, or the hook will silently type it as `Any` and pass code
  that `uv run mypy` rejects.
