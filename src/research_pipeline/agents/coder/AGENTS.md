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
| `huggingface_client.py` | Hub search/info/card + Dataset Viewer REST (`is-valid`, `splits`, `first-rows`, `rows`, `size`), plus the download and JSONL normalization. Every failure degrades to `None`/`{}`/`[]`; never raises. |
| `dataset_spec.py` | What an experiment needs from a dataset, before anything is searched for: the spec's schema, `validate_spec` (coerce/clamp/enum-check a model draft), `fallback_spec` (derive one from the plan) and `search_queries`. Pure. |
| `dataset_inspect.py` | Measured statistics over a sampled slice: duplication, emptiness, malformed records, templated repetition, script mix, benchmark contamination, PII. Pure — no network, no model. |
| `dataset_scoring.py` | The rubric: weights, band tables, the deterministic dimensions (license/quality/provenance), the critic vocabulary, and `score()`. The **only** thing here that produces a dataset score. Pure. |
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
  `network_check`, `gpu_check`, `max_fix_attempts`, `huggingface_lookup_fn`, and the six dataset
  seams (`dataset_search_fn`, `dataset_describe_fn`, `dataset_rows_fn`, `dataset_card_fn`,
  `dataset_download_fn`, `dataset_normalize_fn`) are all constructor args. Tests pass
  `network_check=lambda: False, gpu_check=lambda: False` rather than patching
  `sandbox.has_network_access`/`has_gpu`, and fakes for the dataset seams rather than faking a
  dozen HTTP endpoints. `_agent()` defaults every one of them to inert, so a test that turns the
  network on for an unrelated reason cannot reach the real Hub through one it forgot. Keep any new
  environment dependency injectable the same way — and add it to that default.
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
- **Dataset selection is five nodes, before generation, and each answers a different question.**
  `specify_data_requirements` (LLM: what does this plan need?) -> `shortlist_datasets` (HTTP:
  who is worth appraising?) -> `appraise_datasets` (LLM + Python: what does the evidence say and
  what does it score?) -> `critique_leading_dataset` (LLM: can the leader be rejected?) ->
  `acquire_dataset` (disk: download, normalize, record). They are separate because a run whose
  experiments quietly stopped getting real data should be diagnosable *to the step that stopped* —
  "no candidates" and "every candidate was contaminated" are very different runs. The accepted
  dataset is parked in `current_hf_dataset` and threaded into both the codegen *and* the fix prompt
  from state, so a three-attempt fix loop doesn't re-select. Don't fold any of them back into the
  generation call, and don't merge them into one node.
- **None of the five carries a `RetryPolicy`, and that is not an oversight.** Each absorbs its own
  failures in-node and degrades to "no dataset", so there is nothing left for a graph-level retry
  to catch — and a retry that *did* fire would re-spend a model call to reach the same fallback.
  The two internal loops (per-candidate appraisal, the critic's fall-through to the next candidate)
  stay inside their node rather than becoming graph cycles: the iterations are independent and
  bounded by a setting, so a cycle would add super-steps and a recursion-limit term without making
  anything more observable than the scores the node already returns.
- **Hub search queries must be one or two words.** `dataset_spec.MAX_QUERY_WORDS` is 2 because the
  Hub's `search` parameter matches dataset *names*, and against the live API a two-word query
  returns a full page while four words returns nothing at all — measured: `"python code"` -> 20 hits,
  `"instruction code python"` -> 2, `"python instruction code pairs"` -> 0. The first version of
  this pipeline built five-word queries from the spec and pooled **2** candidates where the fixed
  version pools 79. Several short queries pooled beats one precise query that matches nothing, which
  is also why `_shortlist_datasets` gives each query an equal share of the pool instead of a
  first-come cap: one broad query would otherwise fill the pool alone and the differently-angled
  queries would contribute nothing.
- **The model produces labels and evidence; `dataset_scoring.score()` produces the number.** This is
  the load-bearing rule of the whole appraisal, and the same split the Hypothesis Agent's ranking
  makes one level up. `coerce_appraisal` returns a dict with **no score key and no way to add one**,
  so it does not matter whether the model obeyed the prompt's instruction not to return one. License
  (`license_label`, an allowlist match on Hub metadata), quality (`quality_score`, arithmetic over
  `dataset_inspect`'s measured rates) and provenance (`provenance_label`) are never asked of the
  model at all; schema fit is the model's column mapping *verified* against the schema the viewer
  reported, with mappings onto invented column names dropped silently. Only the two relevance labels
  are the model's, and an unrecognised answer falls to the pessimistic band, never the generous one.
  If you add a dimension, add its band table — do not add a free scalar.
- **The critic's findings come from a fixed vocabulary so they can be routed.** `CRITIC_FINDING_CODES`
  is closed; a code outside it is discarded on parse. Membership of `CRITIC_HARD_FAILS` is what makes
  an objection a veto rather than a penalty, and `CRITIC_PENALTIES` sizes the rest — the model never
  sizes its own penalty, which would be the invented score coming back through a different door.
- **The critic may not veto a band Python already scored.** `unusable_schema` is in
  `CONDITIONAL_HARD_FAILS`, not `CRITIC_HARD_FAILS`: it only lands when `schema_fit` was banded
  `incompatible`. The rubric measures schema fit by checking the model's column mapping against the
  real schema, and `partial` deliberately means "usable but imperfect" — letting an adversarial pass
  convert that into a veto, on the same evidence, is the invented-verdict problem one level up from
  the invented score. `description_mismatch` is conditional too, on
  `inspection_corroborates_a_mismatch`: a genuine mismatch leaves a measurable trace (malformed rows,
  empties, an unexpected script, benchmark text), and on a dataset that measured clean the claim is
  an unverifiable recollection — one vetoed a 0.95 candidate on split-adjusted 1980 stock prices the
  model believed were "too low". Uncorroborated it is a 0.15 penalty: enough that a marginal
  candidate fails, not enough to overturn six measured dimensions. Only `evaluation_contamination`
  and `personal_information` veto unconditionally, because using that data at all is the harm.
  Before adding a hard fail, ask what measured signal would corroborate it; if one exists, make the
  veto conditional on it.
- **Run-level dataset budgets are instance attributes, not state.** `_datasets_accepted` and
  `_dataset_bytes_downloaded` are reset per `run()` alongside `_slurm_jobs_submitted` and mirrored
  into state for tracing, for exactly the same reason: each plan must see the true up-to-date total
  before deciding whether it may download, which only the sequential plan loop makes well-defined.
  The run budget is charged from bytes that actually landed on disk, not from the viewer's predicted
  `/size` — the pre-check can be fooled by missing or wrong size metadata, and the run budget is the
  one that has to hold across a whole sweep.
- **`huggingface_hub` and `pyarrow` are imported inside functions, behind the `datasets-download`
  extra.** Same arrangement as the checkpointer's sqlite/postgres backends: a plain `uv sync` must
  stay unaffected, and their absence degrades to the Dataset Viewer REST URL rather than failing.
  They are imported in the *pipeline* process only — putting either into an experiment's throwaway
  venv is the cost this whole module exists to avoid. Both are nonetheless in the mypy hook's
  `additional_dependencies`, or it types the download path as `Any` and passes code `uv run mypy`
  rejects.
- **Starter selection is a pure function, not a node.** Unlike the dataset selection above (real
  network calls, real model calls, and a side effect on disk), `starters.select_starter` is a
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
  exactly the over-claim the gate exists to prevent. "Names it" has four spellings, because there
  are two ways it can be offered: the repo id raw or percent-encoded (the REST path, where the id
  is in the URL), and the normalized JSONL's full path or bare filename (the download path, where
  the repo id never appears in the code at all — insisting on it there would fail correct code).
  A *downloaded* dataset resolves as `real_local` with `local_path` set, not `real_download`: it is
  on disk, pinned to a revision, and a re-run reads the same bytes. This is also the fix for a long
  standing dead branch — `_provenance_for` read `hf_dataset["dataset"]`, a key the client has never
  returned, so a real dataset genuinely read by generated code was never counted and the verdict was
  withheld as `"unknown"` regardless.
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
- `_dataset_stack()` — fakes for all six dataset seams plus the record of what each was asked;
  `_dataset_model()` scripts the three dataset prompt kinds and a codegen in one go. `HF_CANDIDATE`
  is what `describe_candidate` returns, `HF_DATASET_MATCH` the smaller shape the prompt block and
  `check_hf_dataset_usage` read. `_fake_hf()` routes `huggingface_client`'s `requests.get` by URL
  substring for the client's own unit tests. No test touches the real network.
- `tests/test_dataset_scoring.py`, `tests/test_dataset_inspect.py`, `tests/test_dataset_spec.py` —
  pure unit tests for the three pure modules, no fakes at all. The load-bearing one is
  `test_a_confident_claim_on_an_unrelated_dataset_still_scores_near_zero`: an appraisal response
  claiming `"score": 0.99` on an all-`unrelated` dataset must still score ~0. If you change a weight
  or a band, `test_a_hand_computed_example` and the parametrized per-band test are what will tell
  you what else moved.
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
