"""Shared state schema for the Coder Agent's LangGraph.

Mirrors agents/reviewer/state.py and agents/writer/state.py: a single TypedDict
threaded through every node, each node returning only the keys it produces.
`total=False` because keys appear progressively as the graph advances — only
`planner_output` exists at entry.

Deliberately **no** `Annotated[..., operator.add]` reducer, and deliberately no
`Send` fan-out (contrast agents/hypothesis/state.py and
agents/experiment_planner/state.py). Plans are processed strictly one at a time
by a node that loops back onto itself, advancing `plan_index` — see graph.py for
why that sequencing is load-bearing rather than stylistic. Because only one plan
is ever in flight, `experiments` is a plain last-value channel that each
finishing node rewrites as `[*state["experiments"], <new entry>]`; ordering is
therefore the priority order the plans were sorted into, never completion order.

The `current_*` keys are the per-plan working set: `process_current_plan` resets
them every time the outer loop advances, and the fix-loop cycle
(`attempt` -> `snapshot_and_regenerate` -> `attempt`) reads and rewrites them.

Two things are deliberately *not* here:

- **The generated `run.py`, its requirements path, and the plan's complexity.**
  They stay local to `CoderAgent._attempt_once`, which the `attempt` node calls
  whole. Keeping that helper intact is what keeps the deferred-to-sbatch path
  (`_handle_unrunnable_locally`: self-review, both SLURM job caps, submission)
  byte-for-byte what it was — splitting it apart to thread those three values
  through state would have bought traceability of a branch at the cost of
  re-implementing the one branch in this agent that must not change.

- **The run-level counters as the *authority* for their caps.** This covers
  `slurm_jobs_submitted` and `datasets_accepted` alike; the SLURM one is spelled
  out below and the dataset download budget works identically.

- **The SLURM per-run submission counter as the *authority* for the cap.**
  `slurm_jobs_submitted` is mirrored into state after every attempt so a
  checkpoint shows the true count at each step, but the gate in
  `_handle_unrunnable_locally` still reads `self._slurm_jobs_submitted` on the
  agent, exactly as before. That is safe precisely because the plan loop is
  sequential: at most one plan is ever being processed, so the instance
  attribute and the state key can never disagree, and each plan still sees the
  up-to-date count before deciding whether to auto-submit.
"""

from __future__ import annotations

from typing import TypedDict


class CoderState(TypedDict, total=False):
    # Input — the Experiment Planner Agent's output dict, exactly as handed to run().
    planner_output: dict

    # validate_input
    expected_ids: list[str]
    ordered_plans: list[dict]  # plans sorted by the planner's priority_order

    # probe_environment — probed once per run, never hardcoded (the same code
    # runs on a laptop, a Kaggle notebook, and a Barkla compute node).
    network_available: bool
    gpu_available: bool

    # setup_shared_infrastructure. `shared_dir` is a str (not a Path) because
    # every value in state is serialized by the checkpointer, and it is only
    # ever used as the output's "shared_infrastructure_path" anyway.
    shared_dir: str
    shared_files: dict[str, str]
    # "" unless shared infrastructure still failed a compile/safety check
    # after its own bounded fix loop — see CoderAgent._setup_shared_infrastructure.
    # Threaded into every experiment's codegen/fix prompt via
    # _shared_infra_block so the model knows not to trust a shared import it
    # can't itself see is broken.
    shared_infra_warning: str

    # The sequential per-plan loop. `plan_index` is the cursor into
    # `ordered_plans`; `experiments` accumulates one finished entry per plan,
    # appended explicitly (no reducer — see the module docstring).
    plan_index: int
    experiments: list[dict]

    # Mirrored from CoderAgent._slurm_jobs_submitted after each attempt, so the
    # gated counter is visible in a checkpoint/trace. Not the gate itself.
    slurm_jobs_submitted: int
    # Same arrangement for the run-level dataset budget: mirrored from
    # CoderAgent._datasets_accepted after each acquisition, never the gate.
    datasets_accepted: int

    # Per-plan working set, reset by process_current_plan each time the outer
    # loop advances.
    current_plan: dict
    current_experiment_dir: str  # str for the same serialization reason as shared_dir
    # The dataset-selection working set, in the order the five nodes produce it.
    # All plain JSON, like everything else here — it's checkpointed.
    #
    # `current_dataset_spec` is dataset_spec.DatasetSpec.to_dict(): what this
    # plan needs, stated before anything is searched for. Empty when the network
    # probe failed or CODER_ENABLE_HF_DATASET_SEARCH is off, which is the single
    # gate every later dataset node no-ops on.
    current_dataset_spec: dict
    # Shortlisted candidates with their viewer-confirmed schema and Hub
    # metadata, best-first by the deterministic prefilter.
    current_dataset_candidates: list[dict]
    # Every appraised candidate as dataset_scoring.to_state(), best-first by
    # computed score. Kept whole rather than reduced to a winner so the trace
    # shows what was rejected in favour of what.
    current_dataset_scores: list[dict]
    # Candidates the critic vetoed or that failed the threshold, with their
    # reasons_for_rejection. Same reasoning: "no dataset was used" and "four
    # datasets were considered and all four were contaminated" are very
    # different runs and should not look identical in a checkpoint.
    current_dataset_rejections: list[dict]
    # The accepted candidate, or {} when nothing cleared the bar. Carries
    # local_path once acquire_dataset has downloaded and normalized it — which
    # is what decides whether the codegen prompt points at a local file or at
    # the Dataset Viewer REST URL. Threaded into both the codegen and the fix
    # prompt, so a three-attempt fix loop doesn't re-select.
    current_hf_dataset: dict
    # The id of the starters.STARTERS entry chosen for this plan by
    # starters.select_starter (a pure function of the plan's own text — no LLM
    # call), or "" for "general" (no match, no worked example shown). Set once
    # per plan in process_current_plan and threaded into both the codegen and
    # the fix prompt from state, same pattern as current_hf_dataset.
    current_starter_id: str
    # run_py_sections / assumptions_made / needs_gpu / requirements_txt / readme
    current_generation: dict
    # The run_py_sections of whatever current_generation *was* right before the
    # most recent snapshot_and_regenerate call replaced it — i.e. the code that
    # produced current_fix_history[-1]'s error. Read by the attempt node once
    # that entry's `resolved` is known, to pair "what was broken" with "what
    # fixed it" for fix_pattern_store.record_fix. Not part of the validated
    # output — see fix_pattern_store.py for why this is checkpointed state
    # rather than a value threaded through Python call args: it has to survive
    # exactly one extra node hop (regenerate -> attempt) the same way every
    # other current_* working-set key does.
    current_broken_sections: dict
    current_fix_history: list[dict]
    current_attempt: int  # 0-based, matching the old `for attempt in range(max_fix_attempts + 1)`
    current_outcome: dict  # {"result": ...} or {"error_source", "error_text"}

    # assemble_and_validate — the schema-valid dict run() returns
    result: dict
