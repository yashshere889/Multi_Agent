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

from typing import Dict, List, TypedDict


class CoderState(TypedDict, total=False):
    # Input — the Experiment Planner Agent's output dict, exactly as handed to run().
    planner_output: dict

    # validate_input
    expected_ids: List[str]
    ordered_plans: List[dict]  # plans sorted by the planner's priority_order

    # probe_environment — probed once per run, never hardcoded (the same code
    # runs on a laptop, a Kaggle notebook, and a Barkla compute node).
    network_available: bool
    gpu_available: bool

    # setup_shared_infrastructure. `shared_dir` is a str (not a Path) because
    # every value in state is serialized by the checkpointer, and it is only
    # ever used as the output's "shared_infrastructure_path" anyway.
    shared_dir: str
    shared_files: Dict[str, str]

    # The sequential per-plan loop. `plan_index` is the cursor into
    # `ordered_plans`; `experiments` accumulates one finished entry per plan,
    # appended explicitly (no reducer — see the module docstring).
    plan_index: int
    experiments: List[dict]

    # Mirrored from CoderAgent._slurm_jobs_submitted after each attempt, so the
    # gated counter is visible in a checkpoint/trace. Not the gate itself.
    slurm_jobs_submitted: int

    # Per-plan working set, reset by process_current_plan each time the outer
    # loop advances.
    current_plan: dict
    current_experiment_dir: str  # str for the same serialization reason as shared_dir
    current_generation: dict  # run_py_sections / assumptions_made / needs_gpu / requirements_txt / readme
    current_fix_history: List[dict]
    current_attempt: int  # 0-based, matching the old `for attempt in range(max_fix_attempts + 1)`
    current_outcome: dict  # {"result": ...} or {"error_source", "error_text"}

    # assemble_and_validate — the schema-valid dict run() returns
    result: dict
