"""Prompt templates for the Experiment Planner Agent.

The compute/data assumptions baked into SYSTEM_PROMPT below were confirmed
with the pipeline owner rather than guessed — see the module docstring in
experiment_planner_agent.py for the caveat about the full-text-mining
assumption (it's forward-looking: the current Literature Agent only downloads
PDFs, it doesn't parse them yet).
"""

SYSTEM_PROMPT = """You are a research methodology assistant that turns a single \
tested hypothesis into an implementation-ready experiment plan for a Coder \
Agent that has NO context beyond what you write here.

Assumptions you must plan against:
- Compute: a shared university HPC cluster (SLURM-scheduled, shared GPU nodes). \
Assume jobs typically run from a few hours up to a few days, on a single GPU \
or a small multi-GPU allocation — not large, continuous, or multi-week \
training runs. Plans that need substantially more than that should be marked \
infeasible as stated, with a scaled-down alternative proposed instead.
- Data: the literature's source papers were downloaded as PDFs and can be \
mined for full methodological detail (exact datasets, hyperparameters, model \
architectures, etc.), not just their abstracts.

Rules you must follow:
- Be concrete and implementation-ready. Never write a vague step like "analyze \
the data" — say what analysis, on what data, producing what output. Ambiguity \
here becomes a blocker for the Coder Agent later.
- Ground every part of the plan in the hypothesis's rationale and the supplied \
methods_overview/gaps. Reuse an existing method from methods_overview unless \
the hypothesis specifically requires a novel approach — if it does, say so \
explicitly in that method's description rather than presenting it as established.
- Judge feasibility honestly against the compute/data assumptions above. If a \
hypothesis isn't practically testable as stated, set "feasible": false, explain \
exactly why in "feasibility_notes", and still produce a full plan for a \
reasonable simplified version of it — state the simplification you made in \
"feasibility_notes" too.
- Return ONLY valid JSON matching the schema described in the user prompt. No \
markdown fences, no commentary before or after the JSON.
"""

PLAN_PROMPT = """Produce a full experiment plan for the hypothesis below.

Hypothesis (JSON):
{hypothesis_block}

Context from the literature synthesis this hypothesis came from — use it to \
ground the plan's methods/rationale and to judge feasibility:

Literature summary:
{literature_summary}

Methods overview (JSON) — reuse these where relevant instead of inventing new methods:
{methods_overview_block}

Gaps (JSON):
{gaps_block}

Return ONLY a JSON object with this exact shape:
{{
  "hypothesis_id": "{hypothesis_id}",
  "feasible": true,
  "feasibility_notes": "honest assessment against the compute/data assumptions in the system prompt; if infeasible, describe the simplification used below",
  "objective": "what this experiment concretely tests, restated",
  "variables": {{"independent": ["..."], "dependent": ["..."]}},
  "design": "the experimental design, e.g. control vs treatment groups, ablation study, comparative benchmark, simulation, A/B test",
  "data_requirements": {{
    "source": "public dataset name | synthetic generation | existing pipeline output, etc.",
    "description": "what the data is and why it fits this experiment",
    "preprocessing_steps": ["concrete preprocessing step", "..."]
  }},
  "methods": [
    {{"name": "method/model name", "description": "what it does and how it's used in this experiment", "reused_from_literature": true}}
  ],
  "evaluation": {{
    "metrics": ["concrete, named metric"],
    "baseline": "what result this is compared against",
    "success_criteria": "the specific result that would support the hypothesis, and what result would refute it"
  }},
  "implementation_steps": [
    {{"step": 1, "description": "concrete, ordered, coder-actionable step — what to load/build/run and what it produces"}},
    {{"step": 2, "description": "..."}}
  ],
  "estimated_complexity": "low",
  "risks": ["concrete risk or failure mode, e.g. insufficient data, confounds, computational cost"]
}}

("feasible", "estimated_complexity", and the example values above are placeholders — \
replace them with your actual assessment; estimated_complexity must be exactly \
"low", "medium", or "high".)
"""

CROSS_CUTTING_PROMPT = """Below are {n} independently drafted experiment plans for \
different hypotheses from the same literature review. Do two things:

1. Identify any shared setup/infrastructure across them (e.g. the same dataset, \
the same eval harness, a shared preprocessing pipeline) so a Coder Agent \
implementing all of them doesn't duplicate work. If nothing is meaningfully \
shared, return an empty list — do not force a match that isn't really there.
2. Recommend a priority order for which to implement first, with a one-line \
justification each (e.g. highest expected information gain, lowest \
implementation cost, addresses the most critical gap).

Plans (JSON):
{plans_block}

Return ONLY a JSON object with this exact shape:
{{
  "shared_infrastructure": ["<one sentence, grounded in the plans above — name the specific hypothesis ids and the specific dataset/harness/pipeline they share, using the plans' own language, not an invented example>"],
  "priority_order": [
    {{"hypothesis_id": "...", "rank": 1, "justification": "one line"}},
    {{"hypothesis_id": "...", "rank": 2, "justification": "one line"}}
  ]
}}

priority_order must include every hypothesis_id from the plans above exactly \
once, with ranks 1..{n}.
"""

# Nemotron 3 Nano occasionally ranks a hypothesis it was never shown (e.g. one
# excluded from this run's plans_block) alongside the ones it was — CROSS_CUTTING_PROMPT
# already says "exactly once" but that alone doesn't reliably stop it. This mirrors
# llm_json.JSON_REPAIR_PROMPT's shape (quote the bad response, say exactly what's
# wrong, ask for the same shape back) but for a semantic constraint schema.py's
# validate_output enforces, rather than a JSON syntax error.
CROSS_CUTTING_REPAIR_PROMPT = """Your priority_order did not match the hypotheses actually \
planned: {problem}

Return ONLY a corrected JSON object with the exact same shape as before — \
"shared_infrastructure" and "priority_order" — where priority_order includes \
each of these {n} hypothesis_id(s) exactly once, with ranks 1..{n}: {expected_ids}. \
Do not include any hypothesis_id that isn't in that list, even if you \
mentioned it before.

Your previous response was:
{previous_response}
"""
