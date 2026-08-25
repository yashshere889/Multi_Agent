"""Prompt templates for the Coder Agent.

Each experiment is a single generated run.py, rendered from
templates/run.py.template: a fixed metadata block and orchestration footer
(not model-generated) wrap four model-generated functions — load_data,
build_model, run_experiment, evaluate — plus imports/configuration/helpers.
See sandbox.render_experiment_template for exactly how the pieces are spliced
together.

Every prompt that asks for **source code** asks for it in llm_sections.py's
delimited format, never as JSON string values: escaping multi-line Python
through JSON is a transport this model reliably gets wrong (see that module's
docstring). The delimiter shapes below are built with
`llm_sections.render_section` rather than typed out as literal text, so the
format the model is shown can't drift from the format the parser accepts.
Prompts that ask only for short structured fields (the self-review verdict)
still use JSON via llm_json.py.
"""

from __future__ import annotations

from research_pipeline.llm_sections import render_section

# The order the model is asked to emit them in, each with the placeholder shown
# in the format example. The first seven are spliced into run.py by
# sandbox.render_experiment_template; the rest are per-experiment metadata.
EXPERIMENT_SECTION_PLACEHOLDERS: list[tuple[str, str]] = [
    ("imports", "<extra import lines — leave this section empty if none are needed>"),
    ("configuration", "<module-level constants — leave this section empty if none are needed>"),
    ("load_data_function", "<the complete `def load_data() -> Any:` function>"),
    ("build_model_function", "<the complete `def build_model() -> Any:` function>"),
    (
        "run_experiment_function",
        "<the complete `def run_experiment(data: Any, model: Any) -> dict[str, Any]:` function>",
    ),
    (
        "evaluate_function",
        "<the complete `def evaluate(experiment_output: dict[str, Any]) -> dict[str, Any]:` function>",
    ),
    ("helpers", "<optional helper functions — leave this section empty if none are needed>"),
    ("readme", "<full README.md contents>"),
    (
        "requirements_txt",
        "<one package name per line — leave this section empty if none are needed>",
    ),
    (
        "assumptions_made",
        "- <one assumption per line, each line starting with a dash — leave this section empty if none were needed>",
    ),
    ("needs_network", "true or false"),
    ("needs_gpu", "true or false"),
]

# What CoderAgent._call_sections requires back, and what it splices where.
EXPERIMENT_FIELD_NAMES: list[str] = [name for name, _ in EXPERIMENT_SECTION_PLACEHOLDERS]
RUN_PY_SECTION_NAMES: tuple[str, ...] = (
    "imports",
    "configuration",
    "load_data_function",
    "build_model_function",
    "run_experiment_function",
    "evaluate_function",
    "helpers",
)


def _section_shape(fields: list[tuple[str, str]]) -> str:
    return "\n\n".join(render_section(name, placeholder) for name, placeholder in fields)


EXPERIMENT_SECTION_SHAPE = _section_shape(EXPERIMENT_SECTION_PLACEHOLDERS)

SECTION_FORMAT_RULES = """Rules for this format — read them, they are not the \
usual JSON rules:
- Write every section's content RAW, exactly as it should appear in the file. \
Never escape anything: a new line is an actual new line (never the two \
characters backslash-n), a backslash is a single backslash, and quotes are \
never escaped. This is NOT JSON, so there is nothing to encode.
- Emit every section listed above, in that order, each with its own BEGIN line \
and END line. A section with nothing in it still needs both marker lines, with \
nothing between them.
- Put nothing outside the markers: no markdown fences, no commentary before the \
first BEGIN line or after the last END line, and nothing between an END line \
and the next BEGIN line.
"""

# The shared-infrastructure call is the one place the section *names* aren't
# fixed in advance: it returns one section per file it decided to write, named
# after that file, so llm_sections.parse_sections discovers them from the
# response instead of being handed a list.
SHARED_INFRA_SECTION_SHAPE = (
    render_section("<module_name>.py", "<full file contents>")
    + "\n\n"
    + render_section("README.md", "<what's in here, and which experiments use each module>")
)

SHARED_INFRA_FORMAT_RULES = """Rules for this format — read them, they are not \
the usual JSON rules:
- One section per file. The section name IS the filename (e.g. \
"===BEGIN data_utils.py===" ... "===END data_utils.py==="), and the same name \
must appear on both marker lines.
- Write each file's content RAW, exactly as it should appear on disk. Never \
escape anything: a new line is an actual new line (never the two characters \
backslash-n), a backslash is a single backslash, and quotes are never escaped. \
This is NOT JSON, so there is nothing to encode.
- Put nothing outside the markers: no markdown fences, no commentary before the \
first BEGIN line or after the last END line, and nothing between an END line \
and the next BEGIN line.
"""

SYSTEM_PROMPT = """You are a research software engineer generating real, runnable \
Python code for a single computational experiment, to be handed off with NO \
further clarification available.

You are NOT writing a whole file. You are filling in specific pieces of a \
fixed template (metadata, timing, error handling, and results.json writing \
are already written and will run exactly as described below) — see the exact \
pieces requested in the user prompt.

Hard rules:
- Write real, complete, runnable code. Never write pseudocode, stubs, \
"# TODO: implement this", a bare `pass`/`...` placeholder body, or a function \
that raises NotImplementedError instead of doing the work. A comment \
describing what the function should do is not a substitute for writing it.
- Prefer the Python standard library. Only depend on a third-party package if \
the experiment genuinely needs it (e.g. numpy/pandas/scikit-learn for the \
specific method called for) — list every such import name, one per line, in \
"requirements_txt", with no version pin unless one is scientifically important.
- Add real error handling and logging (Python's `logging` module, using the \
module-level `logger` the template already defines) so a failure is \
diagnosable from the output, not silent — e.g. a clear message if a data file \
is missing. The template's orchestration will catch any exception you don't \
handle and record it, but don't rely on that as your only error handling.
- If a step is ambiguous or underspecified, make the most reasonable \
engineering decision, implement it, and record exactly what you assumed in \
both the "assumptions_made" section and the README's "Assumptions" \
section. Never silently change the experiment's stated objective, variables, \
or success criteria to make implementation easier — if a genuine \
simplification is unavoidable, say so explicitly rather than pretending it \
wasn't needed.
- Ground methods in the plan's "methods" list. Where a method has \
"reused_from_literature": true, implement it as an established technique \
(say which one, in a comment). Where it's false, implement it as clearly \
novel/experimental code and say so in a comment and the README — don't \
present it as an established method.
- Compute assumption: this runs on a shared university HPC cluster (SLURM, \
shared GPU nodes). Code must be able to run well under the stated timeout on \
modest hardware (a few CPU cores, at most one GPU) — prefer a deliberately \
small default (a data subsample, few iterations/epochs) over code that will \
time out, and note in the README how to scale it up for a full run.
- Return ONLY the answer, in EXACTLY the output format the user prompt \
specifies, with no markdown fences and no commentary before or after it. When \
that format is a delimited section format, write the content of each section \
raw and unescaped — real line breaks, real backslashes, real quotes — because \
it is read verbatim, not decoded.
"""

# Appended after the concrete dataset facts by CoderAgent._hf_dataset_block. Kept
# out of .format() reach on purpose: it contains literal JSON braces describing
# the API's response shape, which would have to be doubled in a format template
# and would then be wrong if this text were ever reused verbatim.
HF_DATASET_USAGE_NOTE = """The response body is JSON shaped like:
  {"num_rows_total": 1234, "rows": [{"row_idx": 0, "row": {"<column>": <value>}}]}
so the records you want are `[entry["row"] for entry in response.json()["rows"]]`. \
Keep `length` at 100 or below per request, and page with `offset` if you need more \
rows — but prefer a deliberately small sample, since this runs under a timeout.

Use this dataset ONLY if it genuinely fits the plan's data_requirements. If it \
doesn't, ignore it completely, generate/synthesize the data the plan describes, \
and say in assumptions_made that you did and why.

Either way, load_data() MUST still work when the fetch fails (no network on the \
run host, a rate limit, the dataset moved): wrap the request in try/except, log \
the failure, and fall back to a small synthesized stand-in dataset in the same \
shape, recording that fallback in assumptions_made and the README. Code that \
assumes the fetch — or a local data file — will simply be there is not \
acceptable.
"""

SHARED_IMPORT_NOTE = """If you need the shared infrastructure above, import it using EXACTLY this \
pattern in your imports:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments._shared import <module_name>
"""

EXPERIMENT_CODEGEN_PROMPT = (
    """Fill in the experiment template below for the plan below.

Experiment plan (JSON):
{plan_block}

{shared_infra_block}

{hf_dataset_block}
{provenance_block}
{starter_block}
Environment notes for wherever this will actually run:
- Network access: {network_status}. {network_note}

The template you are filling in for hypothesis {hypothesis_id} already \
defines (you do NOT write these — they're fixed and will run exactly as \
described):
  - Module-level constants: EXPERIMENT_ID, EXPERIMENT_NAME, HYPOTHESIS \
("{objective}"), DESCRIPTION, BASELINE, SUCCESS_CRITERIA, OUTPUT_DIR, and a \
`logger` you can use.
  - An orchestrator that calls, in order: `data = load_data()`, \
`model = build_model()`, `experiment_output = run_experiment(data, model)`, \
`eval_output = evaluate(experiment_output)` — timing each phase, catching any \
exception, and writing the result to results.json. This is exactly what runs \
as `python run.py`, with this experiment's own directory as the working directory.

You fill in these pieces:
  imports              - extra import lines beyond what the template already \
imports (stdlib json/logging/sys/time/traceback/datetime/Path/typing.Any are \
already imported — don't repeat them). Empty section if nothing extra is needed.
  configuration         - module-level constants for tunable parameters \
(paths, seeds, hyperparameters) — never hard-code magic numbers inside the \
functions below. Empty section if nothing is needed.
  load_data_function     - the COMPLETE function, starting with \
`def load_data() -> Any:` — fetches/generates the data described in \
data_requirements and applies every step in preprocessing_steps. Return \
value can be any structure; it's passed straight into build_model/run_experiment.
  build_model_function    - the COMPLETE function, starting with \
`def build_model() -> Any:` — instantiates/configures the method(s) from \
"methods" that this experiment's design needs. If no model/algorithm object \
makes sense for this experiment (e.g. a pure data-analysis experiment), \
return None and say why in a comment.
  run_experiment_function  - the COMPLETE function, starting with \
`def run_experiment(data: Any, model: Any) -> dict[str, Any]:` — executes the \
design (e.g. control vs treatment, ablation, comparative benchmark, \
simulation) described in "design", following implementation_steps in order. \
Return whatever raw output evaluate() needs (predictions, per-condition \
results, etc.) — must be JSON-serializable (no numpy arrays/tensors; convert \
to plain Python types).
  evaluate_function        - the COMPLETE function, starting with \
`def evaluate(experiment_output: dict[str, Any]) -> dict[str, Any]:` — \
computes every metric in evaluation.metrics from experiment_output, and \
compares against BASELINE/SUCCESS_CRITERIA. The returned dict must include \
your named metric keys (e.g. "accuracy": 0.91) PLUS exactly these two \
reserved keys, which the template's orchestrator pulls out automatically — \
do not write results.json yourself, the template does that:
    "meets_success_criteria": true | false | "unknown"   (use "unknown" only \
if genuinely inconclusive given what could actually be computed here)
    "success_notes": "1-3 sentences on what happened and why"
  helpers                - optional extra helper function(s) used by the \
functions above. Empty section if none are needed.
  readme                  - full README.md contents: which hypothesis this \
tests ({hypothesis_id}: "{objective}"), how to run it (`python run.py`, plus \
how to install requirements.txt if non-empty), what output to expect, how to \
interpret results.json relative to SUCCESS_CRITERIA, and an "Assumptions" \
section listing every engineering decision made for underspecified steps.
  requirements_txt         - third-party packages actually imported anywhere \
above, one per line, or an empty section if none are needed.
  assumptions_made         - one concrete assumption per line, each line \
starting with "- ", for every underspecified step you had to decide. Empty \
section if none were needed.
  needs_network            - "true" or "false": your actual assessment of \
whether this experiment's code needs network access to run.
  needs_gpu                - "true" or "false": your actual assessment of \
whether this experiment's code needs a GPU to run.

Return your answer using EXACTLY this delimited format:

"""
    + EXPERIMENT_SECTION_SHAPE
    + "\n\n"
    + SECTION_FORMAT_RULES
)

EXPERIMENT_CODEGEN_FIX_PROMPT = (
    """The code you generated for hypothesis {hypothesis_id} failed. Fix it.

The experiment plan is unchanged (JSON):
{plan_block}

{shared_infra_block}

{hf_dataset_block}
{provenance_block}
{starter_block}
{fix_pattern_block}
The code sections you produced last time:
{previous_sections_block}

{stuck_block}What went wrong — detected by {error_source}:
{error_text}

Network access on this machine: {network_status}.{network_note}

Diagnose the actual cause and regenerate every section, keeping whatever \
already worked and correcting what caused the failure. Do not merely describe \
the bug in "assumptions_made" — the returned code must actually fix it. If the \
failure was a missing or misnamed dependency, correct requirements_txt to \
match what the code imports. If the failure was a flagged unsafe pattern, \
rewrite that logic so it no longer needs it — do not just move or obfuscate it.

Return every section again, in EXACTLY this delimited format:

"""
    + EXPERIMENT_SECTION_SHAPE
    + "\n\n"
    + SECTION_FORMAT_RULES
)

SHARED_INFRA_FIX_PROMPT = (
    """The shared infrastructure you generated failed a check. Fix it.

It will be imported by multiple experiments, so it must be correct standalone \
— not merely plausible-looking.

Shared infrastructure items (from the Experiment Planner):
{shared_items_block}

Context — the experiment plans that will use this shared code (JSON):
{plans_block}

The files you produced last time:
{previous_files_block}

What went wrong:
{error_text}

Diagnose the actual cause and regenerate every file, keeping whatever already \
worked and correcting what caused the failure. Do not merely describe the bug \
— the returned code must actually fix it.

Return every file again, in EXACTLY this delimited format:

"""
    + SHARED_INFRA_SECTION_SHAPE
    + "\n\n"
    + SHARED_INFRA_FORMAT_RULES
)

EXPERIMENT_SELF_REVIEW_PROMPT = """Review the experiment code below for hypothesis {hypothesis_id} \
BEFORE it is submitted to a shared HPC cluster. It cannot be test-run first — \
this review is the only check it gets, and a bug wastes real GPU allocation on \
a machine other researchers are queueing for.

The experiment plan it must implement (JSON):
{plan_block}

The generated run.py in full:
{code_block}

Read it critically against the plan. Flag anything that would crash, hang, \
silently produce meaningless numbers, or fail to test what the plan actually \
asked for — undefined names, wrong shapes, a metric that doesn't match \
evaluation.metrics, an unbounded loop, a data path that won't exist on the \
cluster, a hardcoded local assumption. Be specific about what is wrong and \
where; do not restate what the code does correctly.

Return ONLY a JSON object with this exact shape:
{{
  "looks_correct": true,
  "concerns": ["specific problem and where it is — empty list if none"]
}}
"""

SHARED_INFRA_PROMPT = (
    """The following shared setup/infrastructure was identified as common across \
multiple experiments in this pipeline — implement it ONCE here so individual \
experiments can import it instead of duplicating it.

Shared infrastructure items (from the Experiment Planner):
{shared_items_block}

Context — the experiment plans that will use this shared code (JSON):
{plans_block}

Write one or more small, focused Python modules (e.g. a data-loading module, \
an eval-harness module) that cover the items above — real, runnable code, \
not stubs. Each will live at experiments/_shared/<filename>.py and be \
imported by experiment scripts as `from experiments._shared import \
<module_name>` (each experiment script adds the repo root to sys.path before \
importing — you don't need to handle that here).

Return the files using EXACTLY this delimited format:

"""
    + SHARED_INFRA_SECTION_SHAPE
    + "\n\n"
    + SHARED_INFRA_FORMAT_RULES
)


# ---------------------------------------------------------------------------
# Dataset selection.
#
# Three calls, in the order the graph makes them: state the requirement, gather
# evidence against it, then try to reject the leader. All three ask for short
# structured fields, so they go through _call_json, not _call_sections.
#
# None of them asks for a score. dataset_scoring.py owns that arithmetic
# entirely, from labels drawn from fixed band tables — see its module docstring
# for why. A `score` key in any of these responses is dropped on parse.
# ---------------------------------------------------------------------------

DATASET_SPEC_PROMPT = """State what data this experiment actually needs, before \
any dataset is searched for.

Experiment plan (JSON):
{plan_block}

Answer with ONLY a JSON object in exactly this shape:
{{
  "task": "<the modelling/analysis task in a few words, e.g. 'train/evaluate a coding model'>",
  "domain": "<the subject area, e.g. 'software engineering'>",
  "languages": ["<natural or programming languages the data must be in>"],
  "data_types": ["<the fields an example must contain, e.g. 'instruction', 'code', 'explanation'>"],
  "desired_examples": <integer: how many examples this experiment would ideally have>,
  "minimum_quality": <number 0-1: how clean the data has to be for the result to mean anything>,
  "license_requirements": ["<one or more of: permissive, share-alike, noncommercial-ok, any>"],
  "avoid": ["<properties that would disqualify a dataset, e.g. 'synthetic-only', 'duplicates', 'unrelated programming languages', 'evaluation-test leakage'>"]
}}

Rules:
- `data_types` is the most important field. Name the *fields an individual \
example must have* for this plan's method to run at all — not a description of \
the topic. If the plan fine-tunes on instructions, that is \
["instruction", "code"], not ["software"].
- `desired_examples` should reflect the plan's own scale, not an aspiration. A \
pilot comparing two prompting strategies needs hundreds; a fine-tuning run needs \
tens of thousands.
- Use "any" in `license_requirements` only if the plan genuinely does not care.
- Be specific in `avoid`. Generic entries help nothing; entries naming what \
would actually invalidate *this* experiment do.
"""

DATASET_EVIDENCE_PROMPT = """Determine whether this dataset satisfies the \
following requirements. For every requirement, provide evidence from the \
dataset card or the inspected rows. If evidence is unavailable, mark it UNKNOWN \
rather than assuming it is true.

Requirement spec (JSON):
{spec_block}

Dataset: {dataset_id}
Declared columns: {columns_block}
Rows actually inspected ({rows_sampled} sampled): {sample_block}

Measured statistics over those rows (computed, not claimed):
{inspection_block}

Dataset card (truncated):
{card_block}

Answer with ONLY a JSON object in exactly this shape:
{{
  "requirements": {{
    "<requirement name>": {{
      "status": "pass" | "fail" | "unknown",
      "evidence": "<what in the card or the sampled rows establishes this>"
    }}
  }},
  "task_relevance": "exact" | "related" | "weak" | "unrelated",
  "task_relevance_evidence": "<why, citing the card or the rows>",
  "content_relevance": "exact" | "related" | "weak" | "unrelated",
  "content_relevance_evidence": "<why, citing the sampled rows specifically>",
  "column_mapping": {{"<required data_type>": "<the actual column name that carries it, or null>"}}
}}

Rules:
- Include one entry in `requirements` for each of the spec's `data_types`, one \
for the language requirement, one for the size requirement, and one for each \
entry in the spec's `avoid` list.
- "unknown" is the correct answer far more often than it feels like. A card that \
does not mention something has not established it. Do not infer a property from \
the dataset's *name*.
- `column_mapping` keys must be the spec's `data_types` verbatim, and values must \
be column names copied exactly from "Declared columns" above. Use null when no \
column carries that type — a name that isn't in the list is discarded anyway.
- Judge `content_relevance` from the sampled rows, not the card. The card says \
what the uploader intended; the rows are what the experiment will run on.
- Do NOT return a score. The score is computed from these labels, not supplied.
"""

DATASET_CRITIC_PROMPT = """Assume this dataset is bad. Your job is to find \
reasons it should NOT be downloaded and used for this experiment.

This dataset currently leads the ranking, which means the case *for* it has \
already been made. Do not restate it. Look for the specific ways a plausible \
looking dataset turns out to be unusable.

Requirement spec (JSON):
{spec_block}

Dataset: {dataset_id}
Declared columns: {columns_block}
Rows actually inspected ({rows_sampled} sampled): {sample_block}

Measured statistics over those rows (computed, not claimed):
{inspection_block}

Dataset card (truncated):
{card_block}

Work through this checklist. For each item, decide whether it is a real problem \
*for this experiment*, and report only the ones that are:

  description_mismatch      the sampled rows contradict what the CARD claims about
                            itself (a dataset misrepresenting its own contents) —
                            NOT a mismatch with the requirement spec, which is
                            already scored separately
  mostly_irrelevant         a large share of examples are not relevant to the spec
  unusable_schema           the data cannot serve the task at all, whatever the
                            column names are — not merely that a requested field
                            is absent or spelled differently, which is already
                            scored
  substantial_duplication   there is significant duplication
  mostly_empty_or_broken    a large share of examples are empty, truncated or malformed
  synthetic_only            the data is model-generated rather than collected
  undocumented_provenance   where the data came from is not documented
  unknown_license           the license is absent, unclear, or incompatible
  suspiciously_tiny         the dataset is far too small for what the plan needs
  unrelated_material        it contains substantial material unrelated to the spec
  evaluation_contamination  it contains evaluation/benchmark/test data this plan would be scored against
  personal_information      it contains personal or sensitive information

Answer with ONLY a JSON object in exactly this shape:
{{
  "findings": [
    {{"code": "<one code from the list above, verbatim>", "evidence": "<what in the card or rows shows this>"}}
  ]
}}

Rules:
- Use the codes above exactly. Any other code is discarded, so an objection \
phrased in your own words has no effect at all.
- Relevance, schema fit, licensing, provenance and data quality have ALREADY \
been measured and scored for this dataset, and it passed. Do not re-litigate \
them. A field the spec named that this dataset spells differently, or supplies \
implicitly (a single-ticker price series has no symbol column because every row \
is the same symbol), is a scored imperfection, not a reason to reject. Raise a \
finding only for something the scoring could not see.
- Every finding needs evidence from the card or the sampled rows. A finding you \
cannot evidence is a suspicion, and suspicions do not belong here — omit it.
- Return an empty `findings` list if you genuinely cannot substantiate any \
problem. That is a valid and useful answer; inventing a finding to seem \
thorough rejects a usable dataset.
- Do NOT return a score or an overall verdict. Findings only.
"""

# Appended after the concrete dataset facts by CoderAgent._hf_dataset_block on
# the download path. Kept out of .format() reach for the same reason
# HF_DATASET_USAGE_NOTE is — literal JSON braces.
HF_LOCAL_DATASET_USAGE_NOTE = """Each line of that file is one JSON object, so \
reading it needs nothing but the standard library:
  rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
Do NOT add the `datasets` package, do not use `pandas.read_parquet`, and do not \
fetch anything over the network for this data — it is already on disk, and the \
machine this runs on may have no outbound route at all.

Use this dataset ONLY if it genuinely fits the plan's data_requirements. If it \
doesn't, ignore it completely, generate/synthesize the data the plan describes, \
and say in assumptions_made that you did and why.

Either way, load_data() MUST still work when the file isn't there (a different \
run host, a cleared cache, a moved directory): wrap the read in try/except, log \
the failure, and fall back to a small synthesized stand-in dataset in the same \
shape, recording that fallback in assumptions_made and the README. Code that \
assumes a local data file will simply be there is not acceptable.
"""
