"""Prompt templates for the Hypothesis Agent. Every prompt instructs the model
to ground its output strictly in the supplied papers and to cite paper IDs —
never invent papers, findings, or authors that aren't in the input.
"""

SYSTEM_PROMPT = """You are a research analyst assistant. You are given excerpts \
(titles, authors, abstracts, and sometimes full text) from a fixed set of \
academic papers, each tagged with a bracketed paper ID like [1706.03762].

Rules you must follow:
- Ground every claim strictly in the papers provided. Never invent papers, \
findings, authors, or statistics that are not present in the given text.
- Whenever you state a method, gap, contradiction, or hypothesis, cite the \
paper ID(s) it comes from.
- If the provided papers don't support a strong claim, say so explicitly \
rather than fabricating one.
- Return ONLY valid JSON matching the schema described in the user prompt. \
No markdown fences, no commentary before or after the JSON.
"""

BATCH_ANALYSIS_PROMPT = """Analyze the following batch of papers.

{papers_block}

Return ONLY a JSON object with this exact shape:
{{
  "batch_summary": "2-4 sentence synthesis of what this batch of papers collectively covers",
  "methods": [
    {{"method": "name of a method/approach/dataset/technique used", "paper_ids": ["..."], "notes": "brief note"}}
  ],
  "observations": [
    {{"observation": "an open question, contradiction, limitation, or understudied condition", "paper_ids": ["..."], "type": "gap | contradiction | limitation"}}
  ]
}}
"""

SYNTHESIS_PROMPT = """{research_question_line}You previously analyzed batches of a paper \
set and produced the partial analyses below. Synthesize them into one coherent \
picture of the ENTIRE paper set — do not just concatenate the batches.

Partial analyses (JSON):
{partials_block}

All paper IDs in the full set: {all_paper_ids}

Return ONLY a JSON object with this exact shape:
{{
  "literature_summary": "a concise synthesis (not a per-paper list) of what this body of literature collectively says",
  "methods_overview": [
    {{"method": "name", "papers_using_it": ["paper id", "..."], "notes": "note on how common/rare this is and how it's used"}}
  ],
  "gaps": [
    {{"gap": "an open question, contradiction, understudied condition, or repeated methodological limitation", "supporting_evidence": ["paper id", "..."], "notes": "brief note"}}
  ]
}}

Merge duplicate methods/gaps across batches instead of repeating them. In each \
method's "notes", say whether it's common (used by many papers) or rare (one or two).
"""

HYPOTHESIS_PROMPT = """{research_question_line}Based on the literature synthesis below, \
generate exactly 3 hypotheses for future experimental work.

Literature summary:
{literature_summary}

Methods overview (JSON):
{methods_overview_block}

Gaps (JSON):
{gaps_block}
{interdisciplinary_block}{staged_data_block}
Each hypothesis must:
- Be specific and testable (not a vague research direction), with an outcome that \
could genuinely go either way, and testable on real, publicly available data \
wherever the phenomenon is measured in the real world
- Be internally coherent: never contrast a method with a family it belongs to \
(Sobol' indices are themselves a variance-based method), and give no mechanism \
that does not follow from the claim
- When the research question names a system or model to study (a retirement \
model, a climate model), be a claim about that system — which inputs drive its \
outcomes, and by how much — rather than a comparison of analysis methods, whose \
result is settled by the thresholds and settings chosen rather than by any data
- When real datasets are listed above, be a claim the data itself decides: the \
data must be able to make it false, not merely feed a model whose conclusion its \
own assumptions already fix
- Reference which gap(s) and/or method(s) above it builds on
- Include a brief rationale grounded in the literature summary and gaps above — do not invent evidence
- Note expected independent/dependent variables where applicable (empty lists if not applicable)

Return ONLY a JSON object with this exact shape:
{{
  "hypotheses": [
    {{
      "id": "H1",
      "statement": "...",
      "rationale": "...",
      "related_gaps": ["..."],
      "related_methods": ["..."],
      "suggested_variables": {{"independent": ["..."], "dependent": ["..."]}}
    }},
    {{"id": "H2", "...": "..."}},
    {{"id": "H3", "...": "..."}}
  ]
}}
"""

# Rendered into HYPOTHESIS_PROMPT only when an Interdisciplinary Literature
# Agent ran upstream. Kept as an explicitly labelled block rather than mixing
# cross-field papers into the batch analyses, so the model is nudged to *use*
# the cross-field material as inspiration and can be seen to have done so —
# rather than silently averaging it into the in-domain synthesis.
INTERDISCIPLINARY_BLOCK = """
Cross-disciplinary bridge insights (JSON) — methods/findings from adjacent \
fields that an Interdisciplinary Literature Agent connected to this problem:
{bridge_insights_block}

Where one of these genuinely fits, prefer a hypothesis that draws on it, and \
say so in that hypothesis's rationale. Do not force a cross-field angle onto a \
hypothesis it doesn't fit, and do not treat a bridge insight as evidence in its \
own right.
"""

# Closing lines of staged_data.prompt_block, rendered only when real data is
# staged. Generation and ranking get different instructions because they ask
# different questions of the same inventory.
STAGED_DATA_HYPOTHESIS_INSTRUCTION = (
    "Where the research question allows, prefer hypotheses one of these real datasets can "
    "test: a result computed on real data can support or refute a hypothesis, and one computed "
    "on synthetic data cannot. Judge the fit from the listed columns, and do not force a dataset "
    "onto a hypothesis it does not fit."
)
STAGED_DATA_RANKING_INSTRUCTION = (
    "Count it towards feasibility when one of these datasets can genuinely test a hypothesis, "
    "judged from the listed columns only — never assume a column that is not listed."
)

RANKING_PROMPT = """{research_question_line}Below are 3 hypotheses that were just \
generated from the same literature synthesis, plus the synthesis they came from.

Hypotheses (JSON):
{hypotheses_block}

Literature summary:
{literature_summary}

Gaps (JSON):
{gaps_block}
{interdisciplinary_block}{staged_data_block}
Rank all 3 against each other so exactly one can be taken forward to an \
experiment. Judge each on:
- feasibility: can it realistically be tested on a shared university GPU cluster, \
with data that plausibly exists? A hypothesis testable on real, publicly \
available data outranks one that could only ever run on synthetic data, whose \
result can neither support nor refute anything.
- testability: is the claim specific enough that a result would clearly support \
or refute it — and could the result genuinely come out either way, rather than \
being true or false by construction?
- coherence: is the claim internally consistent — no method contrasted with a \
family it belongs to, no mechanism that does not follow from the claim — and, \
when the research question names a system to study, is it a claim about that \
system rather than about the analysis methods? A hypothesis failing this ranks \
below every hypothesis that passes it, whatever its other merits.
- data: if real datasets are listed, would the data actually decide the claim, \
or would it only feed a model whose answer is fixed by its own assumptions?
- grounding: how well the literature above (and any bridge insights) actually \
supports it — not how interesting it sounds.

Return ONLY a JSON object with this exact shape:
{{
  "ranking": [
    {{"hypothesis_id": "H1", "rank": 1, "score": 8.5, "justification": "why it ranks here, against the criteria above"}},
    {{"hypothesis_id": "H2", "rank": 2, "score": 6.0, "justification": "..."}},
    {{"hypothesis_id": "H3", "rank": 3, "score": 4.0, "justification": "..."}}
  ]
}}

Every hypothesis id above must appear exactly once, `rank` must use each of \
1, 2 and 3 exactly once (1 = best), and `score` is a 0-10 number.
"""
