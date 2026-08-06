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

Each hypothesis must:
- Be specific and testable (not a vague research direction)
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
