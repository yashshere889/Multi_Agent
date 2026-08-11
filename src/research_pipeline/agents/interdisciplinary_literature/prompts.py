"""Prompt templates for the Interdisciplinary Literature Agent.

Both prompts instruct the model to ground its output strictly in the supplied
papers and to cite paper IDs — never invent papers, findings, or authors that
aren't in the input. The field-identification prompt is the one place the model
is asked to propose something *not* in the input (adjacent fields to search),
and even there the rationale must point back at the in-domain papers.
"""

SYSTEM_PROMPT = """You are a research analyst assistant specializing in \
cross-disciplinary transfer: finding methods, findings, and framings from \
*adjacent* fields that could inform a problem stated in one field.

Rules you must follow:
- Ground every claim strictly in the papers provided. Never invent papers, \
findings, authors, or statistics that are not present in the given text.
- Whenever you state a connection or insight, cite the paper ID(s) it comes \
from, using the bracketed IDs shown in the prompt.
- Prefer concrete, transferable mechanisms ("this field's X estimator handles \
the same sparsity problem") over vague thematic similarity ("both fields care \
about data").
- If the provided papers don't support a strong claim, say so explicitly \
rather than fabricating one.
- Return ONLY valid JSON matching the schema described in the user prompt. \
No markdown fences, no commentary before or after the JSON.
"""

ADJACENT_FIELDS_PROMPT = """{research_question_line}Below is a digest of the \
in-domain papers already collected for this research problem.

{papers_block}

Identify up to {max_fields} *adjacent* research fields — disciplines other than \
the one these papers sit in — whose methods, instruments, theory, or empirical \
findings could plausibly inform this problem. Do not name the field these \
papers are already in.

For each field, give:
- a short rationale tying it to something concrete in the papers above
- 1-2 short search queries (keywords or short phrases, not full sentences) that \
would find the relevant work in that field's literature

Return ONLY a JSON object with this exact shape:
{{
  "fields": [
    {{"field": "name of the adjacent field", "rationale": "why it could inform this problem", "queries": ["short query", "..."]}}
  ]
}}
"""

BRIDGE_INSIGHTS_PROMPT = """{research_question_line}Papers from adjacent fields \
were searched for and merged with the in-domain papers. Below are the \
cross-field papers that were found, each tagged with the field it came from, \
followed by a digest of the in-domain papers they should be connected to.

Cross-field papers:
{cross_field_block}

In-domain papers:
{core_block}

Produce "bridge insights": concrete statements of the form "this method or \
finding from field X could inform the core problem because Y". Each must name \
the specific transferable mechanism, not just the topical overlap, and cite the \
cross-field paper ID(s) it comes from.

Return ONLY a JSON object with this exact shape:
{{
  "bridge_insights": [
    {{
      "insight": "the transferable method/finding, stated concretely",
      "source_field": "which adjacent field it comes from",
      "connection_to_core_problem": "why it applies to the in-domain problem above",
      "supporting_paper_ids": ["paper id", "..."]
    }}
  ]
}}

Only produce an insight where the cross-field papers actually support one. \
Returning fewer insights is better than padding the list with weak thematic \
overlap.
"""
