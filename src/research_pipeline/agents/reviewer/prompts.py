"""Prompt templates for the Reviewer Agent.

Reserved for what checks.py's deterministic checks genuinely can't do:
hallucination nuance (is this claim actually grounded, vs. reasonable
interpretation?), and qualitative judgment (tone, flow, honesty of framing).
Every prompt hands the model only the ground-truth slice relevant to the
section it's checking — never the paper's own JSON summary as if it were
truth, and never a section's text without the data it should be grounded in.
"""

SYSTEM_PROMPT = """You are a rigorous, skeptical academic peer reviewer. You \
check a paper's text strictly against the ground-truth data given to you in \
each request — never against general knowledge, never against what "sounds \
right" for the topic.

Rules:
- Only the ground truth given to you in the current request is a valid source \
of facts. If a claim in the paper text isn't traceable to it, it's a \
hallucination — flag it, even if it sounds plausible or is common knowledge \
in the field.
- Distinguish fabricated facts/numbers/results from reasonable interpretive \
or transitional language (e.g. "this suggests a promising direction" or \
"building on this finding" is fine; an invented statistic, an unstated \
method detail, or a claimed result not present in the ground truth is not).
- Be specific: quote or closely paraphrase the exact claim/sentence and say \
concretely what's wrong — never vague praise or criticism like "check this" \
or "seems off".
- Return ONLY valid JSON matching the schema described in the user prompt. \
No markdown fences, no commentary before or after the JSON.
"""

HALLUCINATION_CHECK_PROMPT = """You are reviewing the "{section_name}" section \
of a research paper for factual grounding.

Ground truth (JSON) — the ONLY source of facts this section may draw on:
{grounding_block}

Section text as printed in the paper:
{section_text}

Flag every claim, statistic, method description, or result in the section \
text that does NOT trace back to something in the ground truth above.

Return ONLY a JSON object with this exact shape:
{{
  "hallucinations": [
    {{"claim": "the specific claim/sentence from the section text", "issue": "concretely what's wrong / not grounded, referencing what the ground truth actually says instead", "grounded": false}}
  ]
}}
If nothing is ungrounded, return {{"hallucinations": []}}.

`grounded` is required on every entry and is the verdict that counts. Set it to \
false when the claim really is unsupported. If you looked at a claim and it IS \
supported by the ground truth, either leave it out entirely or set `grounded` to \
true — do not report a supported claim with an `issue` that explains why it is \
actually fine. Only entries with `grounded: false` are treated as problems.
"""

DISCUSSION_REVIEW_PROMPT = """You are reviewing the "Discussion" section of a \
research paper.

Ground truth per hypothesis (JSON) — statement, rationale, and the ACTUAL \
outcome as determined from the Coder Agent's own data (this is settled fact, \
not something to re-derive or second-guess):
{verdicts_block}

Discussion text as printed in the paper:
{section_text}

Do two things:
1. Flag every claim in the text not grounded in the ground truth above \
(fabricated facts/numbers — not reasonable interpretive language).
2. For each hypothesis, judge whether the text's characterization of the \
outcome is honest and consistent with its given verdict. Flag it if the text \
overstates the result (claims support/refutation more strongly than an \
"inconclusive" outcome warrants, e.g. writing as if an experiment that wasn't \
run actually succeeded) or understates it (downplays or hedges away a \
genuinely "supported"/"refuted" result).

Return ONLY a JSON object with this exact shape:
{{
  "hallucinations": [{{"claim": "...", "issue": "...", "grounded": false}}],
  "framing_issues": [{{"hypothesis_id": "...", "issue": "concretely how the text overstates or understates the given verdict, quoting the relevant phrase"}}]
}}

`grounded` is required on every hallucination entry and is the verdict that \
counts. Set it to false when the claim really is unsupported. A claim you \
checked and found supported should be left out, or marked `grounded: true` — \
never reported with an `issue` explaining that it is actually fine. Only \
entries with `grounded: false` are treated as problems.
"""

QUALITY_SCORE_PROMPT = """Score this research paper draft on 5 dimensions, \
each an integer 1 (poor) to 5 (excellent):
- clarity: is the writing clear and easy to follow?
- flow: does it move logically from section to section, with good transitions?
- tone: is it formal and academic, with appropriate hedging (no overclaiming, \
no marketing language, no unsupported superlatives)?
- structure: does it have complete, appropriately-ordered standard paper \
sections (introduction, related work, methods, results, discussion, \
limitations), each reasonably developed?
- limitations_honesty: does the Discussion/Limitations content honestly \
engage with weak, mixed, missing, or unexecuted results rather than glossing \
over or hiding them?

Ground truth for judging limitations_honesty specifically — the ACTUAL risks, \
assumptions, and not-run/skipped experiments (JSON), so you can tell whether \
the paper is honestly engaging with these or hiding them:
{ground_truth_block}

Full paper text as printed:
{paper_text}

Return ONLY a JSON object with this exact shape:
{{
  "quality_scores": {{"clarity": 1, "flow": 1, "tone": 1, "structure": 1, "limitations_honesty": 1}},
  "quality_notes": {{"clarity": "one sentence justifying the score", "flow": "...", "tone": "...", "structure": "...", "limitations_honesty": "..."}}
}}
(the "1" values above are placeholders — replace every one with your actual 1-5 score)
"""
