"""LLM-as-judge precision, for the part no gold set can measure.

Recall is deterministic — a gold set either contains a returned paper or it
doesn't. Precision isn't: no bibliography enumerates every paper that would have
been a legitimate result, so "what fraction of what we returned was actually
worth returning?" needs a judgment. This module is where that judgment is
quarantined, and metrics.py stays LLM-free because of it.

Why the judge must not reuse the screen's rubric
-------------------------------------------------
The obvious implementation — score the pool with
agents/literature/relevance.py's own prompt and average it — measures nothing.
It grades the relevance screen with the screen's own answer key, so a screen
that is confidently wrong scores perfectly, and the number rises whenever the
threshold rises regardless of whether the pool improved.

So the judge here is deliberately a different question asked a different way: a
binary keep/discard verdict with a required one-line justification, rather than
a 0-5 rating, phrased around what a researcher would do with the paper rather
than how well it matches the question. It must stay different. If you find
yourself importing a prompt from relevance.py into this file, the eval has
stopped being independent of the thing it evaluates.

The judge is off by default in the harness (`--judge`) for two reasons: it costs
an LLM call per batch per question, and unlike every other number this harness
produces it isn't reproducible run to run.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.eval import metrics
from research_pipeline.llm_json import invoke_json

logger = logging.getLogger(__name__)

DIGEST_ABSTRACT_CHARS = 300

SYSTEM_PROMPT = """You are a senior researcher triaging a database search for a \
colleague. For each result you decide one thing only: would you pass this paper \
on to them, or bin it?

Rules you must follow:
- Judge from the title and abstract shown. Do not rely on prior knowledge of the \
paper, its authors, or where it was published.
- You are triaging, not grading. A paper you would pass on with reservations is \
still a keep; a paper that merely shares vocabulary with the topic is a bin.
- Give a verdict for every id you are shown, and for no id you were not shown.
- Return ONLY valid JSON. No markdown fences, no commentary.
"""

JUDGE_PROMPT = """Your colleague is working on: {question}

For each result below, decide whether you would pass it on to them.

{papers_block}

Return ONLY a JSON object with this exact shape, one entry per result above:
{{
  "verdicts": [
    {{"id": "P0", "keep": true, "why": "one short clause"}}
  ]
}}
"""

_ID_RE = re.compile(r"P(\d+)")


def _paper_line(index: int, paper: dict) -> str:
    body = (paper.get("abstract") or paper.get("tldr") or "").strip().replace("\n", " ")
    if len(body) > DIGEST_ABSTRACT_CHARS:
        body = body[:DIGEST_ABSTRACT_CHARS] + "…"
    year = paper.get("year") if paper.get("year") is not None else "n.d."
    return f"[P{index}] {paper.get('title') or '(untitled)'} ({year})\n{body or '(no abstract available)'}"


def _batches(papers: Sequence[dict], max_chars: int):
    """Same greedy character-budget grouping the rest of the pipeline uses.
    Written out here rather than imported from relevance.py on purpose — see
    this module's docstring on staying independent of what it evaluates.
    """
    batches, current, current_chars = [], [], 0
    for index, paper in enumerate(papers):
        size = len(_paper_line(index, paper))
        if current and current_chars + size > max_chars:
            batches.append(current)
            current, current_chars = [], 0
        current.append((index, paper))
        current_chars += size
    if current:
        batches.append(current)
    return batches


def judge_pool(
    chat_model: BaseChatModel,
    question: str,
    papers: Sequence[dict],
    *,
    batch_max_chars: int = 12000,
) -> List[Optional[bool]]:
    """A keep/bin verdict per paper, positionally parallel to `papers`.

    None where no usable verdict came back. Unlike the relevance screen — whose
    unknowns are kept, because dropping evidence over its own failure would be a
    bug — unknowns here are simply *excluded* from the precision denominator.
    Guessing either way would bias the number this harness exists to report.
    """
    verdicts: List[Optional[bool]] = [None] * len(papers)
    if not papers:
        return verdicts

    for batch in _batches(papers, batch_max_chars):
        valid = {index for index, _ in batch}
        prompt = JUDGE_PROMPT.format(
            question=question,
            papers_block="\n\n".join(_paper_line(index, paper) for index, paper in batch),
        )
        try:
            raw = invoke_json(chat_model, SYSTEM_PROMPT, prompt)
        except Exception as exc:
            logger.warning("Judge failed on a batch of %d paper(s) (%s) — leaving them unjudged", len(batch), exc)
            continue
        for entry in raw.get("verdicts", []) or []:
            if not isinstance(entry, dict):
                continue
            match = _ID_RE.fullmatch(str(entry.get("id", "")).strip())
            if not match or int(match.group(1)) not in valid:
                logger.warning("Discarding a judge verdict with an unusable id: %r", entry.get("id"))
                continue
            keep = entry.get("keep")
            if not isinstance(keep, bool):
                logger.warning("Discarding a judge verdict for P%s: 'keep' is not a boolean", match.group(1))
                continue
            verdicts[int(match.group(1))] = keep

    return verdicts


def precision(verdicts: Sequence[Optional[bool]]) -> Optional[float]:
    """Fraction of *judged* papers the judge would keep. None when nothing was
    judged, never 0.0 — "the judge was unreachable" and "every paper was junk"
    must not render as the same number."""
    judged = [v for v in verdicts if v is not None]
    if not judged:
        return None
    return sum(1 for v in judged if v) / len(judged)


def agreement_with_screen(papers: Sequence[dict], verdicts: Sequence[Optional[bool]], min_score: int) -> Optional[dict]:
    """How often the relevance screen and an independent judge reach the same
    conclusion about the same paper.

    This is the number that actually says whether the screen works. Precision
    alone can't: a screen that keeps everything scores the same as a good one on
    a pool that was already clean. Only papers carrying both a score and a
    verdict are counted.
    """
    pairs = [
        (p["relevance_score"], v)
        for p, v in zip(papers, verdicts)
        if isinstance(p.get("relevance_score"), int) and v is not None
    ]
    if not pairs:
        return None
    agree = sum(1 for score, keep in pairs if (score >= min_score) is keep)
    return {
        "compared": len(pairs),
        "agreement": agree / len(pairs),
        # The screen kept it, the judge would not have — false positives.
        "screen_kept_judge_binned": sum(1 for score, keep in pairs if score >= min_score and not keep),
        # The screen dropped it, the judge would have kept it — the expensive
        # direction, since a dropped paper is gone before anything can cite it.
        "screen_dropped_judge_kept": sum(1 for score, keep in pairs if score < min_score and keep),
    }
