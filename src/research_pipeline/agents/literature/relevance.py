"""Relevance scoring and filtering for retrieved papers.

Kept out of nodes.py for the same reason clients.py is: these are plain
functions with no LangGraph state coupling, so they unit-test without building
a graph — and the Interdisciplinary Literature Agent reuses them from its own
graph rather than re-implementing "is this paper worth keeping?" a second time.

Why this exists at all: the search clients return whatever a keyword query
matched, and until this module nothing between the API response and the Writer's
citable paper pool ever asked whether a matched paper was actually about the
research question. A pool padded with near-misses doesn't just add noise — every
paper in it is a paper the Writer is entitled to cite.

The split of responsibility follows the pipeline's standing rule that the model
scores and Python decides what the score means:

    score_papers()   -> asks the model for one 0-5 integer per paper, nothing else
    apply_threshold() -> decides, deterministically, which papers that keeps

The model is never asked "which of these should I keep", and never sees the
threshold. Two failure modes are handled here rather than left to the caller,
because both would otherwise turn an accuracy feature into a data-loss bug:

  * A scoring call that fails (unreachable LLM, unparseable JSON, a batch the
    model simply omitted papers from) yields `None` for the affected papers, and
    an unscored paper is always *kept*. A filter must never drop evidence
    because of its own failure.
  * A threshold that would empty the pool falls back to the top `keep_min`
    papers by score. Every downstream agent treats an empty paper list as a hard
    error, so a harsh threshold on a thin result set must degrade to "the best of
    a bad set", not to "no run".
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.literature.state import Paper
from research_pipeline.llm_json import invoke_json

logger = logging.getLogger(__name__)

MIN_SCORE_VALUE = 0
MAX_SCORE_VALUE = 5

# How much of one paper goes into the digest. Deliberately smaller than the
# Interdisciplinary Agent's 400: screening is a coarser judgment than synthesis,
# and a tighter budget fits more papers per call, which is what keeps this to a
# small number of round-trips over a pool of 40+.
DIGEST_ABSTRACT_CHARS = 300

SYSTEM_PROMPT = """You are a research librarian screening database search results.

Rules you must follow:
- Judge each paper using only the title and abstract shown to you. Do not rely \
on prior knowledge of the paper, its authors, or where it was published.
- Score every paper you are given, and score no id that was not given to you.
- Be strict. Keyword search returns near-misses, and saying so is the entire \
point of this step — a low score is a useful answer, not a failure.
- Return ONLY valid JSON matching the schema in the user prompt. No markdown \
fences, no commentary before or after the JSON.
"""

# The two things a paper can be useful *for* in this pipeline, phrased as
# separate rubrics because they are genuinely different questions. Scoring a
# cross-field paper on direct topical relevance would score every one of them
# near zero — being topically distant is what makes it a cross-field paper —
# so the Interdisciplinary Agent asks about transferability instead.
DIRECT_RELEVANCE_CRITERION = """Score how directly each paper bears on the research problem above.

  5 - directly addresses the research problem
  4 - addresses the same problem from a different angle, setting, or dataset
  3 - useful background: shares a core method, subproblem, or evaluation with it
  2 - same broad area, but does not engage the specific problem
  1 - tangential; only a shared term or generic technique connects them
  0 - unrelated, or a keyword-match coincidence"""

TRANSFER_RELEVANCE_CRITERION = """These papers were retrieved from *other* fields than the research problem above, \
so do not score them on topical overlap — being topically distant is expected. \
Score how usefully each one's method or finding could transfer to the research problem.

  5 - names a specific method, model, or finding that could be applied to the problem
  4 - a clearly transferable mechanism that would need some adaptation
  3 - a plausible transfer, though the paper does not itself make the connection
  2 - only an abstract structural similarity (both involve sparsity, networks, …)
  1 - thematic or metaphorical overlap only
  0 - nothing transferable"""

SCORING_PROMPT = """Research problem: {objective}

{criterion}

Papers to score:

{papers_block}

Return ONLY a JSON object with this exact shape, with one entry per paper above:
{{
  "scores": [
    {{"id": "P0", "score": 0, "reason": "one short clause"}}
  ]
}}
"""

_ID_RE = re.compile(r"P(\d+)")


def _paper_line(index: int, paper: Paper) -> str:
    """One paper rendered for the digest, labelled with its position in the pool
    being scored so the model's answer maps back without any title matching."""
    # tldr preferred over abstract where Semantic Scholar supplied one: it is a
    # purpose-written one-sentence summary, so it survives the character budget
    # intact where an abstract gets cut mid-sentence.
    body = (paper.get("tldr") or paper.get("abstract") or "").strip().replace("\n", " ")
    if len(body) > DIGEST_ABSTRACT_CHARS:
        body = body[:DIGEST_ABSTRACT_CHARS] + "…"
    year = paper.get("year") if paper.get("year") is not None else "n.d."
    venue = f", {paper['venue']}" if paper.get("venue") else ""
    return (
        f"[P{index}] {paper.get('title') or '(untitled)'} ({year}{venue})\n"
        f"{body or '(no abstract available)'}"
    )


def _batches(papers: Sequence[Paper], max_chars: int) -> List[List[Tuple[int, Paper]]]:
    """Groups (pool index, paper) pairs so each batch's rendered digest stays
    under max_chars — the same character-count-as-token-budget proxy
    agents/hypothesis/papers.py:chunk_papers uses. Indices are positions in the
    whole pool, not within a batch, so scores merge back without renumbering.

    A single paper over budget still gets its own batch rather than being
    dropped: an unscored paper is kept by apply_threshold, so silently skipping
    one here would quietly weaken the filter instead of failing loudly.
    """
    batches: List[List[Tuple[int, Paper]]] = []
    current: List[Tuple[int, Paper]] = []
    current_chars = 0

    for index, paper in enumerate(papers):
        size = len(_paper_line(index, paper))
        if current and current_chars + size > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append((index, paper))
        current_chars += size

    if current:
        batches.append(current)
    return batches


def _parse_scores(raw: dict, valid_indices: set[int]) -> dict[int, int]:
    """Pulls {pool index: score} out of one model response, discarding anything
    malformed. An id the model invented, an id from another batch, or a score
    outside 0-5 is dropped and logged rather than coerced — the affected paper
    then reads as unscored, which is the safe direction."""
    scores: dict[int, int] = {}
    for entry in raw.get("scores", []) or []:
        if not isinstance(entry, dict):
            continue
        match = _ID_RE.fullmatch(str(entry.get("id", "")).strip())
        if not match:
            logger.warning("Discarding a relevance score with an unparseable id: %r", entry.get("id"))
            continue
        index = int(match.group(1))
        if index not in valid_indices:
            logger.warning("Discarding a relevance score for id P%d, which was not in this batch", index)
            continue
        try:
            score = int(entry["score"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Discarding a relevance score for id P%d: score is missing or not an integer", index)
            continue
        if not MIN_SCORE_VALUE <= score <= MAX_SCORE_VALUE:
            logger.warning("Discarding out-of-range relevance score %d for id P%d", score, index)
            continue
        scores[index] = score
    return scores


def score_papers(
    chat_model: BaseChatModel,
    objective: str,
    papers: Sequence[Paper],
    *,
    criterion: str = DIRECT_RELEVANCE_CRITERION,
    batch_max_chars: int = 12000,
) -> List[Optional[int]]:
    """Scores every paper 0-5 against `objective`, returning a list positionally
    parallel to `papers`.

    An entry is None where no usable score came back — a failed batch, a paper
    the model skipped, a malformed entry. That is not an error: callers treat
    None as "unknown", never as "irrelevant".

    Batches are independent, so one failing call costs only its own papers'
    scores rather than the whole pool's.
    """
    scores: List[Optional[int]] = [None] * len(papers)
    if not papers:
        return scores

    for batch in _batches(papers, batch_max_chars):
        indices = {index for index, _ in batch}
        prompt = SCORING_PROMPT.format(
            objective=objective,
            criterion=criterion,
            papers_block="\n\n".join(_paper_line(index, paper) for index, paper in batch),
        )
        try:
            raw = invoke_json(chat_model, SYSTEM_PROMPT, prompt)
        except Exception as exc:
            # Deliberately broad, and for the same reason nodes.generate_queries
            # catches broadly: the realistic failures here are a down or
            # unreachable LLM server and a client-side timeout, neither of which
            # is an LLMJSONError. Narrowing this to LLMJSONError would let a
            # connection error escape and fail the whole run — turning a filter
            # whose entire contract is "degrade to keeping everything" into the
            # most fragile node in the graph.
            logger.warning(
                "Relevance scoring failed for a batch of %d paper(s) (%s) — leaving them unscored",
                len(batch), exc,
            )
            continue
        for index, score in _parse_scores(raw, indices).items():
            scores[index] = score

        missing = [i for i in indices if scores[i] is None]
        if missing:
            logger.warning("Model returned no usable score for %d paper(s) in a batch — keeping them unscored", len(missing))

    return scores


def apply_threshold(
    papers: Sequence[Paper],
    scores: Sequence[Optional[int]],
    *,
    min_score: int,
    keep_min: int,
) -> Tuple[List[Paper], List[Paper]]:
    """Splits `papers` into (kept, dropped) on `min_score`, annotating each kept
    paper with the `relevance_score` it was judged on.

    Two deterministic guarantees, both about not losing evidence to this step:
    an unscored paper (score None) is always kept, and if the threshold would
    leave nothing at all, the `keep_min` highest-scoring papers are kept anyway.

    `keep_min=0` turns that second guarantee off, for the caller whose empty
    result is a legitimate answer rather than a failed run. It is exactly the
    difference between the two callers: an empty *paper pool* is a hard error in
    every downstream agent, while an empty *cross-field* pool is a case the
    Interdisciplinary Agent's bridge synthesis already handles — rescuing there
    would readmit the untransferable papers the screen exists to remove.

    Papers are copied rather than mutated in place — the search nodes they come
    from are cached by LangGraph (CachePolicy in graph.py), and writing into a
    cached node's output would leak this run's annotations into the next run
    that hits the same cache entry.
    """
    kept: List[Paper] = []
    dropped: List[Paper] = []

    for paper, score in zip(papers, scores):
        annotated: Paper = {**paper, "relevance_score": score}
        # None means "we failed to judge this", which is never grounds to drop.
        if score is None or score >= min_score:
            kept.append(annotated)
        else:
            dropped.append(annotated)

    if not kept and dropped and keep_min > 0:
        # Everything scored below the bar. Returning an empty pool would fail
        # the run outright in the next agent, so the best of a bad set is the
        # more useful answer — loudly, because it means the search itself, not
        # the filter, is what went wrong.
        ranked = sorted(range(len(dropped)), key=lambda i: dropped[i].get("relevance_score") or 0, reverse=True)
        rescued_indices = set(ranked[:keep_min])
        logger.warning(
            "Every one of %d paper(s) scored below the relevance threshold of %d — "
            "keeping the %d highest-scoring rather than returning an empty pool. "
            "This usually means the generated search queries missed the question.",
            len(dropped), min_score, len(rescued_indices),
        )
        return (
            [dropped[i] for i in ranked if i in rescued_indices],
            [p for i, p in enumerate(dropped) if i not in rescued_indices],
        )

    return kept, dropped
