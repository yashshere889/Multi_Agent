"""Deterministic metrics over a literature/interdisciplinary run.

Everything in this module is pure and LLM-free by design. A harness whose
headline numbers came from a model would be measuring the model twice — once in
the pipeline and once in the scorer — and the whole reason to have an eval is to
get a verdict that doesn't move when the model has an off day. The one metric
that genuinely needs judgment (precision over papers no gold set can enumerate)
is quarantined in judge.py and reported separately, never mixed into these.

Paper identity
--------------
`paper_keys` returns *every* identifier a record carries rather than the single
doi-or-title key the pipeline dedupes on
(agents/literature/nodes.py:merge_and_dedupe_node). That difference is
deliberate, not drift: the pipeline needs one hashable key to dedupe a stream in
one pass, while matching a gold entry against a search result is an offline
question that can afford to check every identifier — a gold paper recorded with
a DOI must still match the same work returned by arXiv with only an id and a
title. Title normalization itself is imported from the pipeline rather than
reimplemented, so the two never disagree about what a title *is*.

A useful side effect: papers that the pipeline's single-key dedupe let through
as distinct, but that share an identifier under this stricter matching, are
exactly the near-duplicates `duplicate_rate` reports — an arXiv preprint and its
published DOI version being the textbook case.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

# The pipeline's own title normalization. Imported, never copied: "is this the
# same title?" must have exactly one answer in this repo.
from research_pipeline.agents.literature.nodes import _normalize_title

# arXiv ids carry an optional version suffix (1706.03762v2) and an optional
# "arXiv:" prefix depending on which API reported them; both are stripped so the
# same paper from two sources produces one key.
_ARXIV_RE = re.compile(r"^(?:arxiv:)?(.+?)(?:v\d+)?$", re.IGNORECASE)

# Titles shorter than this are too generic to be safe identity evidence on their
# own ("Introduction", "Survey"), so they're skipped as keys rather than risking
# a false match that would silently inflate recall.
MIN_TITLE_KEY_CHARS = 12


def _normalize_arxiv_id(value: str) -> str:
    match = _ARXIV_RE.match(value.strip())
    return (match.group(1) if match else value).lower()


def paper_keys(paper: dict) -> Set[str]:
    """Every identity key a paper record carries. Two records refer to the same
    work when these sets intersect."""
    keys: Set[str] = set()

    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        # DOIs are frequently stored with a resolver prefix by one source and
        # bare by another.
        keys.add("doi:" + re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi))

    arxiv_id = (paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        keys.add("arxiv:" + _normalize_arxiv_id(arxiv_id))

    title = _normalize_title(paper.get("title") or "")
    if len(title) >= MIN_TITLE_KEY_CHARS:
        keys.add("title:" + title)

    return keys


def same_paper(a: dict, b: dict) -> bool:
    return bool(paper_keys(a) & paper_keys(b))


def match_gold(gold: Sequence[dict], returned: Sequence[dict]) -> Dict[str, list]:
    """Splits a gold set into the papers a run found and the ones it missed.

    Returns the papers themselves, not just counts: a list of *which* known-good
    papers the search consistently misses is the single most actionable thing
    this harness produces — it points straight at the queries.
    """
    returned_keys: Set[str] = set()
    for paper in returned:
        returned_keys |= paper_keys(paper)

    found, missed = [], []
    for entry in gold:
        (found if paper_keys(entry) & returned_keys else missed).append(entry)
    return {"found": found, "missed": missed}


def recall(gold: Sequence[dict], returned: Sequence[dict]) -> Optional[float]:
    """Fraction of the gold set the run retrieved.

    Read this as a *relative* number for A/B, never as an absolute score. A gold
    set built from a survey's reference list holds a hundred-odd papers while a
    default run returns a few dozen, so even a flawless search scores far below
    1.0. What matters is whether a change moves it.
    """
    if not gold:
        return None
    return len(match_gold(gold, returned)["found"]) / len(gold)


def duplicate_groups(returned: Sequence[dict]) -> List[List[dict]]:
    """Papers the pipeline emitted as distinct that are actually the same work.

    Should be empty. Anything here is a dedupe escape — most often the same
    paper as an arXiv preprint and as a published DOI, which the pipeline's
    single doi-or-title key cannot catch because the two records key on
    different fields.
    """
    groups: List[List[dict]] = []
    for paper in returned:
        for group in groups:
            if any(same_paper(paper, member) for member in group):
                group.append(paper)
                break
        else:
            groups.append([paper])
    return [group for group in groups if len(group) > 1]


def _fraction(count: int, total: int) -> Optional[float]:
    return count / total if total else None


def pool_metrics(returned: Sequence[dict]) -> dict:
    """Properties of the returned pool that need no gold set at all."""
    total = len(returned)
    duplicates = duplicate_groups(returned)
    scores = [p["relevance_score"] for p in returned if isinstance(p.get("relevance_score"), int)]

    by_source: Dict[str, int] = {}
    for paper in returned:
        by_source[paper.get("source") or "unknown"] = by_source.get(paper.get("source") or "unknown", 0) + 1

    return {
        "pool_size": total,
        # A paper with no abstract carries almost no signal for the Hypothesis
        # Agent or the Writer, but still occupies a citable slot — so a run that
        # quietly fills up with them is worse than its pool size suggests.
        "abstract_coverage": _fraction(sum(1 for p in returned if (p.get("abstract") or "").strip()), total),
        "with_pdf": _fraction(sum(1 for p in returned if p.get("pdf_url")), total),
        # Zero from a source usually means that source failed, not that it had
        # nothing — the search clients log and continue on error, so a run can
        # look successful on a third of the evidence.
        "by_source": by_source,
        "duplicate_groups": len(duplicates),
        "mean_relevance_score": sum(scores) / len(scores) if scores else None,
        "scored": len(scores),
    }


def literature_metrics(gold: Sequence[dict], literature_output: dict) -> dict:
    """The full deterministic picture for one question's literature run."""
    returned = literature_output.get("merged_papers") or literature_output.get("papers") or []
    matched = match_gold(gold, returned)

    return {
        "recall": recall(gold, returned),
        "gold_total": len(gold),
        "gold_found": len(matched["found"]),
        "missed_titles": [p.get("title") for p in matched["missed"]],
        # Written by the relevance screen; 0 when it was disabled or never ran.
        "screened_out": literature_output.get("papers_filtered_out") or 0,
        "queries": literature_output.get("search_queries") or [],
        **pool_metrics(returned),
    }


def _field_set(paper: dict) -> Set[str]:
    return {str(f).strip().lower() for f in (paper.get("fields_of_study") or []) if str(f).strip()}


def interdisciplinary_metrics(output: dict) -> dict:
    """Checks specific to cross-pollination, none of which need a gold set —
    a same-field survey's references cannot be ground truth for work found in
    *other* fields, so these measure the agent's own claims instead.
    """
    cross_field = output.get("cross_field_papers") or []
    papers = output.get("papers") or []
    insights = output.get("bridge_insights") or []

    # The in-domain papers are the merged pool minus what this agent added.
    # Derived by identity rather than by slicing on len(core_paper_ids), so a
    # change to how the pool is ordered can't silently mis-attribute fields.
    cross_field_keys: Set[str] = set()
    for paper in cross_field:
        cross_field_keys |= paper_keys(paper)
    core_papers = [p for p in papers if not (paper_keys(p) & cross_field_keys)]

    core_fields: Set[str] = set()
    for paper in core_papers:
        core_fields |= _field_set(paper)

    # Does a paper labelled cross-field actually come from another field? The
    # agent stamps `discipline` from the query that found it, which is an
    # assertion, not a check. arXiv categories and Semantic Scholar's
    # fieldsOfStudy are the independent evidence.
    with_fields = [p for p in cross_field if _field_set(p)]
    off_field = [p for p in with_fields if not (_field_set(p) & core_fields)] if core_fields else []

    # The ids a bridge insight may legitimately cite as cross-field evidence,
    # derived the same way the agent derives them (agents/interdisciplinary_
    # literature/..._agent.py:_paper_id) so the two cannot disagree.
    cross_field_ids = {
        str(p.get("arxiv_id") or p.get("paper_id") or p.get("doi") or "") for p in cross_field
    } - {""}
    grounded = [
        i for i in insights
        if cross_field_ids & {str(pid) for pid in (i.get("supporting_paper_ids") or [])}
    ]

    return {
        "fields_explored": [f.get("field") for f in output.get("fields_explored") or []],
        "cross_field_papers": len(cross_field),
        "pool_size": len(papers),
        # Of the cross-field papers whose source reported a field at all, how
        # many really sit outside the in-domain papers' fields. None when
        # nothing reported a field, which is itself worth seeing.
        "off_field_rate": _fraction(len(off_field), len(with_fields)),
        "field_coverage": _fraction(len(with_fields), len(cross_field)),
        "bridge_insights": len(insights),
        # An insight citing only in-domain papers is not evidence of a
        # cross-field transfer, however well it reads. The agent currently
        # checks that cited ids *resolve*, not that they point at cross-field
        # work, so this is the metric that would catch that gap.
        "grounded_insight_rate": _fraction(len(grounded), len(insights)),
        "mean_relevance_score": pool_metrics(cross_field)["mean_relevance_score"],
    }


def aggregate(per_question: Iterable[dict]) -> dict:
    """Means across questions, skipping Nones rather than treating them as 0 —
    a question with no gold set must not drag the mean recall down."""
    rows = list(per_question)
    if not rows:
        return {}

    def mean(key: str) -> Optional[float]:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(values) / len(values) if values else None

    return {
        "questions": len(rows),
        "mean_recall": mean("recall"),
        "mean_pool_size": mean("pool_size"),
        "mean_abstract_coverage": mean("abstract_coverage"),
        "mean_relevance_score": mean("mean_relevance_score"),
        "total_gold_found": sum(r.get("gold_found", 0) for r in rows),
        "total_gold": sum(r.get("gold_total", 0) for r in rows),
        "total_screened_out": sum(r.get("screened_out", 0) for r in rows),
        "total_duplicate_groups": sum(r.get("duplicate_groups", 0) for r in rows),
    }
