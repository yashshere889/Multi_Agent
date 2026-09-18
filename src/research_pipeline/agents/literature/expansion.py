"""Citation-graph expansion: finding papers the queries missed by walking out
from the ones they found.

Keyword search only finds papers whose *wording* matches the question. It
reliably misses the foundational work that everyone in a field cites without
restating its title, and it misses papers that solved the same problem under
different terminology. A reference list has neither weakness — it is a
bibliography an author curated by hand, so one hop along it is high-precision
recall for precisely what the queries could not reach.

Which papers get expanded from matters
--------------------------------------
Expansion runs *after* the relevance screen, seeded from the highest-scoring
papers, because the failure mode of expanding from an unscreened pool is
compounding: one off-topic hit drags in fifty of its references, and the pool
degrades faster than the extra recall improves it. Seeding from screened papers
means every hop starts somewhere known to be on topic.

How candidates are prioritised
------------------------------
By co-citation, counted in Python: a paper cited by several independent seeds is
near-certain to be central to the problem, and a paper cited by exactly one is
often that seed's own tangent. This is a genuinely deterministic quality signal,
so nothing here asks a model which candidates look important — ranking is
`(how many seeds cite it, its own citation count)`, and only the resulting order
decides what survives the budget. The model's only role in expansion is the
relevance screen applied afterwards, the same one the search results go through.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from research_pipeline.agents.literature.clients import semantic_scholar_identifier
from research_pipeline.agents.literature.state import Paper

logger = logging.getLogger(__name__)

# A hop needs an id S2 can resolve; a paper carrying none is skipped as a seed
# (it can still be a candidate, arriving from someone else's bibliography).
FetchRelated = Callable[[str, str, int], List[Paper]]


def choose_seeds(papers: Sequence[Paper], limit: int) -> List[Paper]:
    """The papers worth expanding from: highest relevance score first, then most
    cited, capped at `limit`.

    An unscored paper sorts below every scored one rather than being dropped.
    Unscored means the screen never ran or failed, which is not evidence of
    quality in either direction — but when scored papers are available they are
    the better seeds, and a hop is expensive enough to spend on those first.
    """
    if limit <= 0:
        return []

    def sort_key(paper: Paper) -> Tuple[int, int, int]:
        score = paper.get("relevance_score")
        return (
            1 if isinstance(score, int) else 0,
            score if isinstance(score, int) else 0,
            paper.get("citation_count") or 0,
        )

    eligible = [p for p in papers if semantic_scholar_identifier(p)]
    skipped = len(papers) - len(eligible)
    if skipped:
        logger.debug("%d paper(s) carry no Semantic Scholar-resolvable id and can't be expanded from", skipped)
    return sorted(eligible, key=sort_key, reverse=True)[:limit]


def rank_candidates(
    candidates_by_seed: Sequence[List[Paper]],
    dedupe_key: Callable[[Paper], str],
    known_keys: set,
) -> List[Paper]:
    """Merges each seed's hop into one list ordered by how many seeds cited each
    paper, then by the paper's own citation count.

    `known_keys` are the papers already in the pool; those are dropped rather
    than ranked, since re-adding them is what the merge step exists to prevent.
    Papers are compared on the pipeline's own dedupe key, so "already have it"
    means exactly what it means everywhere else here.
    """
    merged: Dict[str, Paper] = {}
    seed_counts: Dict[str, int] = {}

    for hop in candidates_by_seed:
        # Counted per seed, not per occurrence: one bibliography listing a paper
        # twice is not two seeds agreeing about it.
        seen_this_seed: set = set()
        for paper in hop:
            if not (paper.get("title") or "").strip():
                continue
            key = dedupe_key(paper)
            if key in known_keys or key in seen_this_seed:
                continue
            seen_this_seed.add(key)
            seed_counts[key] = seed_counts.get(key, 0) + 1
            if key not in merged:
                merged[key] = paper

    ranked = sorted(
        merged.items(),
        key=lambda item: (seed_counts[item[0]], item[1].get("citation_count") or 0),
        reverse=True,
    )
    out: List[Paper] = []
    for key, paper in ranked:
        paper = dict(paper)
        # Kept on the record so the ranking is auditable after the fact, and so
        # the eval harness can ask whether co-citation predicted usefulness.
        paper["cited_by_seeds"] = seed_counts[key]
        out.append(paper)
    return out


def expand(
    papers: Sequence[Paper],
    dedupe_key: Callable[[Paper], str],
    fetch_related: FetchRelated,
    *,
    directions: Sequence[str],
    max_seeds: int,
    per_seed: int,
    max_new_papers: int,
) -> List[Paper]:
    """One round of expansion: pick seeds, hop, rank, and return at most
    `max_new_papers` papers not already in `papers`.

    Deliberately one round and not a loop. Each hop multiplies the frontier, so
    a second round costs an order of magnitude more requests for candidates that
    are, by construction, two steps removed from anything the question actually
    matched.
    """
    seeds = choose_seeds(papers, max_seeds)
    if not seeds:
        logger.info("No expandable seed papers — skipping citation expansion")
        return []

    known_keys = {dedupe_key(p) for p in papers if (p.get("title") or "").strip()}

    hops: List[List[Paper]] = []
    for seed in seeds:
        identifier = semantic_scholar_identifier(seed)
        for direction in directions:
            # fetch_related never raises: a failed hop returns [] and costs only
            # its own results, so one unresolvable seed can't end the round.
            hops.append(fetch_related(identifier, direction, per_seed))

    ranked = rank_candidates(hops, dedupe_key, known_keys)
    selected = ranked[:max_new_papers]

    logger.info(
        "Citation expansion: %d seed(s) → %d new candidate(s), keeping %d",
        len(seeds), len(ranked), len(selected),
    )
    if selected:
        logger.debug(
            "Top candidates by co-citation: %s",
            [(p.get("title"), p.get("cited_by_seeds")) for p in selected[:5]],
        )
    return selected
