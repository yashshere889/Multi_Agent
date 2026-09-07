"""Cross-encoder relevance reranking over a paper pool.

The one piece of OpenScholar (https://github.com/AkariAsai/OpenScholar) that
transplants cleanly into this pipeline. Its retriever needs a 746GB local
datastore and its generator a second served model; its *reranker* is a 560M
XLM-RoBERTa cross-encoder — `OpenSciLM/OpenScholar_Reranker`, a fine-tune of
BAAI/bge-reranker-large — that scores (question, paper) pairs directly and
needs neither. Searching is recall-oriented by construction here: four sources,
three generated queries each, everything merged. Nothing has ever judged whether
a returned paper is actually *about* the question.

One factory, one memoized model — the same rule llm.py and checkpointer.py
follow, and for a sharper reason than either: the model is ~2.2GB, and agent
graphs are rebuilt per call, so loading inside a node would re-read it from disk
on every invocation.

Everything here degrades to "behave exactly as before". Reranking is off by
default; when it is on but the optional dependencies are missing, the model
cannot be fetched, or scoring raises, `rerank_papers` returns the pool it was
given, in the order it was given, and says so once. That is not a courtesy — on
a machine that cannot install torch at all (macOS x86_64, or any Python newer
than torch ships wheels for) the degraded path is the only path, and the
pipeline must not care.

Install with `uv sync --extra rerank`, which pulls torch + transformers — several
GB and thousands of inodes, hence an extra rather than a base dependency (see
the same reasoning on the `huggingface` extra in pyproject.toml, and the home
inode quota on Barkla).
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence

from research_pipeline.config import settings

logger = logging.getLogger(__name__)

# bge-reranker-large is an XLM-RoBERTa, so 512 tokens is the architectural
# ceiling rather than a tuning choice — a longer pair is truncated, which is why
# _paper_text puts the title first and the body last.
MAX_LENGTH = 512
# Pairs per forward pass. Small because this runs on CPU by default and a paper
# pool is tens of items, not thousands: a bigger batch buys nothing and costs
# peak memory on a login node.
BATCH_SIZE = 16

# (tokenizer, model), or the sentinel below once a load has failed. Reset in
# tests via reset_reranker(); see tests/conftest.py for why the other
# process-wide singletons in this repo need the same treatment.
_MODEL: Optional[tuple] = None
_LOAD_FAILED = False


def reset_reranker() -> None:
    """Drops the memoized model. Tests only — nothing in a pipeline run should
    want to reload 2.2GB mid-flight."""
    global _MODEL, _LOAD_FAILED
    _MODEL = None
    _LOAD_FAILED = False


def is_available() -> bool:
    """Whether a rerank would actually do anything, without loading anything."""
    return settings.enable_rerank and not _LOAD_FAILED


def _load() -> Optional[tuple]:
    """(tokenizer, model) or None, memoized including the failure."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    try:
        # transformers pulls torch in for a sequence-classification model and
        # raises ImportError without it, so there is nothing to check separately.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model)
        model = AutoModelForSequenceClassification.from_pretrained(settings.reranker_model)
        model.to(settings.rerank_device)
        model.eval()
        logger.info(
            "Loaded reranker %s on %s", settings.reranker_model, settings.rerank_device
        )
        _MODEL = (tokenizer, model)
        return _MODEL
    except ImportError as exc:
        logger.warning(
            "ENABLE_RERANK is on but the reranker's dependencies are missing (%s) — "
            "continuing without reranking. Install them with: uv sync --extra rerank",
            exc,
        )
    except Exception as exc:
        # A model that can't be fetched (offline compute node, no HF cache, a
        # bad RERANKER_MODEL) is exactly as survivable as a missing package.
        logger.warning(
            "Failed to load reranker %s (%s) — continuing without reranking",
            settings.reranker_model, exc,
        )
    _LOAD_FAILED = True
    return None


def score_pairs(query: str, documents: Sequence[str]) -> Optional[List[float]]:
    """Relevance logits for (query, document) pairs, or None if unavailable.

    Higher is more relevant. The raw logit is kept rather than a sigmoid: only
    the ordering is used, and a monotone squashing would just discard the range
    that makes a score readable in metadata.json.
    """
    loaded = _load()
    if loaded is None or not documents:
        return None
    tokenizer, model = loaded

    try:
        import torch

        scores: List[float] = []
        with torch.inference_mode():
            for start in range(0, len(documents), BATCH_SIZE):
                batch = list(documents[start : start + BATCH_SIZE])
                inputs = tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(settings.rerank_device)
                logits = model(**inputs).logits.view(-1).float()
                scores.extend(logits.tolist())
        return scores
    except Exception as exc:
        # Same contract as a failed load: the caller keeps its pool.
        logger.warning("Reranking failed mid-scoring (%s) — leaving the pool untouched", exc)
        return None


def _paper_text(paper: dict) -> str:
    """What the cross-encoder actually reads.

    Title first because it survives truncation, then whichever body the paper
    has — the same `full_text` or `abstract` preference agents/hypothesis/
    papers.py applies, so a paper found by passage search is judged on its
    passages rather than on an abstract it may not have.
    """
    title = (paper.get("title") or "").strip()
    body = (paper.get("full_text") or paper.get("abstract") or "").strip()
    return f"{title}\n\n{body}".strip()


def rerank_papers(
    question: Optional[str],
    papers: List[dict],
    *,
    top_k: Optional[int] = None,
    score_fn: Optional[Callable[[str, Sequence[str]], Optional[List[float]]]] = None,
) -> List[dict]:
    """Reorders `papers` by relevance to `question`, most relevant first.

    Returns the input list unchanged — same objects, same order — whenever
    reranking is off, unavailable, or has nothing to work with (no question, no
    papers, nothing with any text). Every caller can therefore treat this as an
    optional improvement rather than a step that can fail.

    Truncation is separate from ordering and off by default: `top_k` (or
    RERANK_TOP_K) of 0/None reorders and keeps everything. Dropping a paper is
    irreversible downstream — the Writer can only cite what reaches it — so the
    destructive half is opted into explicitly, while the safe half is what
    turning the feature on gets you. Each returned paper carries its
    `rerank_score`, so the ranking that produced an order is inspectable in
    metadata.json rather than implicit in it.
    """
    if not settings.enable_rerank or not papers:
        return papers
    question = (question or "").strip()
    if not question:
        logger.info("Skipping rerank: no research question to score against")
        return papers

    documents = [_paper_text(p) for p in papers]
    if not any(documents):
        logger.info("Skipping rerank: no paper in the pool has any text to score")
        return papers

    # Guarded here as well as inside score_pairs, and not redundantly: score_fn
    # is injectable, so this function cannot lean on the default scorer's own
    # error handling to keep the promise its docstring makes. This is where
    # "an optional improvement that can never fail a run" is actually enforced.
    try:
        scores = (score_fn or score_pairs)(question, documents)
    except Exception as exc:
        logger.warning("Reranking raised (%s) — leaving the pool untouched", exc)
        return papers
    if scores is None or len(scores) != len(papers):
        if scores is not None:
            logger.warning(
                "Reranker returned %d score(s) for %d paper(s) — leaving the pool untouched",
                len(scores), len(papers),
            )
        return papers

    scored = [dict(paper, rerank_score=score) for paper, score in zip(papers, scores)]
    # sorted() is stable, so papers the model scores identically keep the order
    # the merge gave them rather than an arbitrary one — the same reason every
    # other ordering decision in this repo is made in Python.
    ranked = sorted(scored, key=lambda p: p["rerank_score"], reverse=True)

    limit = settings.rerank_top_k if top_k is None else top_k
    if limit and limit > 0 and limit < len(ranked):
        logger.info(
            "Reranked %d paper(s) against the research question, keeping the top %d "
            "(score range %.3f..%.3f, cut at %.3f)",
            len(ranked), limit, ranked[0]["rerank_score"], ranked[-1]["rerank_score"],
            ranked[limit - 1]["rerank_score"],
        )
        return ranked[:limit]

    logger.info(
        "Reranked %d paper(s) against the research question (score range %.3f..%.3f)",
        len(ranked), ranked[0]["rerank_score"], ranked[-1]["rerank_score"],
    )
    return ranked
