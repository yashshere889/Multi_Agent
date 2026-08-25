"""Runs the literature stack over a gold set and scores what it returned.

The point of this module is to make "did that change help?" an answerable
question. Before it, every judgment about the search agents was an impression
formed by reading a few titles.

Search once, score many times
-----------------------------
The naive design runs the pipeline twice — screen on, screen off — and diffs the
two. This doesn't do that, for two reasons. Searching is the slow, rate-limited,
non-deterministic part, so running it twice doubles the cost *and* introduces a
second sample of API flakiness into the very comparison meant to isolate one
variable. And a screen applied offline over a saved pool can be applied at every
threshold at once, which answers the more useful question — not "is 3 better
than nothing" but "what does each threshold actually cost and buy".

So `run_gold_set` searches once with the screen disabled, saves the whole raw
pool, and `score_run` replays the screen over it afterwards using the very same
`relevance.score_papers`/`apply_threshold` production calls. Post-screen numbers
are therefore what production would really produce, and re-scoring a saved run
costs nothing and touches no network.

What is deliberately switched off during a run
----------------------------------------------
Both overrides live in `_eval_mode` and neither changes what is being measured:

  * The relevance screen, because the harness applies it itself, at every
    threshold, from the saved pool.
  * PDF downloading, because the eval measures *retrieval*, not fetching —
  `pdf_url` already records whether a PDF was available, and downloading tens of
  papers per question would dominate the runtime of the loop this exists to make
  fast.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence
from uuid import uuid4

from research_pipeline.agents.literature import graph as literature_graph
from research_pipeline.agents.literature import nodes as literature_nodes
from research_pipeline.agents.literature.graph import build_literature_graph
from research_pipeline.agents.literature.relevance import (
    DIRECT_RELEVANCE_CRITERION,
    apply_threshold,
    score_papers,
)
from research_pipeline.config import settings
from research_pipeline.eval import judge as judge_module
from research_pipeline.eval import metrics
from research_pipeline.llm import get_chat_model

logger = logging.getLogger(__name__)

# Thresholds swept by default: the whole usable range of the 0-5 rubric. 0 is
# "keep everything", i.e. the pre-screen baseline, and is always included so
# every other row has something to be compared against.
DEFAULT_THRESHOLDS = (0, 2, 3, 4)


@contextmanager
def _eval_mode():
    """Disables the in-pipeline relevance screen and PDF downloading for the
    duration of a run. See this module's docstring for why each is safe.

    Patching module-level `settings` is how the tests do it too — `settings` is
    a frozen dataclass built once at import, so an env var set now would not be
    seen by an already-imported module.
    """
    original_settings = literature_nodes.settings
    original_download = literature_graph.download_papers_node
    literature_nodes.settings = replace(original_settings, enable_relevance_filter=False)
    literature_graph.download_papers_node = lambda state: {}
    try:
        yield
    finally:
        literature_nodes.settings = original_settings
        literature_graph.download_papers_node = original_download


def run_question(entry: dict, *, max_results: int, output_dir: Path) -> dict:
    """Searches for one gold entry's question and returns the raw record.

    Failures are caught and recorded rather than raised: a gold set is run
    unattended and one question that trips a rate limit must not cost the other
    nine their results. Same reason batch.py swallows a stage-level exception.
    """
    question = entry["question"]
    started = time.monotonic()
    record: dict = {
        "question": question,
        "gold_total": len(entry.get("papers") or []),
        "gold_source": entry.get("source"),
    }

    try:
        graph = build_literature_graph()
        result = graph.invoke(
            {
                "research_question": question,
                "max_results_per_query": max_results,
                "download_dir": str(output_dir / "papers"),
                "metadata_path": str(output_dir / "papers" / "metadata.json"),
            },
            config={"configurable": {"thread_id": str(uuid4())}},
        )
    except Exception as exc:
        logger.error("Literature search failed for %r: %s", question, exc)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["literature_output"] = {"merged_papers": []}
    else:
        record["literature_output"] = {
            "merged_papers": result.get("merged_papers") or [],
            "search_queries": result.get("search_queries") or [],
        }

    record["elapsed_seconds"] = round(time.monotonic() - started, 1)
    return record


def run_gold_set(
    entries: Sequence[dict],
    *,
    name: str,
    max_results: Optional[int] = None,
    output_dir: str | Path = "evals/runs",
) -> dict:
    """Searches every gold entry's question and returns a saveable run record.

    The record holds the complete returned pool per question, which is what
    makes `score_run` replayable offline — re-scoring at a new threshold, or
    with a metric added later, never re-searches.
    """
    max_results = max_results or settings.default_max_results_per_query
    target = Path(output_dir)
    run: dict = {
        "name": name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        # Snapshotted so a run file is self-describing: two runs are only
        # comparable if these match, and six months later nothing else records
        # what the search was configured to do.
        "config": {
            "max_results_per_query": max_results,
            "llm_model": settings.llm_model,
            "semantic_scholar_enabled": bool(settings.semantic_scholar_api_key),
            "core_enabled": bool(settings.core_api_key),
            "note": "searched with the in-pipeline relevance screen disabled; screening is applied by score_run",
        },
        "questions": [],
    }

    with _eval_mode():
        for i, entry in enumerate(entries, start=1):
            logger.info("[%d/%d] %s", i, len(entries), entry["question"])
            run["questions"].append(run_question(entry, max_results=max_results, output_dir=target))

    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    return run


def _screen_pool(question: str, papers: List[dict], chat_model) -> List[dict]:
    """Annotates a saved pool with relevance scores, using the production
    scorer. Returns the papers with `relevance_score` set."""
    scores = score_papers(
        chat_model,
        question,
        papers,
        criterion=DIRECT_RELEVANCE_CRITERION,
        batch_max_chars=settings.relevance_batch_max_chars,
    )
    # keep_min=0 and min_score=0 keeps every paper while still attaching the
    # score each was judged on — the sweep below decides what to drop, so
    # nothing may be dropped here.
    kept, _ = apply_threshold(papers, scores, min_score=0, keep_min=0)
    return kept


def score_run(
    run: dict,
    gold_by_question: dict,
    *,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
    screen: bool = True,
    judge: bool = False,
    chat_model=None,
) -> dict:
    """Computes every metric for a saved run. Touches the network only when
    `screen` or `judge` is on (both need the LLM); the deterministic metrics are
    computed regardless and never do.
    """
    if (screen or judge) and chat_model is None:
        chat_model = get_chat_model(temperature=0.0)

    scored: dict = {"name": run.get("name"), "config": run.get("config"), "questions": []}

    for record in run.get("questions", []):
        question = record["question"]
        gold = gold_by_question.get(question, [])
        papers = list((record.get("literature_output") or {}).get("merged_papers") or [])

        if screen and papers:
            papers = _screen_pool(question, papers, chat_model)

        row: dict = {
            "question": question,
            "error": record.get("error"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            **metrics.literature_metrics(gold, {**record.get("literature_output", {}), "merged_papers": papers}),
        }

        verdicts: List[Optional[bool]] = []
        if judge and papers:
            verdicts = judge_module.judge_pool(chat_model, question, papers)
            row["judged_precision"] = judge_module.precision(verdicts)
            row["judged"] = sum(1 for v in verdicts if v is not None)
            # The number that says whether the screen itself works, as opposed
            # to whether the pool happened to be clean.
            row["screen_agreement"] = judge_module.agreement_with_screen(
                papers, verdicts, settings.relevance_min_score
            )

        # The sweep: what each threshold would have kept, and what that costs in
        # recall. This is the table that answers "which RELEVANCE_MIN_SCORE".
        row["sweep"] = _sweep(gold, papers, verdicts, thresholds) if papers else []

        if record.get("interdisciplinary_output"):
            row["interdisciplinary"] = metrics.interdisciplinary_metrics(record["interdisciplinary_output"])

        scored["questions"].append(row)

    scored["aggregate"] = metrics.aggregate(scored["questions"])
    scored["sweep"] = _aggregate_sweep(scored["questions"], thresholds)
    return scored


def _sweep(gold, papers, verdicts, thresholds) -> List[dict]:
    rows = []
    for threshold in thresholds:
        keep_flags = [
            not isinstance(p.get("relevance_score"), int) or p["relevance_score"] >= threshold
            for p in papers
        ]
        kept = [p for p, keep in zip(papers, keep_flags) if keep]
        row = {
            "threshold": threshold,
            "kept": len(kept),
            "dropped": len(papers) - len(kept),
            "recall": metrics.recall(gold, kept),
        }
        if verdicts:
            kept_verdicts = [v for v, keep in zip(verdicts, keep_flags) if keep]
            row["judged_precision"] = judge_module.precision(kept_verdicts)
            # Papers an independent judge would have kept that this threshold
            # throws away — the cost side of the trade, and the number to watch
            # when raising RELEVANCE_MIN_SCORE.
            row["lost_good_papers"] = sum(
                1 for v, keep in zip(verdicts, keep_flags) if v is True and not keep
            )
        rows.append(row)
    return rows


def _aggregate_sweep(rows: Sequence[dict], thresholds: Sequence[int]) -> List[dict]:
    aggregated = []
    for i, threshold in enumerate(thresholds):
        per_question = [r["sweep"][i] for r in rows if len(r.get("sweep") or []) > i]
        if not per_question:
            continue

        def mean(key):
            values = [q[key] for q in per_question if isinstance(q.get(key), (int, float))]
            return sum(values) / len(values) if values else None

        aggregated.append({
            "threshold": threshold,
            "mean_kept": mean("kept"),
            "mean_recall": mean("recall"),
            "judged_precision": mean("judged_precision"),
            "lost_good_papers": sum(q.get("lost_good_papers", 0) for q in per_question),
        })
    return aggregated


def compare(baseline: dict, candidate: dict) -> dict:
    """Per-metric deltas between two scored runs, matched on question text."""
    baseline_rows = {r["question"]: r for r in baseline.get("questions", [])}
    keys = ("recall", "pool_size", "abstract_coverage", "mean_relevance_score", "judged_precision")

    per_question = []
    for row in candidate.get("questions", []):
        before = baseline_rows.get(row["question"])
        if before is None:
            continue
        per_question.append({
            "question": row["question"],
            **{
                key: {
                    "before": before.get(key),
                    "after": row.get(key),
                    "delta": (row[key] - before[key])
                    if isinstance(row.get(key), (int, float)) and isinstance(before.get(key), (int, float))
                    else None,
                }
                for key in keys
            },
        })

    return {
        "baseline": baseline.get("name"),
        "candidate": candidate.get("name"),
        "questions": per_question,
        "unmatched": [
            r["question"] for r in candidate.get("questions", []) if r["question"] not in baseline_rows
        ],
    }


def write_run(run: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    return target


def read_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
