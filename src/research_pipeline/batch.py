"""Runs the pipeline over many research questions unattended.

Built for long sweeps on a cluster, so the two things that matter are that one
bad question doesn't take the run down with it, and that a killed or pre-empted
job can be resubmitted without redoing finished work. Both come from the
manifest, which is rewritten after every question rather than at the end.

Each question is an independent graph invocation with its own thread_id and its
own output directory — nothing is shared between them but the manifest.

Resuming works at two granularities, and the manifest drives both. Between
questions, a finished question is skipped outright. Within a question, the
thread_id of every attempt is recorded, so a resubmitted job can continue a
question that was pre-empted half-way instead of redoing its finished stages —
though only when CHECKPOINTER_BACKEND is durable, since the default in-memory
checkpoints do not outlive the job that was pre-empted.
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from research_pipeline import checkpointer
from research_pipeline.config import settings
from research_pipeline.orchestrator.graph import build_pipeline_graph

logger = logging.getLogger(__name__)

MANIFEST_NAME = "batch_manifest.json"


def load_questions(path: str | Path) -> List[str]:
    """One question per line. Blank lines and #-comments are skipped, so a
    question list can be annotated."""
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _slug(question: str, index: int) -> str:
    words = re.sub(r"[^a-z0-9\s-]", "", question.lower()).split()
    return f"q{index:03d}_" + ("-".join(words[:6]) or "question")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_manifest(manifest_path: Path, entry: dict) -> None:
    """Merges one entry into the manifest under an exclusive lock. Locked
    rather than plain-written because a job array has several tasks writing
    their own slices into one shared file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read().strip()
            manifest = json.loads(raw) if raw else {"started_at": _utc_now(), "entries": []}

            entries = {e["index"]: e for e in manifest.get("entries", [])}
            entries[entry["index"]] = {**entries.get(entry["index"], {}), **entry}
            manifest["entries"] = [entries[k] for k in sorted(entries)]
            manifest["updated_at"] = _utc_now()

            handle.seek(0)
            handle.truncate()
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _manifest_entries(manifest_path: Path) -> dict[int, dict]:
    """Every entry the manifest holds, keyed by question index. Read without
    the lock the writer takes: a torn read degrades to "attempt the question
    again", which is the same safe direction as no manifest at all."""
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Ignoring an unreadable manifest at %s — every question will be attempted.", manifest_path)
        return {}
    return {e["index"]: e for e in manifest.get("entries", []) if "index" in e}


def _thread_id_for(entry: Optional[dict], question: str) -> str:
    """The thread_id to run this question under: the one a previous attempt
    recorded, so its checkpoints can be resumed, or a new one.

    The question has to match before a recorded id is reused. Indices are
    positions in a file people edit between sweeps, so entry 7 in the manifest
    is not necessarily question 7 today — and resuming the wrong thread would
    silently graft a half-finished run about one question onto another,
    producing a paper whose sections disagree about what was being asked.
    A mismatch is not an error, just a reason to start clean.
    """
    if entry and entry.get("thread_id") and entry.get("question") == question:
        return str(entry["thread_id"])
    return str(uuid.uuid4())


def run_batch(
    questions_file: str | Path,
    output_root: Optional[str | Path] = None,
    max_results_per_query: int = 5,
    download_dir_root: Optional[str | Path] = None,
    max_iterations: Optional[int] = None,
    quality_threshold: Optional[int] = None,
    max_consecutive_failures: Optional[int] = None,
    resume: bool = True,
    questions: Optional[List[tuple[int, str]]] = None,
) -> dict:
    """Runs every question in `questions_file` end to end.

    `questions` overrides which of them this call actually processes, as
    (original_index, question) pairs — that's how run_batch_slice hands a job
    array task its share while keeping indices stable across the whole batch.
    """
    output_root = Path(output_root or settings.batch_output_root)
    download_dir_root = Path(download_dir_root) if download_dir_root else output_root / "papers"
    max_consecutive_failures = (
        settings.batch_max_consecutive_failures if max_consecutive_failures is None else max_consecutive_failures
    )
    manifest_path = output_root / MANIFEST_NAME

    selected = questions if questions is not None else list(enumerate(load_questions(questions_file)))
    previous = _manifest_entries(manifest_path) if resume else {}
    already_done = {index for index, entry in previous.items() if entry.get("status") == "completed"}

    graph = build_pipeline_graph()
    completed = failed = skipped = 0
    consecutive_failures = 0

    for index, question in selected:
        if index in already_done:
            skipped += 1
            logger.info("Skipping question %d (already completed)", index)
            continue

        slug = _slug(question, index)
        question_output_dir = output_root / slug
        question_download_dir = download_dir_root / slug
        # Recorded in the manifest so the *next* process can find it. A fresh
        # uuid4 per attempt, as this used to be unconditionally, is a thread_id
        # nothing can ever resume: the checkpoints from the pre-empted attempt
        # are still on disk under an id no later run will ever ask for.
        thread_id = _thread_id_for(previous.get(index), question)
        _update_manifest(
            manifest_path,
            {
                "index": index,
                "question": question,
                "output_dir": str(question_output_dir),
                "thread_id": thread_id,
                "status": "running",
                "started_at": _utc_now(),
                "finished_at": None,
                "final_paper_path": None,
                "error": None,
            },
        )

        logger.info("[%d/%d] %s", index + 1, len(selected), question)
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # Empty unless a durable CHECKPOINTER_BACKEND is configured *and*
            # an earlier attempt on this thread stopped part-way: the manifest
            # resumes at whole-question granularity, this resumes within one,
            # so a question pre-empted during the writer doesn't redo its
            # literature search, hypotheses and experiments.
            resuming = checkpointer.pending_nodes(graph, config)
            if resuming:
                logger.info("Resuming question %d at %s", index, ", ".join(resuming))
            state = graph.invoke(
                None
                if resuming
                else {
                    "research_question": question,
                    "max_results_per_query": max_results_per_query,
                    "download_dir": str(question_download_dir),
                    "metadata_path": str(question_download_dir / "metadata.json"),
                    "output_dir": str(question_output_dir),
                    "max_iterations": max_iterations,
                    "quality_threshold": quality_threshold,
                },
                config=config,
            )
            result = state["final_result"]
        except Exception as exc:  # noqa: BLE001
            # The only broad catch in the pipeline. Everywhere else an
            # exception should surface; here it must not, or one unlucky
            # question ends a run that still has ninety-nine to go.
            logger.exception("Question %d failed: %s", index, question)
            failed += 1
            consecutive_failures += 1
            _update_manifest(
                manifest_path,
                {"index": index, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": _utc_now()},
            )
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    "Stopping: %d question(s) failed in a row. Remaining questions are left pending for a later re-run.",
                    consecutive_failures,
                )
                break
            continue

        completed += 1
        consecutive_failures = 0
        _update_manifest(
            manifest_path,
            {
                "index": index,
                "status": "completed",
                "final_paper_path": result["final_paper_path"],
                "converged": result["converged"],
                "iterations_run": result["iterations_run"],
                "finished_at": _utc_now(),
            },
        )

    return {
        "manifest_path": str(manifest_path),
        "total": len(selected),
        "completed": completed,
        "failed": failed,
        "skipped_already_done": skipped,
        "generated_at": _utc_now(),
    }


def run_batch_slice(
    questions_file: str | Path,
    slice_index: int,
    slice_count: int,
    output_root: Optional[str | Path] = None,
    **kwargs,
) -> dict:
    """Processes one contiguous share of the question list, for a SLURM job
    array. Indices stay those of the full list so every task's entries land in
    the shared manifest without colliding."""
    if not 0 <= slice_index < slice_count:
        raise ValueError(f"slice_index must be in [0, {slice_count}), got {slice_index}")

    all_questions = list(enumerate(load_questions(questions_file)))
    total = len(all_questions)
    start = slice_index * total // slice_count
    end = (slice_index + 1) * total // slice_count

    logger.info("Slice %d/%d covers questions %d-%d of %d", slice_index, slice_count, start, max(end - 1, start), total)
    return run_batch(questions_file, output_root=output_root, questions=all_questions[start:end], **kwargs)
