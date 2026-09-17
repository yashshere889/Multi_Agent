#!/usr/bin/env python3
"""Populate the web UI with runs to look at, without an LLM endpoint attached.

Two different things live here, and the difference matters:

`import` takes a run that really happened — a Kaggle or Barkla run whose
outputs are sitting in a directory — and materialises it into the layout the
web server expects (`runs/<uuid>/` with `run.json`, `events.jsonl`, `outputs/`,
`papers/`, `experiments/`). Nothing is invented: every stage event is built by
handing that run's own output files to `webapp.stages.summarize`, the same
function the live runner calls on the orchestrator's deltas. The only rewriting
is of absolute `code_path` values, which pointed at `/kaggle/working` and have
to point inside this run directory for the artifact browser to resolve them.

`fixtures` builds synthetic runs, and they are synthetic on purpose: they cover
UI surfaces that no run currently on disk happens to exercise — a run stopped
mid-pipeline so the Continue control appears, and an experiment that completed
with a withheld verdict so the provenance panel appears. Their numbers are
made up. They are for showing what the interface does, never for showing what
the system found.

    uv run python scripts/seed_demo_runs.py import outputs/kaggle_run_temporal_spatial_logic
    uv run python scripts/seed_demo_runs.py fixtures
    uv run python scripts/seed_demo_runs.py list
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_pipeline.webapp import events, stages  # noqa: E402
from research_pipeline.writer_reviewer_loop import _consolidate_unresolved  # noqa: E402

RUNS_DIR = REPO_ROOT / "runs"

# Marks a run directory this script created, so `list` can tell the two kinds
# apart and a later --clean knows what it is allowed to delete. The web server
# neither reads nor cares about this key.
ORIGIN_IMPORTED = "imported"
ORIGIN_FIXTURE = "fixture"


def _newest(directory: Path, pattern: str) -> Path | None:
    found = sorted(p for p in directory.glob(pattern) if not p.stem.endswith("_invalid"))
    return found[-1] if found else None


def _load(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _new_run_dir() -> tuple[str, Path]:
    run_id = str(uuid.uuid4())
    run_dir = RUNS_DIR / run_id
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "papers").mkdir(parents=True, exist_ok=True)
    (run_dir / "experiments").mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _write_run_json(run_dir: Path, record: dict) -> None:
    (run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# import: a run that really happened


def _rewrite_code_paths(summary: dict, run_dir: Path) -> dict:
    """Point `code_path` at this run's own experiments directory.

    The Coder Agent records where it actually wrote, which on Kaggle was
    `/kaggle/working/experiments/H1`. `webapp.experiments._relative_to_run`
    resolves a path against the run directory and returns None when it falls
    outside — which is correct behaviour and means every file link and code
    snapshot on the Experiments page would silently disappear. The directory
    contents are unchanged; only the recorded location is corrected.
    """
    experiments_root = run_dir / "experiments"
    for experiment in summary.get("experiments") or []:
        hid = experiment.get("hypothesis_id")
        if not hid:
            continue
        experiment["code_path"] = str(experiments_root / hid)
        for entry in experiment.get("fix_history") or []:
            attempt = entry.get("attempt")
            if attempt:
                entry["code_path"] = str(experiments_root / hid / "fix_attempts" / f"attempt_{attempt}" / "run.py")
    return summary


def import_run(source: Path, question: str | None) -> str:
    outputs_src = source / "outputs" if (source / "outputs").is_dir() else source
    papers_src = source / "papers"
    experiments_src = source / "experiments"

    metadata = _load(papers_src / "metadata.json") or {}
    hypotheses = _load(_newest(outputs_src, "hypotheses_*.json"))
    plan = _load(_newest(outputs_src, "experiment_plan_*.json"))
    coder = _load(_newest(outputs_src, "coder_agent_summary_*.json"))
    reviews = [_load(p) for p in sorted(outputs_src.glob("review_2*.json"))]
    reviews = [r for r in reviews if r]
    drafts = [_load(outputs_src / f"v{n}_summary.json") for n in (1, 2, 3)]
    drafts = [d for d in drafts if d]

    question = question or metadata.get("research_question") or "(research question not recorded)"
    run_id, run_dir = _new_run_dir()

    for src, dest in ((outputs_src, run_dir / "outputs"), (papers_src, run_dir / "papers"), (experiments_src, run_dir / "experiments")):
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)

    if coder:
        rewritten = _rewrite_code_paths(json.loads(json.dumps(coder)), run_dir)
        target = run_dir / "outputs" / _newest(outputs_src, "coder_agent_summary_*.json").name
        target.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")

    log = events.EventLog(run_dir)
    # Each stage event is stamped with the moment that stage's own output file
    # says it was generated, so the timeline in the UI is the run's real
    # timeline rather than the moment this script happened to run.
    timeline: list[str] = []

    def stamp(source_doc: dict | None, fallback: str | None = None) -> str:
        when = (source_doc or {}).get("generated_at") or fallback or events.utc_now()
        timeline.append(when)
        return when

    started = metadata.get("generated_at") or events.utc_now()
    timeline.append(started)
    log.append(events.RUN_STARTED, ts=started, question=question, params={"imported_from": str(source)})

    stages_done: list[str] = []

    def stage(name: str, delta: dict, key: str, doc: dict | None) -> None:
        log.append(events.STAGE_COMPLETED, ts=stamp(doc), stage=name, summary=stages.summarize(name, delta))
        stages_done.append(key)

    if metadata.get("papers"):
        stage(
            stages.LITERATURE,
            {"literature_output": {"merged_papers": metadata["papers"], "search_queries": metadata.get("search_queries") or []}},
            "literature_output",
            metadata,
        )
    if hypotheses:
        stage(stages.HYPOTHESIS, {"hypothesis_output": hypotheses}, "hypothesis_output", hypotheses)
    if plan:
        stage(stages.EXPERIMENT_PLANNER, {"planner_output": plan}, "planner_output", plan)
    if coder:
        stage(stages.CODER, {"coder_output": coder}, "coder_output", coder)

    for index, draft in enumerate(drafts, start=1):
        log.append(
            events.STAGE_COMPLETED,
            ts=stamp(draft),
            stage=stages.DRAFT_OR_REVISE,
            summary=stages.summarize(stages.DRAFT_OR_REVISE, {"paper_summary": draft, "iteration": index}),
        )
        if index <= len(reviews):
            review = reviews[index - 1]
            log.append(
                events.STAGE_COMPLETED,
                ts=stamp(review),
                stage=stages.REVIEW,
                summary=stages.summarize(
                    stages.REVIEW,
                    {"review": review, "converged": bool(review.get("overall_pass"))},
                ),
            )

    finished = max(timeline)
    if drafts:
        # The shape finalize_node returns for a run that reached the
        # Writer/Reviewer stage. It deliberately carries no `stages_completed`:
        # that key is the *partial*-run branch, and the UI reads it as "this run
        # stopped early, offer to continue it".
        converged = bool(reviews and reviews[-1].get("overall_pass"))
        final_result = {
            "final_paper_path": str(run_dir / "outputs" / Path(str(drafts[-1].get("paper_path") or "v1.pdf")).name),
            "iterations_run": len(drafts),
            "converged": converged,
            "unresolved_issues": [] if converged or not reviews else _consolidate_unresolved(reviews[-1], 4),
            "review_history_path": str(run_dir / "outputs" / "review_log.json"),
            "generated_at": finished,
        }
    else:
        final_result = {"stages_completed": stages_done}
    log.append(events.RUN_COMPLETED, ts=finished, final_result=final_result)

    _write_run_json(
        run_dir,
        {
            "run_id": run_id,
            "question": question,
            "params": {"imported_from": str(source), "origin": ORIGIN_IMPORTED},
            "status": "completed",
            "pid": None,
            "created_at": started,
            "started_at": started,
            "finished_at": finished,
            "final_result": final_result,
            "error": None,
        },
    )
    return run_id


# --------------------------------------------------------------------------
# fixtures: synthetic, for UI surfaces nothing on disk covers


def _fixture_stopped_after_hypothesis() -> str:
    """A run that stopped mid-pipeline, so the Continue control has something
    to act on. Nothing downstream of the Hypothesis stage exists here."""
    question = "Does curriculum ordering speed up reinforcement-learning fine-tuning?"
    run_id, run_dir = _new_run_dir()
    started = datetime.now(timezone.utc) - timedelta(minutes=18)

    papers = [
        {"paper_id": f"P{i}", "title": t, "local_path": f"papers/{i}.pdf", "source": src}
        for i, (t, src) in enumerate(
            [
                ("Curriculum Learning for Reinforcement Learning Domains", "arxiv"),
                ("Automatic Curricula via Asymmetric Self-Play", "arxiv"),
                ("Teacher-Student Curriculum Learning", "semantic_scholar"),
                ("On the Sample Complexity of Ordered Training", "core"),
                ("Reverse Curriculum Generation for RL", "arxiv"),
            ],
            start=1,
        )
    ]
    hypothesis_output = {
        "hypotheses": [
            {"id": "H1", "statement": "Curriculum ordering helps only when the task distribution has a difficulty gradient the policy can detect."},
            {"id": "H2", "statement": "Ordering gains vanish once the replay buffer is large enough to mix difficulties anyway."},
            {"id": "H3", "statement": "Reverse curricula outperform forward curricula on sparse-reward tasks."},
        ],
        "ranking": [
            {"hypothesis_id": "H1", "rank": 1, "score": 8.4},
            {"hypothesis_id": "H3", "rank": 2, "score": 7.1},
            {"hypothesis_id": "H2", "rank": 3, "score": 6.5},
        ],
        "selected_hypothesis_id": "H1",
        "gaps": ["no controlled study isolates ordering from budget", "sparse-reward results are anecdotal"],
        "source_paper_ids": [p["paper_id"] for p in papers],
    }

    log = events.EventLog(run_dir)
    log.append(events.RUN_STARTED, question=question, params={"end_stage": "hypothesis"})
    log.append(
        events.STAGE_COMPLETED,
        stage=stages.LITERATURE,
        summary=stages.summarize(
            stages.LITERATURE,
            {"literature_output": {"merged_papers": papers, "search_queries": ["curriculum learning reinforcement learning", "task ordering sample efficiency"]}},
        ),
    )
    log.append(
        events.STAGE_COMPLETED,
        stage=stages.HYPOTHESIS,
        summary=stages.summarize(stages.HYPOTHESIS, {"hypothesis_output": hypothesis_output}),
    )
    final_result = {"stages_completed": ["literature_output", "hypothesis_output"]}
    log.append(events.RUN_COMPLETED, final_result=final_result)

    (run_dir / "outputs" / "hypotheses_fixture.json").write_text(json.dumps(hypothesis_output, indent=2), encoding="utf-8")
    _write_run_json(
        run_dir,
        {
            "run_id": run_id,
            "question": question,
            "params": {"end_stage": "hypothesis", "include_interdisciplinary": False, "origin": ORIGIN_FIXTURE},
            "status": "completed",
            "pid": None,
            "created_at": started.isoformat(),
            "started_at": started.isoformat(),
            "finished_at": (started + timedelta(minutes=11)).isoformat(),
            "final_result": final_result,
            "error": None,
        },
    )
    return run_id


def _fixture_withheld_verdict() -> str:
    """An experiment that ran to completion on a synthetic input, so the
    Experiments page shows a withheld verdict beside real-looking metrics —
    the one screen that demonstrates `provenance.apply_to_results`, and the one
    no run currently on disk produces."""
    question = "Do retrieval-augmented language models reduce hallucination on long-form QA?"
    run_id, run_dir = _new_run_dir()
    started = datetime.now(timezone.utc) - timedelta(minutes=42)
    h1_dir = run_dir / "experiments" / "H1"
    (h1_dir / "fix_attempts" / "attempt_1").mkdir(parents=True, exist_ok=True)
    (h1_dir / "fix_attempts" / "attempt_2").mkdir(parents=True, exist_ok=True)

    provenance = {
        "surrogate_count": 1,
        "methodological_validity": "The pipeline is exercised end to end, but the faithfulness labels are generated rather than human-annotated, so these numbers describe the method's behaviour on a surrogate and not the hypothesis.",
        "inputs": [
            {
                "name": "long-form QA answers with sentence-level faithfulness labels",
                "kind": "synthetic_surrogate",
                "detail": "No openly downloadable dataset pairs long-form answers with sentence-level faithfulness labels; a labelled surrogate was generated instead.",
            },
            {"name": "Wikipedia passage index", "kind": "real_download", "detail": "hf://wikipedia/20220301.en"},
        ],
    }
    results = {
        "metrics": {"faithfulness_f1": 0.71, "attribution_precision": 0.64, "n_examples": 1200},
        "meets_success_criteria": "unknown",
        "model_reported_meets_success_criteria": True,
        "verdict_withheld_because": "One or more inputs are synthetic surrogates, so these metrics say nothing about the real-world hypothesis.",
        "notes": "Ran to completion in 84s on CPU.",
    }
    (h1_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (h1_dir / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (h1_dir / "requirements.txt").write_text("numpy\nscikit-learn\n", encoding="utf-8")
    (h1_dir / "README.md").write_text(
        "# H1\n\nSentence-level attribution is approximated by the best-matching retrieved passage.\n"
        "The faithfulness labels are synthesized; see data_provenance.json.\n",
        encoding="utf-8",
    )
    program = (
        '"""H1 — does sentence-level attribution predict faithfulness?"""\n\n'
        "import json\nfrom pathlib import Path\n\n\n"
        "def load_data():\n"
        "    path = Path('long_form_qa.csv')\n"
        "    if not path.exists():\n"
        "        return synthesize()\n"
        "    import pandas as pd\n"
        "    return pd.read_csv(path)\n"
    )
    (h1_dir / "run.py").write_text(program, encoding="utf-8")
    (h1_dir / "fix_attempts" / "attempt_1" / "run.py").write_text(
        "def load_data():\n    import pandas as pd\n    return pd.read_csv('long_form_qa.csv')\n", encoding="utf-8"
    )
    (h1_dir / "fix_attempts" / "attempt_2" / "run.py").write_text(program, encoding="utf-8")

    coder_output = {
        "experiments": [
            {
                "hypothesis_id": "H1",
                "status": "completed",
                "reason": "",
                "starter_used": "sklearn_classification",
                "code_path": str(h1_dir),
                "assumptions_made": [
                    "Sentence-level attribution approximated by the best-matching retrieved passage",
                    "Held-out split fixed at 20% with seed 0",
                ],
                "results": results,
                "data_provenance": provenance,
                "fix_attempts": 2,
                "fix_history": [
                    {
                        "attempt": 1,
                        "error_source": "missing_data_fallback",
                        "error_summary": "load_data() calls pandas.read_csv('long_form_qa.csv') with no guard and nothing to fall back on if the file is absent.",
                        "code_path": str(h1_dir / "fix_attempts" / "attempt_1" / "run.py"),
                        "resolved": True,
                    },
                    {
                        "attempt": 2,
                        "error_source": "run_experiment",
                        "error_summary": 'Traceback (most recent call last):\n  File "run.py", line 88, in evaluate\n    score = matched / total\nZeroDivisionError: division by zero',
                        "code_path": str(h1_dir / "fix_attempts" / "attempt_2" / "run.py"),
                        "resolved": True,
                    },
                ],
                "slurm_job_id": None,
            },
            {
                "hypothesis_id": "H2",
                "status": "code_generated_not_run",
                "reason": "Plan needs a GPU and none was detected in this process; run.sbatch written for review instead.",
                "starter_used": "torch_training",
                "code_path": str(run_dir / "experiments" / "H2"),
                "assumptions_made": [],
                "results": None,
                "fix_attempts": 0,
                "fix_history": [],
                "slurm_job_id": None,
            },
        ],
        "source_hypothesis_ids": ["H1", "H2"],
    }
    h2_dir = run_dir / "experiments" / "H2"
    h2_dir.mkdir(parents=True, exist_ok=True)
    (h2_dir / "run.py").write_text("# generated experiment for H2 — needs a GPU\n", encoding="utf-8")
    (h2_dir / "run.sbatch").write_text(
        "#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --time=02:00:00\n\npython run.py\n",
        encoding="utf-8",
    )
    (run_dir / "outputs" / "coder_agent_summary_fixture.json").write_text(json.dumps(coder_output, indent=2), encoding="utf-8")

    log = events.EventLog(run_dir)
    log.append(events.RUN_STARTED, question=question, params={"end_stage": "coder"})
    log.append(
        events.STAGE_COMPLETED,
        stage=stages.CODER,
        summary=stages.summarize(stages.CODER, {"coder_output": coder_output}),
    )
    final_result = {"stages_completed": ["coder_output"]}
    log.append(events.RUN_COMPLETED, final_result=final_result)

    _write_run_json(
        run_dir,
        {
            "run_id": run_id,
            "question": question,
            "params": {"end_stage": "coder", "origin": ORIGIN_FIXTURE},
            "status": "completed",
            "pid": None,
            "created_at": started.isoformat(),
            "started_at": started.isoformat(),
            "finished_at": (started + timedelta(minutes=26)).isoformat(),
            "final_result": final_result,
            "error": None,
        },
    )
    return run_id


def list_runs() -> None:
    if not RUNS_DIR.is_dir():
        print("no runs directory yet")
        return
    for record_path in sorted(RUNS_DIR.glob("*/run.json")):
        record = _load(record_path) or {}
        origin = (record.get("params") or {}).get("origin") or "live"
        print(f"{origin:9s} {record.get('status',''):10s} {record_path.parent.name}  {record.get('question','')[:64]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="materialise a real completed run into runs/")
    imp.add_argument("source", type=Path, help="directory holding that run's outputs/ (and papers/, experiments/)")
    imp.add_argument("--question", default=None, help="override the research question shown in the UI")

    sub.add_parser("fixtures", help="create the synthetic runs that cover UI-only surfaces")
    sub.add_parser("list", help="show every run directory and where it came from")

    args = parser.parse_args()

    if args.command == "import":
        source = args.source if args.source.is_absolute() else REPO_ROOT / args.source
        if not source.is_dir():
            print(f"no such directory: {source}", file=sys.stderr)
            return 1
        run_id = import_run(source, args.question)
        print(f"imported (real run)  {run_id}")
    elif args.command == "fixtures":
        print(f"fixture (synthetic)  {_fixture_stopped_after_hypothesis()}   stopped after Hypothesis — Continue control")
        print(f"fixture (synthetic)  {_fixture_withheld_verdict()}   completed Coder — withheld-verdict panel")
    else:
        list_runs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
