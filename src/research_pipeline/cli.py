"""Command-line entry point.

Each agent gets its own subcommand, e.g.:
    research-pipeline literature "research question"
    research-pipeline hypothesis --from-file papers/metadata.json

`orchestrate` runs every stage end to end in one command, via the LangGraph
orchestrator in orchestrator/graph.py; the per-agent subcommands stay useful
for running a single stage or resuming from a previous run's JSON on disk.

Add a new agent by giving it its own subpackage under agents/<name>/ and
wiring a subparser for it here, following run_literature_agent (LangGraph
StateGraph agent) or run_hypothesis_agent_cli (plain callable agent) below —
either style is fine, pick whichever fits the agent's control flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from research_pipeline import batch
from research_pipeline.agents.coder import run_coder_agent
from research_pipeline.config import settings
from research_pipeline.agents.experiment_planner import run_experiment_planner_agent
from research_pipeline.agents.hypothesis import run_hypothesis_agent
from research_pipeline.agents.interdisciplinary_literature import (
    run_interdisciplinary_literature_agent,
)
from research_pipeline.agents.literature.graph import build_literature_graph
from research_pipeline.agents.reviewer import run_reviewer_agent
from research_pipeline.agents.writer import run_writer_agent
from research_pipeline.orchestrator.graph import STAGE_FOR_OUTPUT_KEY, build_pipeline_graph
from research_pipeline.writer_reviewer_loop import run_writer_reviewer_loop

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_literature_agent(args: argparse.Namespace) -> None:
    agent = build_literature_graph()
    run_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(
        {
            "research_question": args.question,
            "max_results_per_query": args.max_results,
            "download_dir": args.download_dir,
            "metadata_path": f"{args.download_dir}/metadata.json",
        },
        config=run_config,
    )

    print(f"\nFound {len(result['merged_papers'])} unique papers")
    for paper in result["merged_papers"]:
        print(
            f"- [{paper['source']}] {paper['title']} ({paper.get('year')}) -> {paper.get('local_path')}"
        )


def _papers_from_file(path: str) -> tuple[list, str | None]:
    """Reads either a Literature Agent metadata.json / Interdisciplinary
    Literature Agent output ({"papers": [...], "research_question": ...}) or a
    bare JSON list of paper dicts — both agents' outputs put the paper list
    under the same key, which is what makes them interchangeable here."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return data.get("papers", []), data.get("research_question")
    return data, None


def run_interdisciplinary_literature_agent_cli(args: argparse.Namespace) -> None:
    papers, research_question = _papers_from_file(args.from_file)
    result = run_interdisciplinary_literature_agent(
        papers, research_question=research_question, output_dir=args.output_dir
    )

    print(f"\n{len(result['fields_explored'])} adjacent field(s) explored:")
    for field in result["fields_explored"]:
        print(f"- {field['field']}: {field['rationale']}")
        print(f"    queries: {', '.join(field['queries'])}")

    print(
        f"\n{len(result['cross_field_papers'])} cross-field paper(s) added to {len(result['core_paper_ids'])} in-domain paper(s):"
    )
    for paper in result["cross_field_papers"]:
        print(f"- [{paper['discipline']}] {paper['title']} ({paper.get('year')})")

    print(f"\n{len(result['bridge_insights'])} bridge insight(s):")
    for insight in result["bridge_insights"]:
        print(f"- ({insight['source_field']}) {insight['insight']}")
        print(f"    connection: {insight['connection_to_core_problem']}")


def run_hypothesis_agent_cli(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.from_file).read_text())
    # accepts a Literature Agent metadata.json, an Interdisciplinary Literature
    # Agent output (same "papers" key, plus bridge_insights this agent can use),
    # or a bare JSON list of paper dicts
    if isinstance(data, dict):
        papers = data.get("papers", [])
        research_question = data.get("research_question")
        interdisciplinary_context = data.get("bridge_insights")
    else:
        papers = data
        research_question = None
        interdisciplinary_context = None

    result = run_hypothesis_agent(
        papers,
        research_question=research_question,
        output_dir=args.output_dir,
        interdisciplinary_context=interdisciplinary_context,
    )

    print("\nLiterature summary:")
    print(result["literature_summary"])
    print(f"\n{len(result['hypotheses'])} hypotheses generated:")
    for hypothesis in result["hypotheses"]:
        print(f"- [{hypothesis['id']}] {hypothesis['statement']}")

    print("\nRanking:")
    for entry in sorted(result["ranking"], key=lambda e: e["rank"]):
        print(
            f"  {entry['rank']}. [{entry['hypothesis_id']}] score={entry['score']} — {entry['justification']}"
        )
    print(f"\nSelected: {result['selected_hypothesis_id']}")


def run_experiment_planner_agent_cli(args: argparse.Namespace) -> None:
    hypothesis_output = json.loads(Path(args.from_file).read_text())
    result = run_experiment_planner_agent(hypothesis_output, output_dir=args.output_dir)

    print(f"\n{len(result['experiment_plans'])} experiment plan(s) generated:")
    for plan in result["experiment_plans"]:
        feasibility = (
            "feasible" if plan["feasible"] else "NOT feasible as stated (simplified plan provided)"
        )
        print(
            f"- [{plan['hypothesis_id']}] {plan['objective']} ({feasibility}, complexity={plan['estimated_complexity']})"
        )

    if result["shared_infrastructure"]:
        print("\nShared infrastructure:")
        for item in result["shared_infrastructure"]:
            print(f"- {item}")

    print("\nPriority order:")
    for entry in sorted(result["priority_order"], key=lambda e: e["rank"]):
        print(f"  {entry['rank']}. [{entry['hypothesis_id']}] {entry['justification']}")


def run_coder_agent_cli(args: argparse.Namespace) -> None:
    planner_output = json.loads(Path(args.from_file).read_text())
    result = run_coder_agent(
        planner_output,
        output_dir=args.output_dir,
        interactive_slurm_review=args.interactive_slurm_review,
    )

    print(f"\n{len(result['experiments'])} experiment(s) processed:")
    for experiment in result["experiments"]:
        line = f"- [{experiment['hypothesis_id']}] {experiment['status']}"
        if experiment["reason"]:
            line += f" ({experiment['reason']})"
        print(line)
        if experiment["results"]:
            print(
                f"    metrics: {experiment['results']['metrics']}, meets_success_criteria: {experiment['results']['meets_success_criteria']}"
            )

    print(f"\nShared infrastructure: {result['shared_infrastructure_path']}")


def run_writer_agent_cli(args: argparse.Namespace) -> None:
    literature_output = json.loads(Path(args.literature_file).read_text())
    hypothesis_output = json.loads(Path(args.hypothesis_file).read_text())
    planner_output = json.loads(Path(args.planner_file).read_text())
    coder_output = json.loads(Path(args.coder_file).read_text())

    result = run_writer_agent(
        literature_output,
        hypothesis_output,
        planner_output,
        coder_output,
        output_dir=args.output_dir,
    )

    print(f"\nPaper written to {result['paper_path']}")
    print(f"Sections: {', '.join(result['sections_generated'])}")
    print(f"Supported: {result['hypotheses_supported']}")
    print(f"Refuted: {result['hypotheses_refuted']}")
    print(f"Inconclusive: {result['hypotheses_inconclusive']}")
    print(f"Citations used: {result['citations_used']}")
    if result["notes_for_review"]:
        print("\nNotes for review:")
        for note in result["notes_for_review"]:
            print(f"- {note}")


def run_reviewer_agent_cli(args: argparse.Namespace) -> None:
    literature_output = json.loads(Path(args.literature_file).read_text())
    hypothesis_output = json.loads(Path(args.hypothesis_file).read_text())
    planner_output = json.loads(Path(args.planner_file).read_text())
    coder_output = json.loads(Path(args.coder_file).read_text())
    paper_summary = json.loads(Path(args.paper_summary_file).read_text())

    result = run_reviewer_agent(
        paper_summary["paper_path"],
        paper_summary,
        literature_output,
        hypothesis_output,
        planner_output,
        coder_output,
        quality_threshold=args.quality_threshold,
        output_dir=args.output_dir,
    )

    print(f"\noverall_pass: {result['overall_pass']}")
    print(f"quality_scores: {result['quality_scores']}")
    print(f"hallucinations: {len(result['hallucinations'])}")
    print(f"citation_issues: {len(result['citation_issues'])}")
    print(f"results_accuracy_issues: {len(result['results_accuracy_issues'])}")
    print(f"hypothesis_coverage_issues: {len(result['hypothesis_coverage_issues'])}")
    print(f"\nfeedback_for_writer:\n{result['feedback_for_writer']}")


def run_writer_reviewer_loop_cli(args: argparse.Namespace) -> None:
    literature_output = json.loads(Path(args.literature_file).read_text())
    hypothesis_output = json.loads(Path(args.hypothesis_file).read_text())
    planner_output = json.loads(Path(args.planner_file).read_text())
    coder_output = json.loads(Path(args.coder_file).read_text())

    result = run_writer_reviewer_loop(
        literature_output,
        hypothesis_output,
        planner_output,
        coder_output,
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        output_dir=args.output_dir,
    )

    print(f"\nFinal paper: {result['final_paper_path']}")
    print(f"Iterations run: {result['iterations_run']}")
    print(f"Converged: {result['converged']}")
    print(f"Review history: {result['review_history_path']}")
    if result["unresolved_issues"]:
        print(f"\n{len(result['unresolved_issues'])} unresolved issue(s):")
        for issue in result["unresolved_issues"]:
            print(f"- {issue}")


def _resume_inputs(resume_from_file: str) -> dict:
    """The graph inputs that pick a previous partial run back up.

    Reads the JSON a partial `orchestrate` run wrote (`stages_completed` plus
    each stage's raw output) and turns it into `resume_from` — the stage that
    run stopped after — plus the stage outputs to seed the new run's state with.
    The entry point does the rest; see orchestrator/graph.py's _entry_router.
    """
    previous = json.loads(Path(resume_from_file).read_text())
    completed = previous.get("stages_completed") or []
    if not completed:
        raise SystemExit(
            f"{resume_from_file} has no completed stages to resume from. Point --resume-from-file at an "
            "orchestrate_partial_<timestamp>.json written by a run that stopped at an earlier --end-stage."
        )

    unknown = [key for key in completed if key not in STAGE_FOR_OUTPUT_KEY]
    missing = [key for key in completed if key not in previous]
    if unknown or missing:
        raise SystemExit(
            f"{resume_from_file} lists stage(s) this pipeline cannot resume from: "
            f"{', '.join(unknown + missing)}."
        )

    inputs = {key: previous[key] for key in completed}
    inputs["resume_from"] = STAGE_FOR_OUTPUT_KEY[completed[-1]]
    return inputs


def _write_partial_result(result: dict, output_dir: str | None) -> Path:
    """Persist a partial run's bundle so `--resume-from-file` has something to
    point at. Every stage already wrote its own output file; what only exists in
    memory is *which* stages ran and their outputs gathered into one dict.

    Timestamped rather than overwritten, matching how every agent writes its own
    output — two partial runs of the same question must not clobber each other.
    """
    directory = Path(output_dir or "outputs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"orchestrate_partial_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def run_orchestrate_cli(args: argparse.Namespace) -> None:
    graph = build_pipeline_graph()
    run_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    inputs = {
        "research_question": args.question,
        "max_results_per_query": args.max_results,
        "download_dir": args.download_dir,
        "metadata_path": f"{args.download_dir}/metadata.json",
        "output_dir": args.output_dir,
        "max_iterations": args.max_iterations,
        "quality_threshold": args.quality_threshold,
        "end_stage": args.end_stage,
        "include_interdisciplinary": not args.no_interdisciplinary,
    }
    if args.papers_file:
        # The question is still required and still used: every later stage's
        # prompts frame their work around it, papers or no papers.
        inputs["start_stage"] = "own_papers"
        inputs["seed_papers"] = json.loads(Path(args.papers_file).read_text())

    if args.resume_from_file:
        inputs.update(_resume_inputs(args.resume_from_file))

    state = graph.invoke(inputs, config=run_config)
    result = state["final_result"]

    # A run that stopped before the Writer/Reviewer stage has no paper to report
    # — just which stages ran, each having written its own output file already.
    if "stages_completed" in result:
        print(f"\nStopped after: {args.end_stage}")
        print(f"Stages completed: {', '.join(result['stages_completed'])}")
        print(f"Partial run written to: {_write_partial_result(result, args.output_dir)}")
        return

    print(f"\nFinal paper: {result['final_paper_path']}")
    print(f"Iterations run: {result['iterations_run']}")
    print(f"Converged: {result['converged']}")
    print(f"Review history: {result['review_history_path']}")
    if result["unresolved_issues"]:
        print(f"\n{len(result['unresolved_issues'])} unresolved issue(s):")
        for issue in result["unresolved_issues"]:
            print(f"- {issue}")


def run_orchestrate_batch_cli(args: argparse.Namespace) -> None:
    if (args.slice_index is None) != (args.slice_count is None):
        raise SystemExit("--slice-index and --slice-count must be given together")

    shared = {
        "output_root": args.output_dir,
        "max_results_per_query": args.max_results,
        "download_dir_root": args.download_dir,
        "max_iterations": args.max_iterations,
        "quality_threshold": args.quality_threshold,
        "max_consecutive_failures": args.max_consecutive_failures,
        "resume": not args.no_resume,
    }

    if args.slice_index is None:
        summary = batch.run_batch(args.questions_file, **shared)
    else:
        summary = batch.run_batch_slice(
            args.questions_file, args.slice_index, args.slice_count, **shared
        )

    print(f"\nManifest: {summary['manifest_path']}")
    print(f"Completed: {summary['completed']}/{summary['total']}")
    print(f"Failed: {summary['failed']}")
    if summary["skipped_already_done"]:
        print(f"Skipped (already done): {summary['skipped_already_done']}")


def run_serve_cli(args: argparse.Namespace) -> None:
    """Starts the web UI. Imported lazily so the rest of the CLI keeps working
    without `uv sync --extra webapp` — a headless SLURM sweep has no use for an
    HTTP server and shouldn't have to install one."""
    try:
        import uvicorn

        from research_pipeline.webapp.app import create_app
    except ImportError as exc:
        raise SystemExit(
            f"The web UI needs its optional dependencies ({exc.name}). Install them with:\n"
            "    uv sync --extra webapp"
        ) from exc

    host = args.host or settings.webapp_host
    port = args.port or settings.webapp_port
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Binding %s, which is reachable from other machines. This app has no authentication and can "
            "start jobs and read files — on a shared cluster, bind 127.0.0.1 and use an SSH tunnel instead.",
            host,
        )

    print(f"Research pipeline UI on http://{host}:{port} (runs in {settings.webapp_runs_dir}/)")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-pipeline", description="Multi-agent research pipeline"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    subparsers = parser.add_subparsers(dest="agent", required=True)

    literature = subparsers.add_parser(
        "literature", help="search + download papers for a research question"
    )
    literature.add_argument("question", help="research question to search for")
    literature.add_argument(
        "--max-results", type=int, default=5, help="max results per generated query, per source"
    )
    literature.add_argument(
        "--download-dir", default="papers", help="directory to save downloaded PDFs + metadata"
    )
    literature.set_defaults(func=run_literature_agent)

    interdisciplinary = subparsers.add_parser(
        "interdisciplinary-literature",
        help="search adjacent fields for cross-disciplinary papers + bridge insights",
    )
    interdisciplinary.add_argument(
        "--from-file",
        required=True,
        help="Literature Agent metadata.json, or a bare JSON list of papers",
    )
    interdisciplinary.add_argument(
        "--output-dir",
        default=None,
        help="directory to write interdisciplinary_<timestamp>.json (default: outputs/)",
    )
    interdisciplinary.set_defaults(func=run_interdisciplinary_literature_agent_cli)

    hypothesis = subparsers.add_parser(
        "hypothesis", help="synthesize a paper set into gaps + ranked testable hypotheses"
    )
    hypothesis.add_argument(
        "--from-file",
        required=True,
        help="Literature Agent metadata.json, Interdisciplinary Literature Agent output, or a bare JSON list of papers",
    )
    hypothesis.add_argument(
        "--output-dir",
        default=None,
        help="directory to write hypotheses_<timestamp>.json (default: outputs/)",
    )
    hypothesis.set_defaults(func=run_hypothesis_agent_cli)

    experiment_planner = subparsers.add_parser(
        "experiment-planner",
        help="turn a Hypothesis Agent output into implementation-ready experiment plans",
    )
    experiment_planner.add_argument(
        "--from-file",
        required=True,
        help="Hypothesis Agent output JSON (outputs/hypotheses_<timestamp>.json)",
    )
    experiment_planner.add_argument(
        "--output-dir",
        default=None,
        help="directory to write experiment_plan_<timestamp>.json (default: outputs/)",
    )
    experiment_planner.set_defaults(func=run_experiment_planner_agent_cli)

    coder = subparsers.add_parser(
        "coder",
        help="generate + execute runnable experiment code from an Experiment Planner output",
    )
    coder.add_argument(
        "--from-file",
        required=True,
        help="Experiment Planner output JSON (outputs/experiment_plan_<timestamp>.json)",
    )
    coder.add_argument(
        "--output-dir",
        default=None,
        help="directory to write coder_agent_summary_<timestamp>.json (default: outputs/)",
    )
    coder.add_argument(
        "--interactive-slurm-review",
        action="store_true",
        default=None,
        help=(
            "pause and ask before leaving a plan that can't run here (no GPU, or high "
            "complexity) for manual review — approving still goes through the same "
            "self-review and job-cap gates CODER_AUTO_SUBMIT_SLURM does. Only for this "
            "direct, attended invocation; never set for orchestrate/orchestrate-batch. "
            "Overrides CODER_INTERACTIVE_SLURM_REVIEW when passed."
        ),
    )
    coder.set_defaults(func=run_coder_agent_cli)

    writer = subparsers.add_parser(
        "writer", help="draft a full academic paper (PDF) from the whole pipeline's combined output"
    )
    writer.add_argument("--literature-file", required=True, help="Literature Agent metadata.json")
    writer.add_argument(
        "--hypothesis-file",
        required=True,
        help="Hypothesis Agent output JSON (outputs/hypotheses_<timestamp>.json)",
    )
    writer.add_argument(
        "--planner-file",
        required=True,
        help="Experiment Planner output JSON (outputs/experiment_plan_<timestamp>.json)",
    )
    writer.add_argument(
        "--coder-file",
        required=True,
        help="Coder Agent summary JSON (outputs/coder_agent_summary_<timestamp>.json)",
    )
    writer.add_argument(
        "--output-dir",
        default=None,
        help="directory to write paper_<timestamp>.pdf + paper_summary_<timestamp>.json (default: outputs/)",
    )
    writer.set_defaults(func=run_writer_agent_cli)

    reviewer = subparsers.add_parser(
        "reviewer", help="review a Writer Agent paper against the upstream ground truth"
    )
    reviewer.add_argument("--literature-file", required=True, help="Literature Agent metadata.json")
    reviewer.add_argument("--hypothesis-file", required=True, help="Hypothesis Agent output JSON")
    reviewer.add_argument("--planner-file", required=True, help="Experiment Planner output JSON")
    reviewer.add_argument("--coder-file", required=True, help="Coder Agent summary JSON")
    reviewer.add_argument(
        "--paper-summary-file",
        required=True,
        help="Writer Agent summary JSON (outputs/paper_summary_<timestamp>.json)",
    )
    reviewer.add_argument(
        "--quality-threshold",
        type=int,
        default=None,
        help="minimum 1-5 quality score to pass (default: 4)",
    )
    reviewer.add_argument(
        "--output-dir",
        default=None,
        help="directory to write review_<timestamp>.json (default: outputs/)",
    )
    reviewer.set_defaults(func=run_reviewer_agent_cli)

    loop = subparsers.add_parser(
        "writer-reviewer-loop",
        help="run the Writer and Reviewer Agents in a feedback loop until the paper passes review or max iterations is hit",
    )
    loop.add_argument("--literature-file", required=True, help="Literature Agent metadata.json")
    loop.add_argument("--hypothesis-file", required=True, help="Hypothesis Agent output JSON")
    loop.add_argument("--planner-file", required=True, help="Experiment Planner output JSON")
    loop.add_argument("--coder-file", required=True, help="Coder Agent summary JSON")
    loop.add_argument(
        "--max-iterations", type=int, default=None, help="max Writer/Reviewer cycles (default: 3)"
    )
    loop.add_argument(
        "--quality-threshold",
        type=int,
        default=None,
        help="minimum 1-5 quality score to pass (default: 4)",
    )
    loop.add_argument(
        "--output-dir",
        default=None,
        help="directory for v1.pdf, v2.pdf, ..., and review_log.json (default: outputs/paper)",
    )
    loop.set_defaults(func=run_writer_reviewer_loop_cli)

    orchestrate = subparsers.add_parser(
        "orchestrate",
        help=(
            "run the whole pipeline end to end: literature -> interdisciplinary-literature -> hypothesis "
            "-> experiment-planner -> coder -> writer/reviewer loop"
        ),
    )
    orchestrate.add_argument("question", help="research question to search for")
    orchestrate.add_argument(
        "--max-results", type=int, default=5, help="max results per generated query, per source"
    )
    orchestrate.add_argument(
        "--download-dir", default="papers", help="directory to save downloaded PDFs + metadata"
    )
    orchestrate.add_argument(
        "--output-dir",
        default=None,
        help="directory for every agent's output, plus v1.pdf, v2.pdf, ..., and review_log.json (default: each agent's own default)",
    )
    orchestrate.add_argument(
        "--max-iterations", type=int, default=None, help="max Writer/Reviewer cycles (default: 3)"
    )
    orchestrate.add_argument(
        "--quality-threshold",
        type=int,
        default=None,
        help="minimum 1-5 quality score to pass (default: 4)",
    )
    orchestrate.add_argument(
        "--end-stage",
        choices=[
            "literature",
            "interdisciplinary_literature",
            "hypothesis",
            "experiment_planner",
            "coder",
            "writer_reviewer",
        ],
        default="writer_reviewer",
        help="stop the run after this stage (default: writer_reviewer, i.e. the whole pipeline)",
    )
    orchestrate.add_argument(
        "--no-interdisciplinary",
        action="store_true",
        help="skip the cross-field Interdisciplinary Literature stage",
    )
    orchestrate.add_argument(
        "--papers-file",
        default=None,
        help=(
            "JSON list of paper dicts to start from instead of running the literature search; "
            "the question argument is still used as the research question by every later stage"
        ),
    )
    orchestrate.add_argument(
        "--resume-from-file",
        default=None,
        help=(
            "JSON written by an earlier partial run (orchestrate_partial_<timestamp>.json): its finished "
            "stages are carried forward and this run picks up after the last of them, so use --end-stage "
            "to say how much further to go"
        ),
    )
    orchestrate.set_defaults(func=run_orchestrate_cli)

    orchestrate_batch = subparsers.add_parser(
        "orchestrate-batch",
        help="run the whole pipeline unattended over many research questions from a file",
    )
    orchestrate_batch.add_argument(
        "--questions-file",
        required=True,
        help="one research question per line (# comments allowed)",
    )
    orchestrate_batch.add_argument(
        "--output-dir",
        default=None,
        help="root directory; each question gets its own subdirectory (default: outputs/batch)",
    )
    orchestrate_batch.add_argument(
        "--download-dir",
        default=None,
        help="root directory for downloaded papers (default: <output-dir>/papers)",
    )
    orchestrate_batch.add_argument(
        "--max-results", type=int, default=5, help="max results per generated query, per source"
    )
    orchestrate_batch.add_argument(
        "--max-iterations", type=int, default=None, help="max Writer/Reviewer cycles (default: 3)"
    )
    orchestrate_batch.add_argument(
        "--quality-threshold",
        type=int,
        default=None,
        help="minimum 1-5 quality score to pass (default: 4)",
    )
    orchestrate_batch.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=None,
        help="halt the batch after this many failures in a row (default: 5)",
    )
    orchestrate_batch.add_argument(
        "--no-resume",
        action="store_true",
        help="re-run questions already marked completed in the manifest",
    )
    orchestrate_batch.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="for SLURM job arrays: which slice this task runs",
    )
    orchestrate_batch.add_argument(
        "--slice-count",
        type=int,
        default=None,
        help="for SLURM job arrays: how many slices in total",
    )
    orchestrate_batch.set_defaults(func=run_orchestrate_batch_cli)

    serve = subparsers.add_parser(
        "serve", help="web UI for starting a run and watching it stage by stage"
    )
    serve.add_argument(
        "--host",
        default=None,
        help="address to bind (default: 127.0.0.1; see README before changing)",
    )
    serve.add_argument("--port", type=int, default=None, help="port to bind (default: 8000)")
    serve.set_defaults(func=run_serve_cli)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
