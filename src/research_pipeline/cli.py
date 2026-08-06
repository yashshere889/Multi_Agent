"""Command-line entry point.

Each agent gets its own subcommand, e.g.:
    research-pipeline literature "research question"
    research-pipeline hypothesis --from-file papers/metadata.json

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
from pathlib import Path

from research_pipeline.agents.coder import run_coder_agent
from research_pipeline.agents.experiment_planner import run_experiment_planner_agent
from research_pipeline.agents.hypothesis import run_hypothesis_agent
from research_pipeline.agents.literature.graph import build_literature_graph
from research_pipeline.agents.reviewer import run_reviewer_agent
from research_pipeline.agents.writer import run_writer_agent
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
        print(f"- [{paper['source']}] {paper['title']} ({paper.get('year')}) -> {paper.get('local_path')}")


def run_hypothesis_agent_cli(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.from_file).read_text())
    # accepts either a Literature Agent metadata.json ({"papers": [...], "research_question": ...})
    # or a bare JSON list of paper dicts
    if isinstance(data, dict):
        papers = data.get("papers", [])
        research_question = data.get("research_question")
    else:
        papers = data
        research_question = None

    result = run_hypothesis_agent(papers, research_question=research_question, output_dir=args.output_dir)

    print("\nLiterature summary:")
    print(result["literature_summary"])
    print(f"\n{len(result['hypotheses'])} hypotheses generated:")
    for hypothesis in result["hypotheses"]:
        print(f"- [{hypothesis['id']}] {hypothesis['statement']}")


def run_experiment_planner_agent_cli(args: argparse.Namespace) -> None:
    hypothesis_output = json.loads(Path(args.from_file).read_text())
    result = run_experiment_planner_agent(hypothesis_output, output_dir=args.output_dir)

    print(f"\n{len(result['experiment_plans'])} experiment plan(s) generated:")
    for plan in result["experiment_plans"]:
        feasibility = "feasible" if plan["feasible"] else "NOT feasible as stated (simplified plan provided)"
        print(f"- [{plan['hypothesis_id']}] {plan['objective']} ({feasibility}, complexity={plan['estimated_complexity']})")

    if result["shared_infrastructure"]:
        print("\nShared infrastructure:")
        for item in result["shared_infrastructure"]:
            print(f"- {item}")

    print("\nPriority order:")
    for entry in sorted(result["priority_order"], key=lambda e: e["rank"]):
        print(f"  {entry['rank']}. [{entry['hypothesis_id']}] {entry['justification']}")


def run_coder_agent_cli(args: argparse.Namespace) -> None:
    planner_output = json.loads(Path(args.from_file).read_text())
    result = run_coder_agent(planner_output, output_dir=args.output_dir)

    print(f"\n{len(result['experiments'])} experiment(s) processed:")
    for experiment in result["experiments"]:
        line = f"- [{experiment['hypothesis_id']}] {experiment['status']}"
        if experiment["reason"]:
            line += f" ({experiment['reason']})"
        print(line)
        if experiment["results"]:
            print(f"    metrics: {experiment['results']['metrics']}, meets_success_criteria: {experiment['results']['meets_success_criteria']}")

    print(f"\nShared infrastructure: {result['shared_infrastructure_path']}")


def run_writer_agent_cli(args: argparse.Namespace) -> None:
    literature_output = json.loads(Path(args.literature_file).read_text())
    hypothesis_output = json.loads(Path(args.hypothesis_file).read_text())
    planner_output = json.loads(Path(args.planner_file).read_text())
    coder_output = json.loads(Path(args.coder_file).read_text())

    result = run_writer_agent(literature_output, hypothesis_output, planner_output, coder_output, output_dir=args.output_dir)

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
        paper_summary["paper_path"], paper_summary, literature_output, hypothesis_output, planner_output, coder_output,
        quality_threshold=args.quality_threshold, output_dir=args.output_dir,
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
        literature_output, hypothesis_output, planner_output, coder_output,
        max_iterations=args.max_iterations, quality_threshold=args.quality_threshold, output_dir=args.output_dir,
    )

    print(f"\nFinal paper: {result['final_paper_path']}")
    print(f"Iterations run: {result['iterations_run']}")
    print(f"Converged: {result['converged']}")
    print(f"Review history: {result['review_history_path']}")
    if result["unresolved_issues"]:
        print(f"\n{len(result['unresolved_issues'])} unresolved issue(s):")
        for issue in result["unresolved_issues"]:
            print(f"- {issue}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-pipeline", description="Multi-agent research pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    subparsers = parser.add_subparsers(dest="agent", required=True)

    literature = subparsers.add_parser("literature", help="search + download papers for a research question")
    literature.add_argument("question", help="research question to search for")
    literature.add_argument("--max-results", type=int, default=5, help="max results per generated query, per source")
    literature.add_argument("--download-dir", default="papers", help="directory to save downloaded PDFs + metadata")
    literature.set_defaults(func=run_literature_agent)

    hypothesis = subparsers.add_parser("hypothesis", help="synthesize a paper set into gaps + testable hypotheses")
    hypothesis.add_argument("--from-file", required=True, help="Literature Agent metadata.json, or a bare JSON list of papers")
    hypothesis.add_argument("--output-dir", default=None, help="directory to write hypotheses_<timestamp>.json (default: outputs/)")
    hypothesis.set_defaults(func=run_hypothesis_agent_cli)

    experiment_planner = subparsers.add_parser(
        "experiment-planner", help="turn a Hypothesis Agent output into implementation-ready experiment plans"
    )
    experiment_planner.add_argument("--from-file", required=True, help="Hypothesis Agent output JSON (outputs/hypotheses_<timestamp>.json)")
    experiment_planner.add_argument("--output-dir", default=None, help="directory to write experiment_plan_<timestamp>.json (default: outputs/)")
    experiment_planner.set_defaults(func=run_experiment_planner_agent_cli)

    coder = subparsers.add_parser("coder", help="generate + execute runnable experiment code from an Experiment Planner output")
    coder.add_argument("--from-file", required=True, help="Experiment Planner output JSON (outputs/experiment_plan_<timestamp>.json)")
    coder.add_argument("--output-dir", default=None, help="directory to write coder_agent_summary_<timestamp>.json (default: outputs/)")
    coder.set_defaults(func=run_coder_agent_cli)

    writer = subparsers.add_parser("writer", help="draft a full academic paper (PDF) from the whole pipeline's combined output")
    writer.add_argument("--literature-file", required=True, help="Literature Agent metadata.json")
    writer.add_argument("--hypothesis-file", required=True, help="Hypothesis Agent output JSON (outputs/hypotheses_<timestamp>.json)")
    writer.add_argument("--planner-file", required=True, help="Experiment Planner output JSON (outputs/experiment_plan_<timestamp>.json)")
    writer.add_argument("--coder-file", required=True, help="Coder Agent summary JSON (outputs/coder_agent_summary_<timestamp>.json)")
    writer.add_argument("--output-dir", default=None, help="directory to write paper_<timestamp>.pdf + paper_summary_<timestamp>.json (default: outputs/)")
    writer.set_defaults(func=run_writer_agent_cli)

    reviewer = subparsers.add_parser("reviewer", help="review a Writer Agent paper against the upstream ground truth")
    reviewer.add_argument("--literature-file", required=True, help="Literature Agent metadata.json")
    reviewer.add_argument("--hypothesis-file", required=True, help="Hypothesis Agent output JSON")
    reviewer.add_argument("--planner-file", required=True, help="Experiment Planner output JSON")
    reviewer.add_argument("--coder-file", required=True, help="Coder Agent summary JSON")
    reviewer.add_argument("--paper-summary-file", required=True, help="Writer Agent summary JSON (outputs/paper_summary_<timestamp>.json)")
    reviewer.add_argument("--quality-threshold", type=int, default=None, help="minimum 1-5 quality score to pass (default: 4)")
    reviewer.add_argument("--output-dir", default=None, help="directory to write review_<timestamp>.json (default: outputs/)")
    reviewer.set_defaults(func=run_reviewer_agent_cli)

    loop = subparsers.add_parser(
        "writer-reviewer-loop", help="run the Writer and Reviewer Agents in a feedback loop until the paper passes review or max iterations is hit"
    )
    loop.add_argument("--literature-file", required=True, help="Literature Agent metadata.json")
    loop.add_argument("--hypothesis-file", required=True, help="Hypothesis Agent output JSON")
    loop.add_argument("--planner-file", required=True, help="Experiment Planner output JSON")
    loop.add_argument("--coder-file", required=True, help="Coder Agent summary JSON")
    loop.add_argument("--max-iterations", type=int, default=None, help="max Writer/Reviewer cycles (default: 3)")
    loop.add_argument("--quality-threshold", type=int, default=None, help="minimum 1-5 quality score to pass (default: 4)")
    loop.add_argument("--output-dir", default=None, help="directory for v1.pdf, v2.pdf, ..., and review_log.json (default: outputs/paper)")
    loop.set_defaults(func=run_writer_reviewer_loop_cli)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
