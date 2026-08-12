import json
import logging
from types import SimpleNamespace

import pytest

from research_pipeline.webapp import events, runner, runs, stages
from test_writer_reviewer_loop import _coder_output, _hypothesis_output, _planner_output


def _literature_delta() -> dict:
    return {
        "literature_output": {
            "research_question": "q",
            "search_queries": ["rag hallucination", "retrieval grounding"],
            "merged_papers": [
                {"title": "RAG Paper", "source": "arxiv", "local_path": "papers/rag.pdf"},
                {"title": "Grounding Paper", "source": "semantic_scholar", "local_path": None},
            ],
        }
    }


def _interdisciplinary_delta() -> dict:
    cross_field = {"title": "Ecology Paper", "source": "arxiv", "discipline": "ecology"}
    return {
        "interdisciplinary_output": {
            "papers": [{"title": "RAG Paper", "source": "arxiv"}, cross_field],
            "core_paper_ids": ["1"],
            "cross_field_papers": [cross_field],
            "fields_explored": [{"field": "ecology", "rationale": "same sparsity problem", "queries": ["rarefaction"]}],
            "bridge_insights": [
                {
                    "insight": "rarefaction curves quantify coverage",
                    "source_field": "ecology",
                    "connection_to_core_problem": "retrieval corpora are sampled sparsely too",
                    "supporting_paper_ids": ["9"],
                }
            ],
            "research_question": "q",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "model": "test-model",
        }
    }


def _review(overall_pass=True, iteration=1) -> dict:
    return {
        "review": {
            "iteration": iteration,
            "hallucinations": [] if overall_pass else [{"location": "Results", "claim": "c", "issue": "i"}],
            "citation_issues": [],
            "results_accuracy_issues": [],
            "hypothesis_coverage_issues": [],
            "quality_scores": {"clarity": 5, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5},
            "overall_pass": overall_pass,
            "feedback_for_writer": "" if overall_pass else "tighten the results section",
        },
        "converged": overall_pass,
        "review_history": [],
    }


def _final_result(paper_path: str) -> dict:
    return {
        "final_paper_path": paper_path,
        "iterations_run": 1,
        "converged": True,
        "unresolved_issues": [],
        "review_history_path": "review_log.json",
        "generated_at": "2026-01-01T00:00:00+00:00",
    }


def _happy_updates(paper_path: str) -> list[dict]:
    return [
        {"literature": _literature_delta()},
        {"interdisciplinary_literature": _interdisciplinary_delta()},
        {"hypothesis": {"hypothesis_output": _hypothesis_output()}},
        {"experiment_planner": {"planner_output": _planner_output()}},
        {"coder": {"coder_output": _coder_output()}},
        {
            "draft_or_revise": {
                "iteration": 1,
                "paper_summary": {
                    "paper_path": paper_path,
                    "sections_generated": ["Title", "Abstract", "Results", "References"],
                    "hypotheses_supported": ["H1", "H2"],
                    "hypotheses_refuted": ["H3"],
                    "hypotheses_inconclusive": [],
                    "citations_used": ["1"],
                },
            }
        },
        {"review": _review()},
        {"finalize": {"final_result": _final_result(paper_path)}},
    ]


class FakeGraph:
    """Stands in for the compiled orchestrator. Yields the same
    `{node_name: delta}` shape LangGraph's stream_mode="updates" produces."""

    def __init__(self, updates, raises=None, pending=()):
        self.updates = updates
        self.raises = raises
        # What get_state reports as still to do, i.e. whether this thread looks
        # like a run that stopped part-way. Empty is the ordinary case: a new
        # thread and a finished one both report no next node.
        self.pending = tuple(pending)
        self.seen_inputs = None
        self.seen_config = None

    def get_state(self, config):
        return SimpleNamespace(next=self.pending, values={}, created_at=None)

    def stream(self, inputs, config=None, stream_mode=None):
        self.seen_inputs = inputs
        self.seen_config = config
        assert stream_mode == "updates"
        for update in self.updates:
            yield update
        if self.raises:
            raise self.raises


@pytest.fixture
def store(tmp_path):
    return runs.RunStore(tmp_path / "runs")


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """runner.run() attaches a handler to the root logger and raises its level,
    which is right for a one-shot subprocess but would otherwise leak one
    test's event log into the next test's run."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers, root.level = handlers, level


def _stage_events(run_dir):
    return events.read_events(run_dir, types=[events.STAGE_COMPLETED])


def test_runner_emits_a_stage_event_per_node_in_order(store, monkeypatch):
    record = store.create("does retrieval reduce hallucination?", {"max_results_per_query": 2})
    run_dir = store.run_dir(record["run_id"])
    paper_path = str(run_dir / "outputs" / "v1.pdf")
    graph = FakeGraph(_happy_updates(paper_path))
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    final = runner.run(run_dir)

    assert [e["stage"] for e in _stage_events(run_dir)] == list(stages.STAGE_ORDER)
    assert final["status"] == runs.COMPLETED
    assert final["final_result"]["final_paper_path"] == paper_path
    assert json.loads((run_dir / "run.json").read_text())["status"] == runs.COMPLETED


def test_runner_summarizes_each_stage_from_its_own_delta(store, monkeypatch):
    record = store.create("q", {})
    run_dir = store.run_dir(record["run_id"])
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: FakeGraph(_happy_updates("v1.pdf")))

    runner.run(run_dir)
    by_stage = {e["stage"]: e["summary"] for e in _stage_events(run_dir)}

    assert by_stage["literature"]["papers_found"] == 2
    assert by_stage["literature"]["papers_downloaded"] == 1  # the one with a local_path
    assert [f["field"] for f in by_stage["interdisciplinary_literature"]["fields"]] == ["ecology"]
    assert by_stage["interdisciplinary_literature"]["cross_field_papers"] == 1
    assert [h["id"] for h in by_stage["hypothesis"]["hypotheses"]] == ["H1", "H2", "H3"]
    assert by_stage["hypothesis"]["selected"] == "H1"
    assert [p["complexity"] for p in by_stage["experiment_planner"]["plans"]] == ["low"] * 3
    assert by_stage["coder"]["experiments"][0]["meets_success_criteria"] is True
    assert by_stage["draft_or_revise"]["supported"] == 2
    assert by_stage["draft_or_revise"]["refuted"] == 1
    assert by_stage["review"]["overall_pass"] is True
    assert by_stage["review"]["total_issues"] == 0
    assert by_stage["finalize"]["iterations_run"] == 1


def test_runner_resumes_a_run_whose_thread_has_pending_nodes(store, monkeypatch):
    """Relaunching a run that died part-way continues it: the graph is invoked
    with None, which replays from the checkpoint rather than from the inputs."""
    record = store.create("q", {})
    run_dir = store.run_dir(record["run_id"])
    graph = FakeGraph(_happy_updates("v1.pdf"), pending=("coder",))
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    final = runner.run(run_dir)

    assert graph.seen_inputs is None
    assert graph.seen_config["configurable"]["thread_id"] == record["run_id"]
    assert final["status"] == runs.COMPLETED
    resumed = events.read_events(run_dir, types=[events.RUN_RESUMED])
    assert [e["pending"] for e in resumed] == [["coder"]]


def test_runner_starts_fresh_when_there_is_nothing_to_resume(store, monkeypatch):
    """The ordinary case, and the only one under the default in-memory
    checkpointer: no checkpoints for this thread, so the inputs are sent."""
    record = store.create("q", {})
    run_dir = store.run_dir(record["run_id"])
    graph = FakeGraph(_happy_updates("v1.pdf"))
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    runner.run(run_dir)

    assert graph.seen_inputs["research_question"] == "q"
    assert events.read_events(run_dir, types=[events.RUN_RESUMED]) == []


def test_runner_passes_run_scoped_paths_to_the_graph(store, monkeypatch):
    record = store.create("q", {"max_results_per_query": 7, "max_iterations": 2, "quality_threshold": 5})
    run_dir = store.run_dir(record["run_id"])
    graph = FakeGraph(_happy_updates("v1.pdf"))
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    runner.run(run_dir)

    assert graph.seen_inputs["download_dir"] == str(run_dir / "papers")
    assert graph.seen_inputs["output_dir"] == str(run_dir / "outputs")
    assert graph.seen_inputs["max_results_per_query"] == 7
    assert graph.seen_inputs["max_iterations"] == 2
    assert graph.seen_inputs["quality_threshold"] == 5
    assert graph.seen_config["configurable"]["thread_id"] == record["run_id"]


def test_runner_defaults_max_results_when_the_form_left_it_blank(store, monkeypatch):
    """The literature node reads it with state.get(key, default), which does not
    fall back for a key that is present but None."""
    record = store.create("q", {"max_results_per_query": None})
    run_dir = store.run_dir(record["run_id"])
    graph = FakeGraph(_happy_updates("v1.pdf"))
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    runner.run(run_dir)

    assert graph.seen_inputs["max_results_per_query"] is not None


def test_runner_passes_the_run_shape_params_through_to_the_graph(store, monkeypatch):
    seed_papers = [{"title": "A paper I already have", "source": "manual"}]
    record = store.create(
        "q",
        {
            "start_stage": "own_papers",
            "end_stage": "hypothesis",
            "include_interdisciplinary": False,
            "seed_papers": seed_papers,
        },
    )
    run_dir = store.run_dir(record["run_id"])
    updates = [
        {"seed_literature": _literature_delta()},
        {"hypothesis": {"hypothesis_output": _hypothesis_output()}},
        {"finalize": {"final_result": {"stages_completed": ["literature_output", "hypothesis_output"]}}},
    ]
    graph = FakeGraph(updates)
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    final = runner.run(run_dir)

    assert graph.seen_inputs["start_stage"] == "own_papers"
    assert graph.seen_inputs["end_stage"] == "hypothesis"
    assert graph.seen_inputs["include_interdisciplinary"] is False
    assert graph.seen_inputs["seed_papers"] == seed_papers
    assert final["status"] == runs.COMPLETED


def test_runner_omits_run_shape_params_the_caller_did_not_set(store, monkeypatch):
    """Absent must stay absent, not become an explicit None: the orchestrator
    reads these with state.get(key, default), which falls back only for a key
    that is missing — a present None would read as "explicitly off"."""
    record = store.create("q", {"max_results_per_query": 3})
    run_dir = store.run_dir(record["run_id"])
    graph = FakeGraph(_happy_updates("v1.pdf"))
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    runner.run(run_dir)

    for key in ("start_stage", "end_stage", "include_interdisciplinary", "seed_papers"):
        assert key not in graph.seen_inputs


def test_runner_omits_run_shape_params_that_are_present_but_null(store, monkeypatch):
    record = store.create("q", {"start_stage": None, "end_stage": None, "seed_papers": []})
    run_dir = store.run_dir(record["run_id"])
    graph = FakeGraph(_happy_updates("v1.pdf"))
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    runner.run(run_dir)

    for key in ("start_stage", "end_stage", "seed_papers"):
        assert key not in graph.seen_inputs


def test_runner_reports_the_seeded_literature_node_as_the_literature_stage(store, monkeypatch):
    """seed_literature is a different node but the same stage — it produces the
    same literature_output delta, and dropping it as an unknown node would leave
    a bring-your-own-papers run with no Literature row at all."""
    record = store.create("q", {"start_stage": "own_papers", "seed_papers": [{"title": "t"}]})
    run_dir = store.run_dir(record["run_id"])
    updates = [{"seed_literature": _literature_delta()}] + _happy_updates("v1.pdf")[1:]
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: FakeGraph(updates))

    runner.run(run_dir)

    stage_events = _stage_events(run_dir)
    assert [e["stage"] for e in stage_events] == list(stages.STAGE_ORDER)
    assert "seed_literature" not in [e["stage"] for e in stage_events]
    # And it is summarized as a literature delta, not dropped to an empty dict.
    assert stage_events[0]["summary"]["papers_found"] == 2


def test_runner_records_a_failure_instead_of_dying_silently(store, monkeypatch):
    record = store.create("q", {})
    run_dir = store.run_dir(record["run_id"])
    updates = [{"literature": _literature_delta()}]
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: FakeGraph(updates, raises=ValueError("boom")))

    final = runner.run(run_dir)

    assert final["status"] == runs.FAILED
    assert "ValueError: boom" in final["error"]
    failed = events.read_events(run_dir, types=[events.RUN_FAILED])
    assert len(failed) == 1
    assert "boom" in failed[0]["traceback"]
    # The stage that did finish is still reported.
    assert [e["stage"] for e in _stage_events(run_dir)] == ["literature"]


def test_runner_fails_a_graph_that_finishes_without_a_final_result(store, monkeypatch):
    record = store.create("q", {})
    run_dir = store.run_dir(record["run_id"])
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: FakeGraph([{"literature": _literature_delta()}]))

    final = runner.run(run_dir)

    assert final["status"] == runs.FAILED
    assert "final_result" in final["error"]


def test_runner_ignores_langgraph_bookkeeping_keys(store, monkeypatch):
    record = store.create("q", {})
    run_dir = store.run_dir(record["run_id"])
    updates = [{"__interrupt__": ()}] + _happy_updates("v1.pdf")
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: FakeGraph(updates))

    runner.run(run_dir)

    assert [e["stage"] for e in _stage_events(run_dir)] == list(stages.STAGE_ORDER)


def test_pipeline_logging_reaches_the_event_stream(store, monkeypatch):
    record = store.create("q", {})
    run_dir = store.run_dir(record["run_id"])

    def stream_and_log(*_args, **_kwargs):
        logging.getLogger("research_pipeline.agents.coder").info("generated code for H1")
        logging.getLogger("httpx").info("noisy third-party chatter")
        logging.getLogger("httpx").warning("connection reset")
        yield from _happy_updates("v1.pdf")

    graph = FakeGraph([])
    graph.stream = stream_and_log
    monkeypatch.setattr(runner, "build_pipeline_graph", lambda: graph)

    runner.run(run_dir)
    messages = [e["message"] for e in events.read_events(run_dir, types=[events.LOG])]

    assert "generated code for H1" in messages
    assert "connection reset" in messages  # WARNING from a third party still gets through
    assert "noisy third-party chatter" not in messages  # third-party INFO does not
