import json
import uuid
from types import SimpleNamespace

import pytest

from research_pipeline.agents.coder import CoderAgentError
from research_pipeline.agents.reviewer.reviewer_agent import ReviewerAgent
from research_pipeline.agents.writer.writer_agent import WriterAgent
from research_pipeline.orchestrator import nodes
from research_pipeline.orchestrator.graph import build_pipeline_graph, should_continue_revising

from test_writer_reviewer_loop import (
    FakeWriterModel,
    _coder_output,
    _hypothesis_output,
    _planner_output,
)


def _literature_state() -> dict:
    """The raw LiteratureState the literature subgraph returns — 'merged_papers',
    not the metadata.json 'papers' key the standalone CLI reads off disk."""
    return {
        "research_question": "does RAG help?",
        "merged_papers": [
            {"title": "RAG Paper", "authors": ["A. Smith"], "abstract": "abc", "year": 2020, "arxiv_id": "1", "source": "arxiv"}
        ],
    }


def _interdisciplinary_output() -> dict:
    """The Interdisciplinary Literature Agent's output: the same paper list under
    'papers', merged with whatever the cross-field search added."""
    cross_field = {
        "title": "Ecology Paper", "authors": ["E. Cologist"], "abstract": "curves", "year": 2019,
        "arxiv_id": "9", "source": "arxiv", "discipline": "ecology",
    }
    return {
        "papers": _literature_state()["merged_papers"] + [cross_field],
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
        "research_question": "does RAG help?",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


# -- individual nodes ---------------------------------------------------------------


def test_literature_node_passes_inputs_through_to_the_subgraph(monkeypatch):
    captured = {}

    class FakeGraph:
        def invoke(self, state, config):
            captured.update(state=state, config=config)
            return _literature_state()

    monkeypatch.setattr(nodes, "build_literature_graph", lambda: FakeGraph())
    result = nodes.run_literature_node({"research_question": "does RAG help?", "max_results_per_query": 3, "download_dir": "papers"})

    assert captured["state"]["research_question"] == "does RAG help?"
    assert captured["state"]["max_results_per_query"] == 3
    assert captured["state"]["metadata_path"] == "papers/metadata.json"
    assert captured["config"]["configurable"]["thread_id"]
    assert result == {"literature_output": _literature_state()}


def test_interdisciplinary_node_forwards_merged_papers_and_research_question(monkeypatch):
    captured = {}

    def fake_run(papers, research_question=None, output_dir=None):
        captured.update(papers=papers, research_question=research_question, output_dir=output_dir)
        return _interdisciplinary_output()

    monkeypatch.setattr(nodes, "run_interdisciplinary_literature_agent", fake_run)
    result = nodes.run_interdisciplinary_literature_node({"literature_output": _literature_state(), "output_dir": "out"})

    assert captured["papers"] == _literature_state()["merged_papers"]
    assert captured["research_question"] == "does RAG help?"
    assert captured["output_dir"] == "out"
    assert result == {"interdisciplinary_output": _interdisciplinary_output()}


def test_hypothesis_node_reads_the_interdisciplinary_output_including_bridge_insights(monkeypatch):
    captured = {}

    def fake_run(papers, research_question=None, interdisciplinary_context=None, output_dir=None):
        captured.update(
            papers=papers,
            research_question=research_question,
            interdisciplinary_context=interdisciplinary_context,
            output_dir=output_dir,
        )
        return _hypothesis_output()

    monkeypatch.setattr(nodes, "run_hypothesis_agent", fake_run)
    result = nodes.run_hypothesis_node({"interdisciplinary_output": _interdisciplinary_output(), "output_dir": "out"})

    assert captured["papers"] == _interdisciplinary_output()["papers"]
    assert captured["research_question"] == "does RAG help?"
    assert captured["interdisciplinary_context"] == _interdisciplinary_output()["bridge_insights"]
    assert captured["output_dir"] == "out"
    assert result == {"hypothesis_output": _hypothesis_output()}


def test_planner_node_plans_only_the_selected_hypothesis(monkeypatch):
    captured = {}

    def fake_run(hypothesis_output, output_dir=None, hypothesis_ids=None):
        captured.update(hypothesis_output=hypothesis_output, hypothesis_ids=hypothesis_ids)
        return _planner_output()

    monkeypatch.setattr(nodes, "run_experiment_planner_agent", fake_run)
    result = nodes.run_planner_node({"hypothesis_output": _hypothesis_output()})

    # the planner still receives the whole hypothesis output — only the plan set narrows
    assert captured["hypothesis_output"] == _hypothesis_output()
    assert captured["hypothesis_ids"] == [_hypothesis_output()["selected_hypothesis_id"]]
    assert result == {"planner_output": _planner_output()}


def test_writeup_papers_use_the_merged_cross_field_pool_when_one_exists():
    state = {"literature_output": _literature_state(), "interdisciplinary_output": _interdisciplinary_output()}
    assert nodes._papers_for_writeup(state) == _interdisciplinary_output()


def test_writeup_papers_fall_back_to_the_literature_output():
    assert nodes._papers_for_writeup({"literature_output": _literature_state()}) == _literature_state()


def test_planner_node_falls_back_to_all_hypotheses_without_a_selection(monkeypatch):
    """A hypothesis output produced before ranking existed, read back off disk."""
    captured = {}

    def fake_run(hypothesis_output, output_dir=None, hypothesis_ids=None):
        captured.update(hypothesis_ids=hypothesis_ids)
        return _planner_output()

    monkeypatch.setattr(nodes, "run_experiment_planner_agent", fake_run)
    legacy = {k: v for k, v in _hypothesis_output().items() if k != "selected_hypothesis_id"}
    nodes.run_planner_node({"hypothesis_output": legacy})

    assert captured["hypothesis_ids"] is None


def test_coder_node_forwards_planner_output(monkeypatch):
    captured = {}

    def fake_run(planner_output, output_dir=None):
        captured.update(planner_output=planner_output)
        return _coder_output()

    monkeypatch.setattr(nodes, "run_coder_agent", fake_run)
    result = nodes.run_coder_node({"planner_output": _planner_output()})

    assert captured["planner_output"] == _planner_output()
    assert result == {"coder_output": _coder_output()}


# -- the conditional edge -----------------------------------------------------------


def test_routing_finalizes_once_the_review_passes():
    assert should_continue_revising({"converged": True, "iteration": 1, "max_iterations": 3}) == "finalize"


def test_routing_finalizes_when_iterations_are_exhausted():
    assert should_continue_revising({"converged": False, "iteration": 3, "max_iterations": 3}) == "finalize"


def test_routing_revises_while_iterations_remain():
    assert should_continue_revising({"converged": False, "iteration": 1, "max_iterations": 3}) == "revise"


def test_routing_falls_back_to_the_configured_max_iterations():
    assert should_continue_revising({"converged": False, "iteration": 99, "max_iterations": None}) == "finalize"


# -- the whole graph ----------------------------------------------------------------


def _quality_response(scores) -> str:
    return json.dumps({"quality_scores": scores, "quality_notes": {"clarity": "too dense"}})


def _stub_upstream_agents(monkeypatch):
    class FakeLiteratureGraph:
        def invoke(self, state, config):
            return _literature_state()

    monkeypatch.setattr(nodes, "build_literature_graph", lambda: FakeLiteratureGraph())
    monkeypatch.setattr(nodes, "run_interdisciplinary_literature_agent", lambda *a, **k: _interdisciplinary_output())
    monkeypatch.setattr(nodes, "run_hypothesis_agent", lambda *a, **k: _hypothesis_output())
    monkeypatch.setattr(nodes, "run_experiment_planner_agent", lambda *a, **k: _planner_output())
    monkeypatch.setattr(nodes, "run_coder_agent", lambda *a, **k: _coder_output())


def _invoke(tmp_path, reviewer_model, max_iterations=3):
    graph = build_pipeline_graph()
    state = graph.invoke(
        {
            "research_question": "does RAG help?",
            "output_dir": str(tmp_path),
            "max_iterations": max_iterations,
            "quality_threshold": 4,
        },
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "writer": WriterAgent(chat_model=FakeWriterModel(), output_dir=tmp_path),
                "reviewer": ReviewerAgent(chat_model=reviewer_model, output_dir=tmp_path),
            }
        },
    )
    return state


def test_pipeline_runs_every_stage_and_converges_on_the_first_review(tmp_path, monkeypatch):
    _stub_upstream_agents(monkeypatch)

    class AlwaysPassReviewerModel:
        def invoke(self, messages):
            if "Score this research paper draft" in messages[1][1]:
                return SimpleNamespace(content=_quality_response({"clarity": 5, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5}))
            return SimpleNamespace(content=json.dumps({"hallucinations": [], "framing_issues": []}))

    state = _invoke(tmp_path, AlwaysPassReviewerModel())
    result = state["final_result"]

    assert state["interdisciplinary_output"] == _interdisciplinary_output()
    assert state["hypothesis_output"] == _hypothesis_output()
    assert state["planner_output"] == _planner_output()
    assert state["coder_output"] == _coder_output()
    assert result["converged"] is True
    assert result["iterations_run"] == 1
    assert result["unresolved_issues"] == []
    assert result["final_paper_path"] == str(tmp_path / "v1.pdf")
    assert (tmp_path / "v1.pdf").exists()
    assert len(json.loads((tmp_path / "review_log.json").read_text())) == 1


def test_pipeline_loops_back_through_the_writer_then_converges(tmp_path, monkeypatch):
    _stub_upstream_agents(monkeypatch)

    class FailOnceReviewerModel:
        def __init__(self):
            self.quality_calls = 0

        def invoke(self, messages):
            if "Score this research paper draft" in messages[1][1]:
                self.quality_calls += 1
                clarity = 2 if self.quality_calls == 1 else 5
                return SimpleNamespace(content=_quality_response({"clarity": clarity, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5}))
            return SimpleNamespace(content=json.dumps({"hallucinations": [], "framing_issues": []}))

    result = _invoke(tmp_path, FailOnceReviewerModel())["final_result"]

    assert result["converged"] is True
    assert result["iterations_run"] == 2
    assert (tmp_path / "v1.pdf").exists()
    assert (tmp_path / "v2.pdf").exists()
    log = json.loads((tmp_path / "review_log.json").read_text())
    assert [entry["review"]["overall_pass"] for entry in log] == [False, True]


def test_pipeline_stops_at_max_iterations_with_unresolved_issues(tmp_path, monkeypatch):
    _stub_upstream_agents(monkeypatch)

    class AlwaysFailReviewerModel:
        def invoke(self, messages):
            if "Score this research paper draft" in messages[1][1]:
                return SimpleNamespace(content=_quality_response({"clarity": 2, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5}))
            return SimpleNamespace(content=json.dumps({"hallucinations": [], "framing_issues": []}))

    result = _invoke(tmp_path, AlwaysFailReviewerModel(), max_iterations=2)["final_result"]

    assert result["converged"] is False
    assert result["iterations_run"] == 2
    assert any("clarity" in issue for issue in result["unresolved_issues"])
    assert result["final_paper_path"] == str(tmp_path / "v2.pdf")


def test_pipeline_lets_an_agent_failure_propagate(tmp_path, monkeypatch):
    _stub_upstream_agents(monkeypatch)

    def boom(*args, **kwargs):
        raise CoderAgentError("no runnable experiments")

    monkeypatch.setattr(nodes, "run_coder_agent", boom)

    with pytest.raises(CoderAgentError):
        _invoke(tmp_path, object())
