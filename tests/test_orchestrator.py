import json
import uuid
from types import SimpleNamespace

import pytest

from research_pipeline.agents.coder import CoderAgentError
from research_pipeline.agents.reviewer.reviewer_agent import ReviewerAgent
from research_pipeline.agents.writer.writer_agent import WriterAgent
from research_pipeline.orchestrator import nodes
from research_pipeline.orchestrator.graph import (
    STAGE_FOR_OUTPUT_KEY,
    build_pipeline_graph,
    should_continue_revising,
)

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
    """Replaces every upstream agent with a stub returning canned output, and
    returns a `calls` dict recording each stage's invocation arguments — so a
    test can assert both *that* a stage ran and *what* it was handed."""
    calls: dict[str, list] = {stage: [] for stage in ("literature", "interdisciplinary", "hypothesis", "planner", "coder")}

    class FakeLiteratureGraph:
        def invoke(self, state, config):
            calls["literature"].append(state)
            return _literature_state()

    def _record(stage, output):
        def stub(*args, **kwargs):
            calls[stage].append((args, kwargs))
            return output

        return stub

    monkeypatch.setattr(nodes, "build_literature_graph", lambda: FakeLiteratureGraph())
    monkeypatch.setattr(nodes, "run_interdisciplinary_literature_agent", _record("interdisciplinary", _interdisciplinary_output()))
    monkeypatch.setattr(nodes, "run_hypothesis_agent", _record("hypothesis", _hypothesis_output()))
    monkeypatch.setattr(nodes, "run_experiment_planner_agent", _record("planner", _planner_output()))
    monkeypatch.setattr(nodes, "run_coder_agent", _record("coder", _coder_output()))
    return calls


def _invoke(tmp_path, reviewer_model, max_iterations=3, **extra_inputs):
    """extra_inputs carries the optional run-shape keys (start_stage/end_stage/
    include_interdisciplinary/seed_papers). Passing none of them must reproduce
    the original fixed linear run — which is what every pre-existing test here
    still asserts, unmodified."""
    graph = build_pipeline_graph()
    state = graph.invoke(
        {
            "research_question": "does RAG help?",
            "output_dir": str(tmp_path),
            "max_iterations": max_iterations,
            "quality_threshold": 4,
            **extra_inputs,
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
        def invoke(self, messages, **kwargs):
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

        def invoke(self, messages, **kwargs):
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
        def invoke(self, messages, **kwargs):
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


# -- run shape: start_stage / end_stage / include_interdisciplinary ------------------


class AlwaysPassReviewerModel:
    def invoke(self, messages, **kwargs):
        if "Score this research paper draft" in messages[1][1]:
            return SimpleNamespace(
                content=_quality_response({"clarity": 5, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5})
            )
        return SimpleNamespace(content=json.dumps({"hallucinations": [], "framing_issues": []}))


def _seed_papers() -> list[dict]:
    return [
        {"title": "Seeded RAG Paper", "authors": ["A. Smith"], "abstract": "abc", "year": 2020, "arxiv_id": "s1"},
        {"title": "Seeded Retrieval Paper", "authors": ["B. Jones"], "abstract": "def", "year": 2021, "arxiv_id": "s2"},
    ]


def test_hypothesis_node_falls_back_to_literature_output_without_interdisciplinary(monkeypatch):
    captured = {}

    def fake_run(papers, research_question=None, interdisciplinary_context=None, output_dir=None):
        captured.update(papers=papers, research_question=research_question, interdisciplinary_context=interdisciplinary_context)
        return _hypothesis_output()

    monkeypatch.setattr(nodes, "run_hypothesis_agent", fake_run)
    result = nodes.run_hypothesis_node({"literature_output": _literature_state()})

    assert captured["papers"] == _literature_state()["merged_papers"]
    assert captured["research_question"] == "does RAG help?"
    # no cross-field stage ran, so there are no bridge insights to pass on
    assert captured["interdisciplinary_context"] is None
    assert result == {"hypothesis_output": _hypothesis_output()}


def test_pipeline_skips_interdisciplinary_when_disabled(tmp_path, monkeypatch):
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(tmp_path, AlwaysPassReviewerModel(), include_interdisciplinary=False)

    assert calls["interdisciplinary"] == []
    assert "interdisciplinary_output" not in state
    # the Hypothesis Agent ran straight off the Literature Agent's papers
    assert calls["hypothesis"][0][0][0] == _literature_state()["merged_papers"]
    assert calls["hypothesis"][0][1]["interdisciplinary_context"] is None
    # ...and the run still completed end to end
    assert state["final_result"]["converged"] is True


def test_end_stage_interdisciplinary_stops_after_literature_when_it_is_disabled(tmp_path, monkeypatch):
    """A contradictory pair — stop after the cross-field stage, but don't run it
    — resolves to stopping after literature rather than stranding the run."""
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(
        tmp_path, object(), end_stage="interdisciplinary_literature", include_interdisciplinary=False
    )

    assert calls["interdisciplinary"] == []
    assert calls["hypothesis"] == []
    assert state["final_result"]["stages_completed"] == ["literature_output"]


@pytest.mark.parametrize(
    "end_stage,expected_stages",
    [
        ("literature", ["literature"]),
        ("interdisciplinary_literature", ["literature", "interdisciplinary"]),
        ("hypothesis", ["literature", "interdisciplinary", "hypothesis"]),
        ("experiment_planner", ["literature", "interdisciplinary", "hypothesis", "planner"]),
        ("coder", ["literature", "interdisciplinary", "hypothesis", "planner", "coder"]),
    ],
)
def test_pipeline_stops_after_requested_end_stage(tmp_path, monkeypatch, end_stage, expected_stages):
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(tmp_path, object(), end_stage=end_stage)

    ran = [stage for stage, invocations in calls.items() if invocations]
    assert ran == expected_stages
    # the Writer/Reviewer cycle never started
    assert "review_history" not in state
    assert not (tmp_path / "v1.pdf").exists()

    state_key_for = {
        "literature": "literature_output",
        "interdisciplinary": "interdisciplinary_output",
        "hypothesis": "hypothesis_output",
        "planner": "planner_output",
        "coder": "coder_output",
    }
    expected_keys = [state_key_for[stage] for stage in expected_stages]
    result = state["final_result"]
    assert result["stages_completed"] == expected_keys
    assert result["generated_at"]
    # each stage's raw output is carried on the result, usable directly
    for key in expected_keys:
        assert result[key] == state[key]


def test_pipeline_runs_to_the_end_when_end_stage_is_writer_reviewer(tmp_path, monkeypatch):
    """The explicit spelling of the default — same full run, same result shape."""
    calls = _stub_upstream_agents(monkeypatch)

    result = _invoke(tmp_path, AlwaysPassReviewerModel(), end_stage="writer_reviewer")["final_result"]

    assert all(invocations for invocations in calls.values())
    assert result["converged"] is True
    assert result["final_paper_path"] == str(tmp_path / "v1.pdf")
    assert "stages_completed" not in result


def test_pipeline_starts_from_user_supplied_papers(tmp_path, monkeypatch):
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(
        tmp_path,
        AlwaysPassReviewerModel(),
        start_stage="own_papers",
        seed_papers=_seed_papers(),
    )

    # the literature *search* never ran
    assert calls["literature"] == []
    assert state["literature_output"]["merged_papers"] == [
        {**paper, "source": "user_provided"} for paper in _seed_papers()
    ]
    assert state["literature_output"]["research_question"] == "does RAG help?"
    # ...and the seeded papers are what the downstream stages received
    assert calls["interdisciplinary"][0][0][0] == state["literature_output"]["merged_papers"]
    assert calls["hypothesis"][0][0][0] == _interdisciplinary_output()["papers"]
    assert state["final_result"]["converged"] is True


def test_pipeline_starts_from_user_supplied_papers_without_interdisciplinary(tmp_path, monkeypatch):
    """Both skips at once: seeded papers straight into the Hypothesis Agent."""
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(
        tmp_path,
        object(),
        start_stage="own_papers",
        seed_papers=_seed_papers(),
        include_interdisciplinary=False,
        end_stage="hypothesis",
    )

    assert calls["literature"] == []
    assert calls["interdisciplinary"] == []
    assert calls["hypothesis"][0][0][0] == state["literature_output"]["merged_papers"]
    assert state["final_result"]["stages_completed"] == ["literature_output", "hypothesis_output"]


# -- resuming a stopped run ---------------------------------------------------------


def test_stage_for_output_key_pairs_every_partial_key_with_the_stage_that_wrote_it():
    """The one place that pairing is written down — the webapp and the CLI both
    read a stopped run's last output key through it rather than re-deriving it."""
    assert STAGE_FOR_OUTPUT_KEY == {
        "literature_output": "literature",
        "interdisciplinary_output": "interdisciplinary_literature",
        "hypothesis_output": "hypothesis",
        "planner_output": "experiment_planner",
        "coder_output": "coder",
    }
    # writer_reviewer is the one stage with no *_output key of its own
    assert "writer_reviewer" not in STAGE_FOR_OUTPUT_KEY.values()


def test_resuming_enters_after_the_stage_the_previous_run_stopped_at(tmp_path, monkeypatch):
    """The whole feature in one test: the stages already done are seeded into
    state rather than re-run, and the graph starts at the one after them."""
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(
        tmp_path,
        object(),
        resume_from="hypothesis",
        end_stage="coder",
        literature_output=_literature_state(),
        interdisciplinary_output=_interdisciplinary_output(),
        hypothesis_output=_hypothesis_output(),
    )

    # nothing up to and including the resume point ran a second time
    assert [stage for stage, invocations in calls.items() if invocations] == ["planner", "coder"]
    # ...and the planner was handed the *seeded* hypothesis output
    assert calls["planner"][0][0][0] == _hypothesis_output()
    assert not (tmp_path / "v1.pdf").exists()

    result = state["final_result"]
    # the bundle stays cumulative, so this run can itself be resumed from
    assert result["stages_completed"] == [
        "literature_output", "interdisciplinary_output", "hypothesis_output", "planner_output", "coder_output",
    ]
    assert result["hypothesis_output"] == _hypothesis_output()
    assert result["coder_output"] == _coder_output()


def test_resuming_from_literature_runs_the_cross_field_stage_next(tmp_path, monkeypatch):
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(
        tmp_path,
        object(),
        resume_from="literature",
        end_stage="interdisciplinary_literature",
        literature_output=_literature_state(),
    )

    assert calls["literature"] == []
    assert calls["interdisciplinary"][0][0][0] == _literature_state()["merged_papers"]
    assert state["final_result"]["stages_completed"] == ["literature_output", "interdisciplinary_output"]


def test_resuming_honours_include_interdisciplinary(tmp_path, monkeypatch):
    """The resume entry point asks the same router every interior edge asks, so
    a disabled cross-field stage is skipped on the way *in* exactly as in flight."""
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(
        tmp_path,
        object(),
        resume_from="literature",
        end_stage="hypothesis",
        include_interdisciplinary=False,
        literature_output=_literature_state(),
    )

    assert calls["literature"] == [] and calls["interdisciplinary"] == []
    assert calls["hypothesis"][0][0][0] == _literature_state()["merged_papers"]
    assert state["final_result"]["stages_completed"] == ["literature_output", "hypothesis_output"]


def test_resuming_from_the_coder_stage_goes_straight_into_the_writer_reviewer_cycle(tmp_path, monkeypatch):
    """The far end of the range: every upstream stage seeded, so the run is
    nothing but the Writer/Reviewer loop over a previous run's work."""
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(
        tmp_path,
        AlwaysPassReviewerModel(),
        resume_from="coder",
        literature_output=_literature_state(),
        hypothesis_output=_hypothesis_output(),
        planner_output=_planner_output(),
        coder_output=_coder_output(),
    )

    assert all(invocations == [] for invocations in calls.values())
    assert state["final_result"]["converged"] is True
    assert state["final_result"]["final_paper_path"] == str(tmp_path / "v1.pdf")


def test_a_run_without_resume_from_still_starts_at_the_literature_stage(tmp_path, monkeypatch):
    """Regression: resume_from is the only thing that moves the entry point, so
    its absence has to leave the original behavior exactly as it was."""
    calls = _stub_upstream_agents(monkeypatch)

    state = _invoke(tmp_path, object(), end_stage="literature")

    assert len(calls["literature"]) == 1
    assert state["final_result"]["stages_completed"] == ["literature_output"]
