import json
from types import SimpleNamespace

from research_pipeline.agents.reviewer.reviewer_agent import ReviewerAgent
from research_pipeline.agents.writer.writer_agent import WriterAgent
from research_pipeline.writer_reviewer_loop import route_feedback_to_sections, run_writer_reviewer_loop


# -- route_feedback_to_sections: pure routing logic --------------------------------------


def _review(**overrides) -> dict:
    base = {
        "iteration": 1,
        "hallucinations": [],
        "citation_issues": [],
        "results_accuracy_issues": [],
        "hypothesis_coverage_issues": [],
        "quality_scores": {"clarity": 5, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5},
        "overall_pass": True,
        "feedback_for_writer": "",
    }
    return {**base, **overrides}


def test_route_feedback_routes_by_known_heading_prefix():
    review = _review(hallucinations=[{"location": "Introduction", "claim": "c", "issue": "i"}])
    routed = route_feedback_to_sections(review, quality_threshold=4)
    assert "Introduction" in routed
    assert "c" in routed["Introduction"][0]


def test_route_feedback_routes_hypothesis_scoped_location_by_top_level_section():
    review = _review(results_accuracy_issues=[{"location": "Results > H2", "claimed": "94%", "actual": "accuracy=0.87"}])
    routed = route_feedback_to_sections(review, quality_threshold=4)
    assert "Results" in routed
    assert "H2" not in routed  # not its own bucket — routed under the section, not the hypothesis


def test_route_feedback_sends_unrecognized_location_to_general():
    review = _review(citation_issues=[{"location": "References", "issue": "bad id"}])
    routed = route_feedback_to_sections(review, quality_threshold=4)
    assert "References" not in routed
    assert "__general__" in routed


def test_route_feedback_sends_low_quality_scores_to_general():
    review = _review(quality_scores={"clarity": 2, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5})
    routed = route_feedback_to_sections(review, quality_threshold=4)
    assert any("clarity" in bullet for bullet in routed["__general__"])


def test_route_feedback_empty_when_nothing_to_report():
    routed = route_feedback_to_sections(_review(), quality_threshold=4)
    assert routed == {}


# -- run_writer_reviewer_loop: orchestration, with fake chat models ---------------------


class FakeWriterModel:
    """A believable-enough draft: reports the actual accuracy figure in
    Results text (so ReviewerAgent's deterministic metric check has something
    real to match against) and cites paper 1 elsewhere."""

    def invoke(self, messages):
        prompt = messages[1][1]
        if "Propose a single academic paper title" in prompt:
            return SimpleNamespace(content="A Title")
        if "Results (JSON):" in prompt:
            return SimpleNamespace(content="The reported accuracy was 0.9, consistent with expectations.")
        return SimpleNamespace(content="Drafted prose citing [[cite:1]].")


def _hyp(hid) -> dict:
    return {
        "id": hid, "statement": f"statement {hid}", "rationale": "rationale", "related_gaps": [], "related_methods": [],
        "suggested_variables": {"independent": [], "dependent": []},
    }


def _hypothesis_output() -> dict:
    return {
        "literature_summary": "summary", "methods_overview": [], "gaps": [],
        "hypotheses": [_hyp("H1"), _hyp("H2"), _hyp("H3")],
        "ranking": [{"hypothesis_id": f"H{i}", "rank": i, "score": 9 - i, "justification": "j"} for i in (1, 2, 3)],
        "selected_hypothesis_id": "H1",
        "source_paper_ids": ["1"], "generated_at": "2026-01-01T00:00:00+00:00", "model": "test-model",
    }


def _plan(hid) -> dict:
    return {
        "hypothesis_id": hid, "feasible": True, "feasibility_notes": "ok", "objective": "o",
        "variables": {"independent": [], "dependent": []}, "design": "d",
        "data_requirements": {"source": "s", "description": "d", "preprocessing_steps": []},
        "methods": [], "evaluation": {"metrics": ["accuracy"], "baseline": "b", "success_criteria": "sc"},
        "implementation_steps": [{"step": 1, "description": "d"}], "estimated_complexity": "low", "risks": [],
    }


def _planner_output() -> dict:
    plans = [_plan("H1"), _plan("H2"), _plan("H3")]
    return {
        "experiment_plans": plans, "shared_infrastructure": [],
        "priority_order": [{"hypothesis_id": p["hypothesis_id"], "rank": i + 1, "justification": "j"} for i, p in enumerate(plans)],
        "source_hypothesis_ids": [p["hypothesis_id"] for p in plans],
        "generated_at": "2026-01-01T00:00:00+00:00", "model": "test-model",
    }


def _coder_output() -> dict:
    experiments = [
        {"hypothesis_id": hid, "status": "completed", "reason": "", "assumptions_made": [], "code_path": f"experiments/{hid}",
         "results": {"metrics": {"accuracy": 0.9}, "meets_success_criteria": True, "notes": "ok"}}
        for hid in ("H1", "H2", "H3")
    ]
    return {
        "experiments": experiments, "shared_infrastructure_path": "experiments/_shared",
        "source_hypothesis_ids": ["H1", "H2", "H3"], "generated_at": "2026-01-01T00:00:00+00:00", "model": "test-model",
    }


def _literature_output() -> dict:
    return {"papers": [{"title": "RAG Paper", "authors": ["A. Smith"], "abstract": "abc", "year": 2020, "arxiv_id": "1", "source": "arxiv"}]}


def _quality_response(scores) -> str:
    return json.dumps({"quality_scores": scores, "quality_notes": {"clarity": "too dense"}})


def test_loop_converges_on_first_iteration_when_review_passes(tmp_path):
    class AlwaysPassReviewerModel:
        def invoke(self, messages):
            prompt = messages[1][1]
            if "Score this research paper draft" in prompt:
                return SimpleNamespace(content=_quality_response({"clarity": 5, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5}))
            return SimpleNamespace(content=json.dumps({"hallucinations": [], "framing_issues": []}))

    writer = WriterAgent(chat_model=FakeWriterModel(), output_dir=tmp_path)
    reviewer = ReviewerAgent(chat_model=AlwaysPassReviewerModel(), output_dir=tmp_path)

    result = run_writer_reviewer_loop(
        _literature_output(), _hypothesis_output(), _planner_output(), _coder_output(),
        max_iterations=3, quality_threshold=4, output_dir=tmp_path, writer=writer, reviewer=reviewer,
    )

    assert result["converged"] is True
    assert result["iterations_run"] == 1
    assert result["unresolved_issues"] == []
    assert result["final_paper_path"] == str(tmp_path / "v1.pdf")
    assert (tmp_path / "v1.pdf").exists()
    assert (tmp_path / "review_log.json").exists()
    log = json.loads((tmp_path / "review_log.json").read_text())
    assert len(log) == 1


def test_loop_revises_and_converges_on_second_iteration(tmp_path):
    class FailOnceReviewerModel:
        def __init__(self):
            self.quality_calls = 0

        def invoke(self, messages):
            prompt = messages[1][1]
            if "Score this research paper draft" in prompt:
                self.quality_calls += 1
                clarity = 2 if self.quality_calls == 1 else 5
                return SimpleNamespace(content=_quality_response({"clarity": clarity, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5}))
            return SimpleNamespace(content=json.dumps({"hallucinations": [], "framing_issues": []}))

    writer = WriterAgent(chat_model=FakeWriterModel(), output_dir=tmp_path)
    reviewer = ReviewerAgent(chat_model=FailOnceReviewerModel(), output_dir=tmp_path)

    result = run_writer_reviewer_loop(
        _literature_output(), _hypothesis_output(), _planner_output(), _coder_output(),
        max_iterations=3, quality_threshold=4, output_dir=tmp_path, writer=writer, reviewer=reviewer,
    )

    assert result["converged"] is True
    assert result["iterations_run"] == 2
    assert (tmp_path / "v1.pdf").exists()
    assert (tmp_path / "v2.pdf").exists()
    log = json.loads((tmp_path / "review_log.json").read_text())
    assert len(log) == 2
    assert log[0]["review"]["overall_pass"] is False
    assert log[1]["review"]["overall_pass"] is True


def test_loop_stops_at_max_iterations_with_unresolved_issues(tmp_path):
    class AlwaysFailReviewerModel:
        def invoke(self, messages):
            prompt = messages[1][1]
            if "Score this research paper draft" in prompt:
                return SimpleNamespace(content=_quality_response({"clarity": 2, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5}))
            return SimpleNamespace(content=json.dumps({"hallucinations": [], "framing_issues": []}))

    writer = WriterAgent(chat_model=FakeWriterModel(), output_dir=tmp_path)
    reviewer = ReviewerAgent(chat_model=AlwaysFailReviewerModel(), output_dir=tmp_path)

    result = run_writer_reviewer_loop(
        _literature_output(), _hypothesis_output(), _planner_output(), _coder_output(),
        max_iterations=2, quality_threshold=4, output_dir=tmp_path, writer=writer, reviewer=reviewer,
    )

    assert result["converged"] is False
    assert result["iterations_run"] == 2
    assert len(result["unresolved_issues"]) > 0
    assert any("clarity" in issue for issue in result["unresolved_issues"])
    assert result["final_paper_path"] == str(tmp_path / "v2.pdf")
