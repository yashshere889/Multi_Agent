import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.agents.reviewer import checks
from research_pipeline.agents.reviewer.reviewer_agent import ReviewerAgent, ReviewerAgentError
from research_pipeline.agents.reviewer.schema import SchemaValidationError, validate_output
from research_pipeline.agents.writer.citations import build_paper_index
from research_pipeline.agents.writer.pdf_builder import build_pdf


# -- schema.py: output validation ------------------------------------------------------


def _valid_output() -> dict:
    return {
        "iteration": 1,
        "hallucinations": [{"location": "Introduction", "claim": "c", "issue": "i"}],
        "citation_issues": [],
        "results_accuracy_issues": [],
        "hypothesis_coverage_issues": [{"hypothesis_id": "H1", "location": "Discussion > H1", "issue": "i"}],
        "quality_scores": {"clarity": 4, "flow": 4, "tone": 4, "structure": 4, "limitations_honesty": 4},
        "overall_pass": False,
        "feedback_for_writer": "fix things",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def test_validate_output_accepts_well_formed_result():
    validate_output(_valid_output())  # should not raise


def test_validate_output_rejects_missing_field():
    data = _valid_output()
    del data["overall_pass"]
    with pytest.raises(SchemaValidationError, match="overall_pass"):
        validate_output(data)


def test_validate_output_rejects_out_of_range_score():
    data = _valid_output()
    data["quality_scores"]["clarity"] = 7
    with pytest.raises(SchemaValidationError, match="clarity"):
        validate_output(data)


def test_validate_output_rejects_missing_score_key():
    data = _valid_output()
    del data["quality_scores"]["tone"]
    with pytest.raises(SchemaValidationError, match="tone"):
        validate_output(data)


# -- checks.py: deterministic citation/results/coverage checks --------------------------


def _paper(**overrides) -> dict:
    base = {"title": "RAG Paper", "authors": ["A. Smith"], "abstract": "abc", "year": 2020, "arxiv_id": "1", "source": "arxiv"}
    return {**base, **overrides}


def test_check_citations_flags_id_not_in_index():
    index = build_paper_index([_paper()])
    issues = checks.check_citations({}, index, citations_used=["1", "999"])
    assert len(issues) == 1
    assert "999" in issues[0]["issue"]


def test_check_citations_flags_unmatched_surname_year_in_text():
    index = build_paper_index([_paper()])
    section_texts = {"Introduction": "As shown by (Nobody, 2099), this holds."}
    issues = checks.check_citations(section_texts, index, citations_used=["1"])
    assert len(issues) == 1
    assert "Nobody, 2099" in issues[0]["issue"]
    assert issues[0]["location"] == "Introduction"


def test_check_citations_accepts_real_citation_in_either_style():
    index = build_paper_index([_paper()])
    section_texts = {"Intro": "(Smith, 2020) and separately Smith (2020) again."}
    issues = checks.check_citations(section_texts, index, citations_used=["1"])
    assert issues == []


def test_check_citations_flags_leaked_unresolved_marker_syntax():
    """A 2026-08-17 production run (job 10247173) shipped a "reviewed" paper
    with malformed [[cite:...]]-shaped marker text printed raw in the final
    PDF — a defect neither this function (before this check existed) nor the
    LLM hallucination pass had a check aimed at (the LLM pass quoted the
    leaked text back as an ordinary "needs more grounding" hallucination
    instead). This test's marker uses an id that isn't in the index at all,
    so it's a genuine leak regardless of what CitationRegistry's own
    resolution logic can or can't already handle — the Reviewer must catch
    any [[cite:/[[citet: text that reaches the printed page, on its own."""
    index = build_paper_index([_paper()])
    section_texts = {"Related Work": "Composite indices lack validation [[cite:999],[888]]. Later text."}
    issues = checks.check_citations(section_texts, index, citations_used=["1"])
    assert len(issues) == 1
    assert issues[0]["location"] == "Related Work"
    assert "[[cite:" in issues[0]["issue"]


def test_check_citations_caps_leaked_marker_issues_per_section():
    index = build_paper_index([_paper()])
    leaked = " ".join(f"[[cite:{900 + i}],[{950 + i}]]." for i in range(10))
    issues = checks.check_citations({"Methods": leaked}, index, citations_used=["1"])
    assert len(issues) == 5  # _MAX_LEAKED_MARKER_ISSUES_PER_SECTION, not 10


def test_check_citations_does_not_flag_a_fully_resolved_section():
    index = build_paper_index([_paper()])
    section_texts = {"Intro": "A grounded claim (Smith, 2020) with no marker syntax left."}
    issues = checks.check_citations(section_texts, index, citations_used=["1"])
    assert issues == []


def _experiment(hid, status, accuracy=None, meets=None, reason=""):
    if status != "completed":
        return {"hypothesis_id": hid, "status": status, "reason": reason, "assumptions_made": [], "code_path": None, "results": None}
    return {
        "hypothesis_id": hid, "status": "completed", "reason": "", "assumptions_made": [], "code_path": f"experiments/{hid}",
        "results": {"metrics": {"accuracy": accuracy}, "meets_success_criteria": meets, "notes": ""},
    }


def test_check_results_accuracy_flags_mismatched_metric():
    coder_output = {"experiments": [_experiment("H1", "completed", accuracy=0.87, meets=True)]}
    results_subsections = {"H1": "Accuracy was 94% on the benchmark."}
    issues = checks.check_results_accuracy(results_subsections, coder_output)
    assert len(issues) == 1
    assert "0.87" in issues[0]["actual"]
    assert "94%" in issues[0]["claimed"]


def test_check_results_accuracy_accepts_matching_metric_in_percentage_form():
    coder_output = {"experiments": [_experiment("H1", "completed", accuracy=0.8, meets=True)]}
    results_subsections = {"H1": "Accuracy was 80% on the benchmark."}
    assert checks.check_results_accuracy(results_subsections, coder_output) == []


def test_check_results_accuracy_accepts_matching_metric_in_raw_form():
    coder_output = {"experiments": [_experiment("H1", "completed", accuracy=0.8, meets=True)]}
    results_subsections = {"H1": "The measured accuracy metric was 0.8."}
    assert checks.check_results_accuracy(results_subsections, coder_output) == []


def test_check_results_accuracy_flags_skipped_experiment_described_as_successful():
    coder_output = {"experiments": [_experiment("H2", "skipped", reason="infeasible")]}
    results_subsections = {"H2": "The experiment succeeded and results show strong performance."}
    issues = checks.check_results_accuracy(results_subsections, coder_output)
    assert len(issues) == 1
    assert "skipped" in issues[0]["actual"]


def test_check_results_accuracy_accepts_skipped_experiment_with_clear_disclaimer():
    coder_output = {"experiments": [_experiment("H2", "skipped", reason="infeasible")]}
    results_subsections = {"H2": "This experiment was not executed because it was marked infeasible."}
    assert checks.check_results_accuracy(results_subsections, coder_output) == []


def _verdict(hid, verdict, reason="r", statement="s", rationale="rat"):
    return {"hypothesis_id": hid, "statement": statement, "rationale": rationale, "verdict": verdict, "reason": reason}


def test_check_hypothesis_coverage_flags_missing_subsection():
    verdicts = {"H1": _verdict("H1", "supported")}
    issues = checks.check_hypothesis_coverage({}, {}, ["H1"], verdicts)
    locations = {i["location"] for i in issues}
    assert "Results" in locations
    assert "Discussion" in locations


def test_check_hypothesis_coverage_flags_overstated_verdict():
    verdicts = {"H3": _verdict("H3", "inconclusive", reason="skipped")}
    results_subsections = {"H3": "text"}
    discussion_subsections = {"H3": "The experiment clearly supports the hypothesis with strong evidence."}
    issues = checks.check_hypothesis_coverage(results_subsections, discussion_subsections, ["H3"], verdicts)
    assert len(issues) == 1
    assert issues[0]["hypothesis_id"] == "H3"
    assert "inconclusive" in issues[0]["issue"]


def test_check_hypothesis_coverage_accepts_consistent_framing():
    verdicts = {"H1": _verdict("H1", "supported")}
    results_subsections = {"H1": "text"}
    discussion_subsections = {"H1": "Results support the hypothesis."}
    assert checks.check_hypothesis_coverage(results_subsections, discussion_subsections, ["H1"], verdicts) == []


# -- reviewer_agent.py: orchestration, with a fake (JSON-returning) chat model ----------


class FakeChatModel:
    """Returns canned JSON looked up by a keyword found in the prompt — mirrors
    the other agents' FakeChatModel, but for JSON responses (invoke_json), not
    plain prose (unlike WriterAgent's fake, which returns text directly)."""

    def __init__(self, response_by_keyword: dict[str, str], default: str = '{"hallucinations": []}'):
        self._response_by_keyword = response_by_keyword
        self._default = default
        self.calls = []
        # ReviewerAgent._call_json now passes a per-call max_tokens (see
        # _bounded_max_tokens), so **kwargs is required here — mirrors
        # test_coder_agent.py's FakeChatModel for the same reason.
        self.call_kwargs = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        prompt_text = messages[-1][1]
        for keyword, response in self._response_by_keyword.items():
            if keyword in prompt_text:
                return SimpleNamespace(content=response)
        return SimpleNamespace(content=self._default)


def _quality_response(scores=None) -> str:
    scores = scores or {"clarity": 5, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5}
    return json.dumps({"quality_scores": scores, "quality_notes": {}})


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


def _plan(hid, feasible=True) -> dict:
    return {
        "hypothesis_id": hid, "feasible": feasible, "feasibility_notes": "ok", "objective": "o",
        "variables": {"independent": [], "dependent": []}, "design": "d",
        "data_requirements": {"source": "s", "description": "d", "preprocessing_steps": []},
        "methods": [], "evaluation": {"metrics": ["accuracy"], "baseline": "b", "success_criteria": "sc"},
        "implementation_steps": [{"step": 1, "description": "d"}], "estimated_complexity": "low", "risks": [],
    }


def _planner_output(plans) -> dict:
    return {
        "experiment_plans": plans, "shared_infrastructure": [],
        "priority_order": [{"hypothesis_id": p["hypothesis_id"], "rank": i + 1, "justification": "j"} for i, p in enumerate(plans)],
        "source_hypothesis_ids": [p["hypothesis_id"] for p in plans],
        "generated_at": "2026-01-01T00:00:00+00:00", "model": "test-model",
    }


def _coder_output(experiments) -> dict:
    return {
        "experiments": experiments, "shared_infrastructure_path": "experiments/_shared",
        "source_hypothesis_ids": [e["hypothesis_id"] for e in experiments],
        "generated_at": "2026-01-01T00:00:00+00:00", "model": "test-model",
    }


def _build_good_paper(tmp_path: Path) -> tuple[Path, dict]:
    sections = [
        ("Introduction", "Motivating text (Smith, 2020)."),
        ("Related Work", "Smith (2020) covers this."),
        ("Hypotheses", "H1: s.\n\nH2: s.\n\nH3: s."),
        ("Methods", "## H1\n\nMethods.\n\n## H2\n\nMethods.\n\n## H3\n\nMethods."),
        ("Results", "## H1\n\nAccuracy was 0.9.\n\n## H2\n\nAccuracy was 0.9.\n\n## H3\n\nAccuracy was 0.9."),
        ("Discussion", "## H1 — supported\n\nResults support this.\n\n## H2 — supported\n\nResults support this.\n\n## H3 — supported\n\nResults support this."),
        ("Limitations", "Some limitations."),
        ("Future Work", "Future work."),
    ]
    path = tmp_path / "paper.pdf"
    build_pdf(path, title="Title", abstract="Abstract.", sections=sections, references=["[1] A. Smith (2020). RAG Paper."])
    paper_summary = {
        "paper_path": str(path),
        "sections_generated": ["Title", "Abstract"] + [h for h, _ in sections] + ["References"],
        "hypotheses_supported": ["H1", "H2", "H3"], "hypotheses_refuted": [], "hypotheses_inconclusive": [],
        "citations_used": ["1"], "notes_for_review": [],
        "generated_at": "2026-01-01T00:00:00+00:00", "model": "test-model",
    }
    return path, paper_summary


def _literature_output() -> dict:
    return {"papers": [{"title": "RAG Paper", "authors": ["A. Smith"], "abstract": "abc", "year": 2020, "arxiv_id": "1", "source": "arxiv"}]}


def test_run_passes_when_paper_is_accurate_and_scores_are_high(tmp_path):
    paper_path, paper_summary = _build_good_paper(tmp_path)
    coder_output = _coder_output([
        _experiment("H1", "completed", accuracy=0.9, meets=True),
        _experiment("H2", "completed", accuracy=0.9, meets=True),
        _experiment("H3", "completed", accuracy=0.9, meets=True),
    ])
    fake_model = FakeChatModel({"Score this research paper draft": _quality_response()})
    agent = ReviewerAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        paper_path, paper_summary, _literature_output(), _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]), coder_output,
        iteration=1, quality_threshold=4,
    )

    assert result["overall_pass"] is True
    assert result["hallucinations"] == []
    assert result["citation_issues"] == []
    assert result["results_accuracy_issues"] == []
    assert result["hypothesis_coverage_issues"] == []
    assert result["iteration"] == 1

    written = list(tmp_path.glob("review_*.json"))
    assert len(written) == 1


def test_run_fails_when_quality_score_below_threshold(tmp_path):
    paper_path, paper_summary = _build_good_paper(tmp_path)
    coder_output = _coder_output([
        _experiment("H1", "completed", accuracy=0.9, meets=True),
        _experiment("H2", "completed", accuracy=0.9, meets=True),
        _experiment("H3", "completed", accuracy=0.9, meets=True),
    ])
    fake_model = FakeChatModel({"Score this research paper draft": _quality_response({"clarity": 2, "flow": 5, "tone": 5, "structure": 5, "limitations_honesty": 5})})
    agent = ReviewerAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        paper_path, paper_summary, _literature_output(), _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]), coder_output,
        iteration=1, quality_threshold=4,
    )

    assert result["overall_pass"] is False
    assert "clarity" in result["feedback_for_writer"]


def test_run_uses_llm_flagged_hallucination(tmp_path):
    paper_path, paper_summary = _build_good_paper(tmp_path)
    coder_output = _coder_output([
        _experiment("H1", "completed", accuracy=0.9, meets=True),
        _experiment("H2", "completed", accuracy=0.9, meets=True),
        _experiment("H3", "completed", accuracy=0.9, meets=True),
    ])
    hallucination_response = json.dumps({"hallucinations": [{"claim": "Motivating text (Smith, 2020).", "issue": "no such claim in the literature summary"}]})
    fake_model = FakeChatModel({
        "Score this research paper draft": _quality_response(),
        'reviewing the "Introduction"': hallucination_response,
    })
    agent = ReviewerAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        paper_path, paper_summary, _literature_output(), _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]), coder_output,
        iteration=1, quality_threshold=4,
    )

    assert result["overall_pass"] is False
    assert len(result["hallucinations"]) == 1
    assert result["hallucinations"][0]["location"] == "Introduction"


def test_run_rejects_malformed_hypothesis_input(tmp_path):
    paper_path, paper_summary = _build_good_paper(tmp_path)
    agent = ReviewerAgent(chat_model=FakeChatModel({}), output_dir=tmp_path)
    with pytest.raises(ReviewerAgentError, match="Hypothesis Agent's output schema"):
        agent.run(paper_path, paper_summary, _literature_output(), {"hypotheses": "not a list"}, _planner_output([]), _coder_output([]))
