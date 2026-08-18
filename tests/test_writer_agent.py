import json
from types import SimpleNamespace

import pytest

from research_pipeline.agents.writer.citations import CitationRegistry, build_paper_index, strip_unverified_literal_citations
from research_pipeline.agents.writer.schema import SchemaValidationError, validate_output
from research_pipeline.agents.writer.writer_agent import WriterAgent, WriterAgentError, compute_hypothesis_verdict, extract_literature_papers


# -- schema.py: output validation ------------------------------------------------------


def _valid_output() -> dict:
    return {
        "paper_path": "outputs/paper_20260101T000000Z.pdf",
        "sections_generated": ["Title", "Abstract", "Introduction"],
        "hypotheses_supported": ["H1"],
        "hypotheses_refuted": ["H2"],
        "hypotheses_inconclusive": ["H3"],
        "citations_used": ["1", "2"],
        "notes_for_review": [],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def test_validate_output_accepts_well_formed_result():
    validate_output(_valid_output())  # should not raise


def test_validate_output_rejects_missing_field():
    data = _valid_output()
    del data["paper_path"]
    with pytest.raises(SchemaValidationError, match="paper_path"):
        validate_output(data)


def test_validate_output_rejects_hypothesis_in_two_buckets():
    data = _valid_output()
    data["hypotheses_refuted"].append("H1")  # H1 is already "supported"
    with pytest.raises(SchemaValidationError, match="overlap"):
        validate_output(data)


# -- citations.py: paper indexing + marker resolution -----------------------------------


def _raw_paper(**overrides) -> dict:
    base = {"title": "Retrieval Augmented Generation", "authors": ["A. Smith"], "abstract": "abc", "year": 2020, "arxiv_id": "1706.03762", "source": "arxiv"}
    return {**base, **overrides}


def test_build_paper_index_uses_same_id_scheme_as_hypothesis_agent():
    index = build_paper_index([_raw_paper(), _raw_paper(title="Other", arxiv_id=None, paper_id="abc")])
    assert set(index.keys()) == {"1706.03762", "abc"}


def test_build_paper_index_skips_unusable_papers():
    index = build_paper_index([_raw_paper(), {"title": "", "abstract": ""}])
    assert len(index) == 1


def _scanned_registry(index: dict, texts_by_section: dict[str, str]) -> CitationRegistry:
    """Runs the required scan() -> finalize() sequence before resolve() is
    allowed (see citations.py's module docstring for why author-year
    disambiguation needs the full cited-paper set up front)."""
    registry = CitationRegistry(index)
    for section, text in texts_by_section.items():
        registry.scan(text, section=section)
    registry.finalize()
    return registry


def test_citation_registry_resolves_parenthetical_citation():
    index = build_paper_index([_raw_paper(arxiv_id="1")])
    text = "Claim A [[cite:1]]."
    registry = _scanned_registry(index, {"Intro": text})
    assert registry.resolve(text, section="Intro") == "Claim A (Smith, 2020)."


def test_citation_registry_resolves_narrative_citet_citation():
    index = build_paper_index([_raw_paper(arxiv_id="1")])
    text = "As [[citet:1]] show, X holds."
    registry = _scanned_registry(index, {"Intro": text})
    assert registry.resolve(text, section="Intro") == "As Smith (2020) show, X holds."


def test_citation_registry_groups_multiple_ids_in_one_marker():
    index = build_paper_index([_raw_paper(arxiv_id="1"), _raw_paper(title="Second", authors=["C. Jones"], year=2021, arxiv_id="2")])
    text = "Two sources agree [[cite:1,2]]."
    registry = _scanned_registry(index, {"Intro": text})
    assert registry.resolve(text, section="Intro") == "Two sources agree (Smith, 2020; Jones, 2021)."


def test_citation_registry_merges_adjacent_parenthetical_markers():
    index = build_paper_index([_raw_paper(arxiv_id="1"), _raw_paper(title="Second", authors=["C. Jones"], year=2021, arxiv_id="2")])
    text = "Two sources agree [[cite:1]][[cite:2]]."
    registry = _scanned_registry(index, {"Intro": text})
    assert registry.resolve(text, section="Intro") == "Two sources agree (Smith, 2020; Jones, 2021)."


def test_citation_registry_strips_unknown_id_and_records_it_as_unresolved():
    index = build_paper_index([_raw_paper(arxiv_id="1")])
    text = "A real claim [[cite:1]] and a fake one [[cite:999]]."
    registry = _scanned_registry(index, {"Related Work": text})
    resolved = registry.resolve(text, section="Related Work")
    assert "[[cite:999]]" not in resolved
    assert "999" not in resolved
    assert registry.unresolved == [{"section": "Related Work", "paper_id": "999"}]


def test_citation_registry_tolerates_stray_bracket_around_single_id():
    """Regression test for a 2026-08-17 production run (job 10247173): the
    model habitually wraps a cited id in its own extra bracket. For a single
    id this happened to still satisfy the old regex, but the captured id kept
    a stray leading "[" and so was always treated as unknown — a real,
    grounded citation silently dropped over one stray character."""
    index = build_paper_index([_raw_paper(arxiv_id="1")])
    text = "Claim A [[cite:[1]]."
    registry = _scanned_registry(index, {"Intro": text})
    assert registry.resolve(text, section="Intro") == "Claim A (Smith, 2020)."
    assert registry.unresolved == []


def test_citation_registry_tolerates_stray_brackets_around_multiple_ids():
    """The actual malformed shape observed in production:
    [[cite:ID1],[ID2]] — the first id bare, the second individually
    bracketed. The old regex ([^\\]]+ for the id group) couldn't match this
    at all (the internal "]" breaks it), so the whole marker fell through
    unresolved and printed as literal bracket-soup in the final PDF."""
    index = build_paper_index([_raw_paper(arxiv_id="1"), _raw_paper(title="Second", authors=["C. Jones"], year=2021, arxiv_id="2")])
    text = "Two sources agree [[cite:1],[2]]."
    registry = _scanned_registry(index, {"Intro": text})
    resolved = registry.resolve(text, section="Intro")
    assert resolved == "Two sources agree (Smith, 2020; Jones, 2021)."
    assert "[[cite:" not in resolved
    assert registry.unresolved == []


def test_citation_registry_caps_repeated_unresolved_notes_from_one_marker():
    """Regression test for the other half of job 10247173: a degenerate
    generation loop produced one syntactically valid marker repeating the
    same non-paper id thousands of times (e.g. [[cite:H3,H3,H3,...]]) — it
    resolves to "" correctly, but logged one identical "unknown paper id"
    note per repetition, so a single bad completion produced ~2,400
    near-duplicate notes. Repeats within one marker are now capped."""
    index = build_paper_index([_raw_paper(arxiv_id="1")])
    text = "[[cite:" + ",".join(["999"] * 20) + "]]"
    registry = _scanned_registry(index, {"Limitations": text})
    assert registry.unresolved == [{"section": "Limitations", "paper_id": "999"}] * 3


def test_citation_registry_unclosed_marker_still_fails_to_match():
    """A genuinely truncated marker (no closing "]]" at all) must still fail
    to match — the fix for stray brackets must not become so lenient that it
    starts accepting an incomplete marker as if it had closed."""
    index = build_paper_index([_raw_paper(arxiv_id="1")])
    text = "Claim A [[cite:1"
    registry = _scanned_registry(index, {"Intro": text})
    assert registry.resolve(text, section="Intro") == text  # left untouched, not silently resolved
    assert registry.unresolved == []


def test_citation_registry_disambiguates_same_author_same_year():
    index = build_paper_index([
        _raw_paper(arxiv_id="1", title="Alpha Study", authors=["A. Smith"], year=2020),
        _raw_paper(arxiv_id="2", title="Beta Study", authors=["A. Smith", "B. Jones"], year=2020),
    ])
    text = "See [[cite:1]] and [[cite:2]]."
    registry = _scanned_registry(index, {"Intro": text})
    resolved = registry.resolve(text, section="Intro")
    assert resolved == "See (Smith, 2020a) and (Smith and Jones, 2020b)."


def test_citation_registry_formats_references_alphabetically_not_by_citation_order():
    index = build_paper_index([
        _raw_paper(arxiv_id="1", title="Zeta Paper", authors=["Z. Zephyr"], year=2019),
        _raw_paper(arxiv_id="2", title="Alpha Paper", authors=["A. Smith", "B. Jones"], year=2020),
    ])
    # cited in reverse-alphabetical order in the text...
    text = "[[cite:1]] and [[cite:2]]."
    registry = _scanned_registry(index, {"Intro": text})
    registry.resolve(text, section="Intro")
    # ...but References is alphabetical by first-author surname, so Smith (2) comes before Zephyr (1)
    assert registry.citation_order() == ["2", "1"]
    references = registry.references()
    assert references[0].startswith("A. Smith and B. Jones (2020). Alpha Paper.")
    assert references[1].startswith("Z. Zephyr (2019). Zeta Paper.")


def test_strip_unverified_literal_citations_drops_unknown_parenthetical():
    index = build_paper_index([_raw_paper(arxiv_id="1")])  # A. Smith, 2020
    text, notes = strip_unverified_literal_citations(
        "This finding (Mamidi, 2022) suggests otherwise.", index, section="Discussion"
    )
    assert "(Mamidi, 2022)" not in text
    assert "Mamidi" not in text
    assert any("Mamidi" in n and "Discussion" in n for n in notes)


def test_strip_unverified_literal_citations_keeps_known_parenthetical():
    index = build_paper_index([_raw_paper(arxiv_id="1")])  # A. Smith, 2020
    text, notes = strip_unverified_literal_citations(
        "This finding (Smith, 2020) is consistent.", index, section="Discussion"
    )
    assert "(Smith, 2020)" in text
    assert notes == []


def test_strip_unverified_literal_citations_narrative_keeps_name_drops_year():
    index = build_paper_index([_raw_paper(arxiv_id="1")])  # A. Smith, 2020
    text, notes = strip_unverified_literal_citations(
        "Kruff and Tran (2023) show promise.", index, section="Related Work"
    )
    assert text == "Kruff and Tran show promise."
    assert any("Kruff and Tran" in n and "2023" in n for n in notes)


def test_strip_unverified_literal_citations_keeps_known_narrative():
    index = build_paper_index([_raw_paper(arxiv_id="1")])  # A. Smith, 2020
    text, notes = strip_unverified_literal_citations(
        "Smith (2020) shows promise.", index, section="Related Work"
    )
    assert text == "Smith (2020) shows promise."
    assert notes == []


# -- writer_agent.py: pure helper functions ----------------------------------------------


def test_extract_literature_papers_accepts_bare_list():
    papers, question = extract_literature_papers([_raw_paper()])
    assert len(papers) == 1
    assert question is None


def test_extract_literature_papers_accepts_disk_metadata_shape():
    papers, question = extract_literature_papers({"papers": [_raw_paper()], "research_question": "how does RAG work?"})
    assert len(papers) == 1
    assert question == "how does RAG work?"


def test_extract_literature_papers_accepts_in_memory_merged_papers_shape():
    papers, _ = extract_literature_papers({"merged_papers": [_raw_paper()]})
    assert len(papers) == 1


def test_extract_literature_papers_rejects_unrecognized_shape():
    with pytest.raises(WriterAgentError, match="literature_output must be"):
        extract_literature_papers("not a list or dict")


def _experiment(hid, status, meets=None):
    base = {"hypothesis_id": hid, "status": status, "reason": "" if status == "completed" else "needs more compute", "assumptions_made": []}
    if status == "skipped":
        return {**base, "code_path": None, "results": None}
    if status == "code_generated_not_run":
        return {**base, "code_path": f"experiments/{hid}", "results": None}
    return {**base, "code_path": f"experiments/{hid}", "results": {"metrics": {"accuracy": 0.9}, "meets_success_criteria": meets, "notes": ""}}


def test_compute_hypothesis_verdict_supported_when_meets_criteria_true():
    verdict, _ = compute_hypothesis_verdict("H1", {"H1": _experiment("H1", "completed", True)})
    assert verdict == "supported"


def test_compute_hypothesis_verdict_refuted_when_meets_criteria_false():
    verdict, _ = compute_hypothesis_verdict("H1", {"H1": _experiment("H1", "completed", False)})
    assert verdict == "refuted"


def test_compute_hypothesis_verdict_inconclusive_when_meets_criteria_unknown():
    verdict, _ = compute_hypothesis_verdict("H1", {"H1": _experiment("H1", "completed", "unknown")})
    assert verdict == "inconclusive"


def test_compute_hypothesis_verdict_inconclusive_when_skipped():
    verdict, reason = compute_hypothesis_verdict("H1", {"H1": _experiment("H1", "skipped")})
    assert verdict == "inconclusive"
    assert "skipped" in reason


def test_compute_hypothesis_verdict_inconclusive_when_code_generated_not_run():
    verdict, reason = compute_hypothesis_verdict("H1", {"H1": _experiment("H1", "code_generated_not_run")})
    assert verdict == "inconclusive"
    assert "code_generated_not_run" in reason


def test_compute_hypothesis_verdict_inconclusive_when_no_experiment_record():
    verdict, reason = compute_hypothesis_verdict("H1", {})
    assert verdict == "inconclusive"
    assert "No Coder Agent record" in reason


# -- writer_agent.py: orchestration, with a fake chat model ------------------------------


class FakeChatModel:
    """Returns canned text looked up by a keyword found in the prompt. Unlike
    the other agents' fakes, responses are plain prose (not JSON) — the Writer
    Agent's section-drafting calls return text directly, not invoke_json."""

    def __init__(self, response_by_keyword: dict[str, str], default: str = "Drafted prose."):
        self._response_by_keyword = response_by_keyword
        self._default = default
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        prompt_text = messages[-1][1]
        for keyword, response in self._response_by_keyword.items():
            if keyword in prompt_text:
                return SimpleNamespace(content=response)
        return SimpleNamespace(content=self._default)


def _hyp(hid: str, related_gaps=None) -> dict:
    return {
        "id": hid,
        "statement": f"statement for {hid}",
        "rationale": "grounded rationale",
        "related_gaps": related_gaps if related_gaps is not None else ["gap A"],
        "related_methods": ["RAG"],
        "suggested_variables": {"independent": ["x"], "dependent": ["y"]},
    }


def _hypothesis_output() -> dict:
    return {
        "literature_summary": "the field collectively finds X",
        "methods_overview": [{"method": "RAG", "papers_using_it": ["1"], "notes": "common"}],
        "gaps": [{"gap": "gap A", "supporting_evidence": ["1"], "notes": "repeated"}],
        "hypotheses": [_hyp("H1"), _hyp("H2"), _hyp("H3")],
        "ranking": [{"hypothesis_id": f"H{i}", "rank": i, "score": 9 - i, "justification": "j"} for i in (1, 2, 3)],
        "selected_hypothesis_id": "H1",
        "source_paper_ids": ["1"],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def _plan(hid: str, feasible: bool = True) -> dict:
    return {
        "hypothesis_id": hid,
        "feasible": feasible,
        "feasibility_notes": "fits in a single job" if feasible else "too much compute; simplified",
        "objective": f"test {hid}",
        "variables": {"independent": ["x"], "dependent": ["y"]},
        "design": "comparative benchmark",
        "data_requirements": {"source": "public dataset", "description": "d", "preprocessing_steps": []},
        "methods": [{"name": "RAG", "description": "baseline", "reused_from_literature": True}],
        "evaluation": {"metrics": ["F1"], "baseline": "vanilla RAG", "success_criteria": "F1 improves"},
        "implementation_steps": [{"step": 1, "description": "load dataset"}],
        "estimated_complexity": "low",
        "risks": ["insufficient data"],
    }


def _planner_output(plans: list[dict]) -> dict:
    return {
        "experiment_plans": plans,
        "shared_infrastructure": [],
        "priority_order": [{"hypothesis_id": p["hypothesis_id"], "rank": i + 1, "justification": "j"} for i, p in enumerate(plans)],
        "source_hypothesis_ids": [p["hypothesis_id"] for p in plans],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def _coder_output(experiments: list[dict]) -> dict:
    return {
        "experiments": experiments,
        "shared_infrastructure_path": "experiments/_shared",
        "source_hypothesis_ids": [e["hypothesis_id"] for e in experiments],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def _literature_output() -> dict:
    return {"papers": [{"title": "Retrieval Augmented Generation", "authors": ["A. Smith"], "abstract": "abc", "year": 2020, "arxiv_id": "1", "source": "arxiv"}]}


def test_run_end_to_end_writes_pdf_and_summary(tmp_path):
    fake_model = FakeChatModel(
        {"Propose a single academic paper title": "A Study Of RAG"},
        default="Drafted prose grounded in the literature [[cite:1]].",
    )
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        _literature_output(),
        _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3", feasible=False)]),
        _coder_output([
            _experiment("H1", "completed", True),
            _experiment("H2", "completed", False),
            _experiment("H3", "skipped"),
        ]),
    )

    assert result["hypotheses_supported"] == ["H1"]
    assert result["hypotheses_refuted"] == ["H2"]
    assert result["hypotheses_inconclusive"] == ["H3"]
    assert result["citations_used"] == ["1"]

    written_pdfs = list(tmp_path.glob("paper_*.pdf"))
    written_summaries = list(tmp_path.glob("paper_summary_*.json"))
    assert len(written_pdfs) == 1
    assert len(written_summaries) == 1
    assert written_pdfs[0].read_bytes().startswith(b"%PDF")
    assert json.loads(written_summaries[0].read_text())["paper_path"] == str(written_pdfs[0])


def test_run_rejects_malformed_hypothesis_input(tmp_path):
    agent = WriterAgent(chat_model=FakeChatModel({}), output_dir=tmp_path)
    with pytest.raises(WriterAgentError, match="Hypothesis Agent's output schema"):
        agent.run(_literature_output(), {"hypotheses": "not a list"}, _planner_output([]), _coder_output([]))


def test_run_rejects_planner_output_missing_a_hypothesis(tmp_path):
    # A narrowed-but-internally-consistent planner_output (source_hypothesis_ids
    # matches experiment_plans) is valid — see
    # test_run_succeeds_when_planner_output_is_narrowed_to_selected_hypothesis.
    # This tests the genuinely invalid case: source_hypothesis_ids claims a
    # hypothesis that has no corresponding plan.
    agent = WriterAgent(chat_model=FakeChatModel({}), output_dir=tmp_path)
    planner_output = _planner_output([_plan("H1"), _plan("H2")])  # missing H3
    planner_output["source_hypothesis_ids"] = ["H1", "H2", "H3"]
    with pytest.raises(WriterAgentError, match="Experiment Planner's output schema"):
        agent.run(
            _literature_output(),
            _hypothesis_output(),
            planner_output,
            _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True), _experiment("H3", "skipped")]),
        )


def test_run_succeeds_when_planner_output_is_narrowed_to_selected_hypothesis(tmp_path):
    # Mirrors the orchestrator's run_planner_node: hypothesis_output still
    # carries all 3 hypotheses (for framing), but planner_output/coder_output
    # only cover the selected one (H1) — source_hypothesis_ids says so, and
    # experiment_plans/experiments agree. This must not be treated as
    # planner_output "missing" H2/H3.
    fake_model = FakeChatModel({}, default="Drafted prose about H1.")
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        _literature_output(),
        _hypothesis_output(),  # selected_hypothesis_id == "H1", 3 hypotheses total
        _planner_output([_plan("H1")]),
        _coder_output([_experiment("H1", "completed", True)]),
    )

    assert result["hypotheses_supported"] == ["H1"]
    assert result["hypotheses_refuted"] == []
    assert result["hypotheses_inconclusive"] == []


def test_run_rejects_coder_output_missing_a_hypothesis(tmp_path):
    agent = WriterAgent(chat_model=FakeChatModel({}), output_dir=tmp_path)
    with pytest.raises(WriterAgentError, match="Coder Agent's output schema"):
        agent.run(
            _literature_output(),
            _hypothesis_output(),
            _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]),
            _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True)]),  # missing H3
        )


def test_run_flags_fabricated_citation_and_strips_it(tmp_path):
    fake_model = FakeChatModel(
        {"Write the Introduction": "A grounded claim [[cite:1]] and a fabricated one [[cite:999]]."},
    )
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        _literature_output(),
        _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]),
        _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True), _experiment("H3", "completed", True)]),
    )

    assert any("999" in note for note in result["notes_for_review"])
    pdf_path = next(tmp_path.glob("paper_*.pdf"))
    assert "999" not in pdf_path.read_bytes().decode("latin-1")


def test_run_strips_literal_fabricated_citation_text_and_flags_it(tmp_path):
    # "Mamidi" doesn't collide with _literature_output()'s "A. Smith, 2020" paper,
    # so it can only have reached the PDF as literal (non-marker) text the model
    # typed directly, in violation of SYSTEM_PROMPT's citation rule.
    fake_model = FakeChatModel(
        {"Write the Introduction": "A grounded claim [[cite:1]] and a fabricated one (Mamidi, 2022)."},
    )
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        _literature_output(),
        _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]),
        _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True), _experiment("H3", "completed", True)]),
    )

    assert any("Mamidi" in note for note in result["notes_for_review"])
    pdf_path = next(tmp_path.glob("paper_*.pdf"))
    assert "Mamidi" not in pdf_path.read_bytes().decode("latin-1")


def test_run_flags_skipped_experiment_not_disclaimed_in_results(tmp_path):
    # the fake model's default response never mentions "not run"/"skipped" etc.,
    # so the validation pass should catch H3 (skipped) not being disclaimed
    fake_model = FakeChatModel({}, default="Some generic drafted prose about the experiment.")
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        _literature_output(),
        _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3", feasible=False)]),
        _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True), _experiment("H3", "skipped")]),
    )

    assert any("H3" in note and "skipped" in note for note in result["notes_for_review"])


def test_run_raises_and_dumps_debug_file_on_summary_schema_failure(tmp_path, monkeypatch):
    fake_model = FakeChatModel({})
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    # force an invalid result shape past drafting to exercise the failure path
    import research_pipeline.agents.writer.writer_agent as writer_agent_module

    original_validate = writer_agent_module.validate_output

    def _bad_validate(data):
        raise writer_agent_module.SchemaValidationError("forced failure for test")

    monkeypatch.setattr(writer_agent_module, "validate_output", _bad_validate)

    with pytest.raises(WriterAgentError, match="schema validation"):
        agent.run(
            _literature_output(),
            _hypothesis_output(),
            _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]),
            _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True), _experiment("H3", "completed", True)]),
        )

    assert list(tmp_path.glob("paper_summary_*_invalid.json"))
    assert list(tmp_path.glob("paper_*.pdf"))  # the PDF itself is still written


# -- writer_agent.py: revision (previous_sections/feedback_by_section threading) --------


def test_run_with_paper_filename_and_summary_filename_overrides(tmp_path):
    fake_model = FakeChatModel({}, default="Drafted prose grounded in the literature [[cite:1]].")
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(
        _literature_output(), _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]),
        _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True), _experiment("H3", "completed", True)]),
        paper_filename="v1.pdf", summary_filename="v1_summary.json",
    )

    assert result["paper_path"] == str(tmp_path / "v1.pdf")
    assert (tmp_path / "v1.pdf").exists()
    assert (tmp_path / "v1_summary.json").exists()


def test_revise_includes_previous_text_and_feedback_in_revision_prompt(tmp_path):
    fake_model = FakeChatModel({}, default="Drafted prose grounded in the literature [[cite:1]].")
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    v1 = agent.run(
        _literature_output(), _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]),
        _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True), _experiment("H3", "completed", True)]),
        paper_filename="v1.pdf", summary_filename="v1_summary.json",
    )

    fake_model.calls = []  # only inspect calls made during the revision pass
    feedback_by_section = {"Introduction": ["Be more specific about the gap addressed."]}
    v2 = agent.revise(
        _literature_output(), _hypothesis_output(),
        _planner_output([_plan("H1"), _plan("H2"), _plan("H3")]),
        _coder_output([_experiment("H1", "completed", True), _experiment("H2", "completed", True), _experiment("H3", "completed", True)]),
        previous_paper_path=v1["paper_path"], previous_summary=v1, feedback_by_section=feedback_by_section,
        paper_filename="v2.pdf", summary_filename="v2_summary.json",
    )

    assert v2["paper_path"] == str(tmp_path / "v2.pdf")
    assert (tmp_path / "v2.pdf").exists()

    introduction_calls = [c for c in fake_model.calls if "Write the Introduction" in c[-1][1]]
    assert len(introduction_calls) == 1
    prompt_text = introduction_calls[0][-1][1]
    assert "You previously wrote this as:" in prompt_text
    assert "Be more specific about the gap addressed." in prompt_text

    # a section with no specific feedback still gets its previous text for continuity
    limitations_calls = [c for c in fake_model.calls if "Write the Limitations" in c[-1][1]]
    assert "You previously wrote this as:" in limitations_calls[0][-1][1]


def test_draft_related_work_multi_batch_passes_feedback_to_each_batch(monkeypatch, tmp_path):
    # Force chunk_papers() to split these two papers into two separate
    # batches (settings is a frozen dataclass, so patch the function directly
    # rather than the char-budget setting it reads), exercising _draft_batch's
    # revision path.
    import research_pipeline.agents.writer.writer_agent as writer_agent_module

    monkeypatch.setattr(writer_agent_module, "chunk_papers", lambda normalized, max_chars: [normalized[:1], normalized[1:]])
    fake_model = FakeChatModel({}, default="Some related work prose.")
    agent = WriterAgent(chat_model=fake_model, output_dir=tmp_path)

    raw_papers = [
        _raw_paper(arxiv_id="1"),
        _raw_paper(title="Second Paper", authors=["C. Jones"], year=2021, arxiv_id="2"),
    ]
    hypothesis_output = _hypothesis_output()
    revision = {
        "previous_sections": {"Related Work": "previous full synthesized text"},
        "previous_subsections": {},
        "feedback_by_section": {
            "Related Work": ['Citation issue: citation-like text "(Mamidi, 2022)" ... possibly fabricated'],
        },
    }

    agent._draft_related_work(raw_papers, hypothesis_output, hypothesis_output["hypotheses"], revision)

    batch_calls = [c for c in fake_model.calls if "Write a draft of part of the Related Work section" in c[-1][1]]
    assert len(batch_calls) == 2
    for call in batch_calls:
        prompt_text = call[-1][1]
        assert "Mamidi" in prompt_text  # feedback reached every batch
        assert "previous full synthesized text" not in prompt_text  # but not the mismatched previous-text
