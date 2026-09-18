import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from research_pipeline.agents.interdisciplinary_literature import (
    interdisciplinary_literature_agent as agent_module,
)

from research_pipeline.agents.interdisciplinary_literature.interdisciplinary_literature_agent import (
    InterdisciplinaryLiteratureAgent,
    InterdisciplinaryLiteratureAgentError,
)
from research_pipeline.agents.interdisciplinary_literature.schema import SchemaValidationError, validate_output


@pytest.fixture(autouse=True)
def _relevance_filter_off(monkeypatch):
    """Every test below except the screening section at the end predates the
    cross-field relevance screen and counts the agent's LLM calls exactly — the
    screen adds one per run, which would consume the canned bridges response.

    Turning it off here keeps those tests about what they were written to test
    (field identification, dedupe, bridge grounding); the screening section
    re-enables it explicitly via _relevance_filter_on."""
    monkeypatch.setattr(agent_module, "settings", replace(agent_module.settings, enable_relevance_filter=False))


def _relevance_filter_on(monkeypatch, **overrides):
    monkeypatch.setattr(
        agent_module,
        "settings",
        replace(agent_module.settings, enable_relevance_filter=True, **overrides),
    )


# -- schema.py: output validation ------------------------------------------------------


def _cross_field_paper(title: str = "Ecology Paper") -> dict:
    return {
        "source": "arxiv",
        "arxiv_id": "9",
        "title": title,
        "authors": ["E. Cologist"],
        "abstract": "rarefaction curves for sparse sampling",
        "year": 2019,
        "discipline": "ecology",
    }


def _valid_output() -> dict:
    return {
        "papers": [{"title": "RAG Paper", "abstract": "a", "arxiv_id": "1"}, _cross_field_paper()],
        "core_paper_ids": ["1"],
        "cross_field_papers": [_cross_field_paper()],
        "fields_explored": [{"field": "ecology", "rationale": "same sparsity problem", "queries": ["rarefaction"]}],
        "bridge_insights": [
            {
                "insight": "rarefaction curves quantify coverage under sparse sampling",
                "source_field": "ecology",
                "connection_to_core_problem": "retrieval corpora are sampled just as sparsely",
                "supporting_paper_ids": ["9"],
            }
        ],
        "research_question": "does RAG help?",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def test_validate_output_accepts_well_formed_result():
    validate_output(_valid_output())  # should not raise


def test_validate_output_accepts_a_null_research_question():
    data = _valid_output()
    data["research_question"] = None
    validate_output(data)  # should not raise


def test_validate_output_accepts_a_run_that_found_no_adjacent_fields():
    data = _valid_output()
    data["papers"] = [{"title": "RAG Paper", "abstract": "a", "arxiv_id": "1"}]
    data["cross_field_papers"] = []
    data["fields_explored"] = []
    data["bridge_insights"] = []
    validate_output(data)  # should not raise


def test_validate_output_rejects_missing_field():
    data = _valid_output()
    del data["bridge_insights"]
    with pytest.raises(SchemaValidationError, match="bridge_insights is missing"):
        validate_output(data)


def test_validate_output_rejects_a_missing_research_question_key():
    data = _valid_output()
    del data["research_question"]
    with pytest.raises(SchemaValidationError, match="research_question is missing"):
        validate_output(data)


def test_validate_output_rejects_malformed_bridge_insight():
    data = _valid_output()
    del data["bridge_insights"][0]["connection_to_core_problem"]
    with pytest.raises(SchemaValidationError, match="connection_to_core_problem"):
        validate_output(data)


def test_validate_output_rejects_cross_field_paper_missing_its_discipline():
    data = _valid_output()
    del data["cross_field_papers"][0]["discipline"]
    with pytest.raises(SchemaValidationError, match="discipline is missing"):
        validate_output(data)


def test_validate_output_rejects_a_cross_field_paper_not_in_the_merged_pool():
    data = _valid_output()
    data["cross_field_papers"] = [_cross_field_paper("A Paper That Was Never Merged")]
    with pytest.raises(SchemaValidationError, match="not present in output.papers"):
        validate_output(data)


# -- the agent: orchestration, with a fake chat model and fake search clients -----------


class FakeChatModel:
    """Stands in for a real ChatOpenAI: returns canned JSON responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=next(self._responses))


def _fields_response(*fields: str) -> str:
    return json.dumps({
        "fields": [
            {"field": field, "rationale": f"{field} faces the same problem", "queries": [f"{field} query"]}
            for field in fields
        ]
    })


def _bridges_response(*paper_ids: str) -> str:
    return json.dumps({
        "bridge_insights": [
            {
                "insight": "rarefaction curves quantify coverage under sparse sampling",
                "source_field": "ecology",
                "connection_to_core_problem": "retrieval corpora are sampled just as sparsely",
                "supporting_paper_ids": list(paper_ids),
            }
        ]
    })


def _core_papers() -> list[dict]:
    return [{"title": "RAG Paper", "abstract": "about retrieval", "arxiv_id": "1", "source": "arxiv"}]


def _agent(tmp_path, chat_model, arxiv_results=None, s2_results=None, core_results=None, **kwargs):
    """Every network call this agent makes goes through the three injected search
    functions, so a test never reaches arXiv, Semantic Scholar, or CORE."""
    calls = {"arxiv": [], "s2": [], "core": []}

    def fake_arxiv(queries, max_results):
        calls["arxiv"].append((list(queries), max_results))
        return [dict(p) for p in (arxiv_results or [])]

    def fake_s2(queries, max_results):
        calls["s2"].append((list(queries), max_results))
        return [dict(p) for p in (s2_results or [])]

    def fake_core(queries, max_results):
        calls["core"].append((list(queries), max_results))
        return [dict(p) for p in (core_results or [])]

    agent = InterdisciplinaryLiteratureAgent(
        chat_model=chat_model,
        output_dir=tmp_path,
        search_arxiv_fn=fake_arxiv,
        search_semantic_scholar_fn=fake_s2,
        search_core_fn=fake_core,
        **kwargs,
    )
    return agent, calls


def test_run_end_to_end_merges_cross_field_papers_and_bridges(tmp_path):
    found = [{"title": "Rarefaction in Ecology", "abstract": "curves", "arxiv_id": "9", "source": "arxiv"}]
    model = FakeChatModel([_fields_response("ecology"), _bridges_response("9")])
    agent, calls = _agent(tmp_path, model, arxiv_results=found)

    result = agent.run(_core_papers(), research_question="does RAG help?")

    assert [f["field"] for f in result["fields_explored"]] == ["ecology"]
    assert calls["arxiv"] == [(["ecology query"], agent.max_results_per_query)]
    assert calls["s2"] == [(["ecology query"], agent.max_results_per_query)]
    assert result["core_paper_ids"] == ["1"]
    # the merged pool is the in-domain papers plus what the cross-field search added,
    # in the shape the Hypothesis Agent already consumes
    assert [p["title"] for p in result["papers"]] == ["RAG Paper", "Rarefaction in Ecology"]
    assert result["cross_field_papers"][0]["discipline"] == "ecology"
    assert result["bridge_insights"][0]["supporting_paper_ids"] == ["9"]
    assert result["research_question"] == "does RAG help?"

    written = list(tmp_path.glob("interdisciplinary_*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["core_paper_ids"] == ["1"]


def test_run_includes_core_results_in_cross_field_search(tmp_path):
    found = [{"title": "A CORE Paper", "abstract": "x", "paper_id": "99", "source": "core"}]
    model = FakeChatModel([_fields_response("ecology"), _bridges_response("99")])
    agent, calls = _agent(tmp_path, model, core_results=found)

    result = agent.run(_core_papers(), research_question="does RAG help?")

    assert calls["core"] == [(["ecology query"], agent.max_results_per_query)]
    assert [p["title"] for p in result["papers"]] == ["RAG Paper", "A CORE Paper"]
    assert result["cross_field_papers"][0]["discipline"] == "ecology"


def test_run_dedupes_core_results_against_the_other_two_sources(tmp_path):
    # arXiv and CORE both return a paper with the same DOI — only one should survive
    arxiv_results = [{"title": "Ecology Curves", "abstract": "new", "doi": "10.1/eco", "source": "arxiv"}]
    core_results = [{"title": "Ecology Curves (mirror)", "abstract": "same doi", "doi": "10.1/eco", "source": "core"}]
    model = FakeChatModel([_fields_response("ecology"), _bridges_response()])
    agent, _ = _agent(tmp_path, model, arxiv_results=arxiv_results, core_results=core_results)

    result = agent.run(_core_papers())

    assert [p["title"] for p in result["cross_field_papers"]] == ["Ecology Curves"]
    assert len(result["papers"]) == 2


def test_run_fans_out_one_search_per_field_bounded_by_max_fields(tmp_path):
    model = FakeChatModel([_fields_response("ecology", "economics", "linguistics", "geology"), _bridges_response()])
    agent, calls = _agent(
        tmp_path,
        model,
        arxiv_results=[{"title": "Cross Paper", "abstract": "x", "arxiv_id": "9", "source": "arxiv"}],
        max_fields=2,
    )

    result = agent.run(_core_papers())

    assert [f["field"] for f in result["fields_explored"]] == ["ecology", "economics"]
    assert len(calls["arxiv"]) == 2


def test_run_dedupes_cross_field_results_against_each_other_and_the_core_papers(tmp_path):
    # one duplicate of the in-domain paper (same title, different casing/punctuation),
    # one duplicate by DOI across the two sources, one genuinely new paper
    arxiv_results = [
        {"title": "RAG paper.", "abstract": "dupe of the core paper", "arxiv_id": "1b", "source": "arxiv"},
        {"title": "Ecology Curves", "abstract": "new", "doi": "10.1/eco", "source": "arxiv"},
    ]
    s2_results = [
        {"title": "Ecology Curves (preprint)", "abstract": "same doi", "doi": "10.1/eco", "source": "semantic_scholar"},
    ]
    model = FakeChatModel([_fields_response("ecology"), _bridges_response()])
    agent, _ = _agent(tmp_path, model, arxiv_results=arxiv_results, s2_results=s2_results)

    result = agent.run(_core_papers())

    assert [p["title"] for p in result["cross_field_papers"]] == ["Ecology Curves"]
    assert len(result["papers"]) == 2


def test_run_drops_bridge_insight_paper_ids_the_model_invented(tmp_path):
    found = [{"title": "Rarefaction in Ecology", "abstract": "curves", "arxiv_id": "9", "source": "arxiv"}]
    model = FakeChatModel([_fields_response("ecology"), _bridges_response("9", "NOT_A_REAL_ID")])
    agent, _ = _agent(tmp_path, model, arxiv_results=found)

    result = agent.run(_core_papers())

    assert result["bridge_insights"][0]["supporting_paper_ids"] == ["9"]


def test_run_skips_the_bridge_call_entirely_when_no_cross_field_papers_were_found(tmp_path):
    model = FakeChatModel([_fields_response("ecology")])  # no second response configured
    agent, _ = _agent(tmp_path, model)  # searches return nothing

    result = agent.run(_core_papers())

    assert result["cross_field_papers"] == []
    assert result["bridge_insights"] == []
    assert len(model.calls) == 1  # the model was never asked to invent a bridge


def test_run_passes_the_in_domain_papers_through_when_no_adjacent_field_is_identified(tmp_path):
    model = FakeChatModel([json.dumps({"fields": []})])
    agent, calls = _agent(tmp_path, model)

    result = agent.run(_core_papers())

    assert result["fields_explored"] == []
    assert calls["arxiv"] == []  # nothing to search
    assert [p["title"] for p in result["papers"]] == ["RAG Paper"]
    assert result["bridge_insights"] == []


def test_run_searches_on_the_field_name_when_the_model_gives_no_query(tmp_path):
    model = FakeChatModel([
        json.dumps({"fields": [{"field": "ecology", "rationale": "r", "queries": []}]}),
        _bridges_response(),
    ])
    agent, calls = _agent(
        tmp_path, model, arxiv_results=[{"title": "Cross", "abstract": "x", "arxiv_id": "9", "source": "arxiv"}]
    )

    agent.run(_core_papers())

    assert calls["arxiv"] == [(["ecology"], agent.max_results_per_query)]


def test_run_raises_on_no_usable_papers(tmp_path):
    agent, _ = _agent(tmp_path, FakeChatModel([]))
    with pytest.raises(InterdisciplinaryLiteratureAgentError, match="No usable papers"):
        agent.run([{"title": "", "abstract": ""}])


def test_call_json_recovers_via_repair_prompt(tmp_path):
    model = FakeChatModel(["not json at all", _fields_response("ecology"), _bridges_response("9")])
    agent, _ = _agent(
        tmp_path, model, arxiv_results=[{"title": "Cross", "abstract": "x", "arxiv_id": "9", "source": "arxiv"}]
    )

    result = agent.run(_core_papers())

    assert [f["field"] for f in result["fields_explored"]] == ["ecology"]
    assert model.calls[1][-1][0] == "human"  # repair prompt was sent


def test_run_coerces_a_field_the_model_left_without_a_rationale(tmp_path):
    model = FakeChatModel([json.dumps({"fields": [{"field": "ecology", "rationale": None, "queries": ["q"]}]})])
    agent, _ = _agent(tmp_path, model)  # searches return nothing, so no bridge call

    result = agent.run(_core_papers())

    assert result["fields_explored"][0]["rationale"] == ""


def test_run_writes_an_invalid_output_dump_when_validation_fails(tmp_path, monkeypatch):
    from research_pipeline.agents.interdisciplinary_literature import interdisciplinary_literature_agent as module

    def boom(_result):
        raise SchemaValidationError("forced failure")

    monkeypatch.setattr(module, "validate_output", boom)
    model = FakeChatModel([json.dumps({"fields": []})])
    agent, _ = _agent(tmp_path, model)

    with pytest.raises(InterdisciplinaryLiteratureAgentError, match="schema validation"):
        agent.run(_core_papers())

    assert list(tmp_path.glob("interdisciplinary_*_invalid.json"))


# -- the cross-field relevance screen ---------------------------------------------------
#
# Before this screen existed, every paper an adjacent-field query returned entered
# `papers` — and so became citable by the Writer — purely on the strength of a query
# the model generated from a rationale two hops from the research question.


def _scores_response(*scores: int) -> str:
    return json.dumps({"scores": [{"id": f"P{i}", "score": s} for i, s in enumerate(scores)]})


def _two_cross_field_papers() -> list[dict]:
    return [
        {"title": "Rarefaction in Ecology", "abstract": "curves", "arxiv_id": "9", "source": "arxiv"},
        {"title": "Unrelated Geology Paper", "abstract": "rocks", "arxiv_id": "10", "source": "arxiv"},
    ]


def test_a_cross_field_paper_below_the_threshold_never_reaches_the_citable_pool(tmp_path, monkeypatch):
    _relevance_filter_on(monkeypatch, interdisciplinary_relevance_min_score=3)
    model = FakeChatModel([_fields_response("ecology"), _scores_response(5, 0), _bridges_response("9")])
    agent, _ = _agent(tmp_path, model, arxiv_results=_two_cross_field_papers())

    result = agent.run(_core_papers(), research_question="does RAG help?")

    assert [p["title"] for p in result["cross_field_papers"]] == ["Rarefaction in Ecology"]
    # The pool the Hypothesis Agent and Writer consume must not still contain it.
    assert [p["title"] for p in result["papers"]] == ["RAG Paper", "Rarefaction in Ecology"]


def test_the_surviving_papers_carry_the_score_they_were_judged_on(tmp_path, monkeypatch):
    _relevance_filter_on(monkeypatch, interdisciplinary_relevance_min_score=3)
    model = FakeChatModel([_fields_response("ecology"), _scores_response(4, 0), _bridges_response("9")])
    agent, _ = _agent(tmp_path, model, arxiv_results=_two_cross_field_papers())

    result = agent.run(_core_papers(), research_question="does RAG help?")

    assert result["cross_field_papers"][0]["relevance_score"] == 4


def test_filtering_renumbers_the_pool_so_positional_ids_stay_resolvable(tmp_path, monkeypatch):
    """_paper_id falls back to a paper's *position* in the merged pool when it
    carries no id of its own, so dropping a paper without rebuilding the pool
    would leave every later id pointing at the wrong paper."""
    _relevance_filter_on(monkeypatch, interdisciplinary_relevance_min_score=3)
    idless = [
        {"title": "Dropped, No Id", "abstract": "x", "source": "arxiv"},
        {"title": "Kept, No Id", "abstract": "y", "source": "arxiv"},
    ]
    # One core paper, so the surviving cross-field paper sits at pool index 1
    # and must be citable as paper_1 rather than the paper_2 it would have been.
    model = FakeChatModel([_fields_response("ecology"), _scores_response(0, 5), _bridges_response("paper_1")])
    agent, _ = _agent(tmp_path, model, arxiv_results=idless)

    result = agent.run(_core_papers(), research_question="does RAG help?")

    assert [p["title"] for p in result["papers"]] == ["RAG Paper", "Kept, No Id"]
    assert result["bridge_insights"][0]["supporting_paper_ids"] == ["paper_1"]


def test_in_domain_papers_are_never_re_screened(tmp_path, monkeypatch):
    """They arrived from the Literature Agent, which already screened them
    against the same question — re-filtering another agent's validated output
    would make this agent silently lossy on its own input."""
    _relevance_filter_on(monkeypatch, interdisciplinary_relevance_min_score=5)
    model = FakeChatModel([_fields_response("ecology"), _scores_response(0)])
    agent, _ = _agent(tmp_path, model, arxiv_results=[_two_cross_field_papers()[0]])

    result = agent.run(_core_papers(), research_question="does RAG help?")

    # Every cross-field paper was screened out, but the core paper survives and
    # the run still produces a valid pool rather than failing.
    assert result["cross_field_papers"] == []
    assert [p["title"] for p in result["papers"]] == ["RAG Paper"]


def test_a_failed_screen_keeps_every_cross_field_paper(tmp_path, monkeypatch):
    """The filter's failure direction is a wider pool, never a narrower one."""
    _relevance_filter_on(monkeypatch, interdisciplinary_relevance_min_score=3)
    model = FakeChatModel([
        _fields_response("ecology"),
        "not json",          # scoring call
        "still not json",    # its repair retry
        _bridges_response("9"),
    ])
    agent, _ = _agent(tmp_path, model, arxiv_results=_two_cross_field_papers())

    result = agent.run(_core_papers(), research_question="does RAG help?")

    assert len(result["cross_field_papers"]) == 2
    assert all(p["relevance_score"] is None for p in result["cross_field_papers"])


def test_the_screen_uses_the_transferability_rubric_not_topical_relevance(tmp_path, monkeypatch):
    """Scoring a cross-field paper on topical overlap would score every one of
    them near zero — being topically distant is what makes it cross-field."""
    _relevance_filter_on(monkeypatch)
    model = FakeChatModel([_fields_response("ecology"), _scores_response(5), _bridges_response("9")])
    agent, _ = _agent(tmp_path, model, arxiv_results=[_two_cross_field_papers()[0]])

    agent.run(_core_papers(), research_question="does RAG help?")

    scoring_prompt = model.calls[1][1][1]
    assert "could transfer" in scoring_prompt
    assert "does RAG help?" in scoring_prompt


def test_the_screen_falls_back_to_core_titles_with_no_research_question(tmp_path, monkeypatch):
    _relevance_filter_on(monkeypatch)
    agent, _ = _agent(tmp_path, FakeChatModel([]))

    objective = agent._scoring_objective({"research_question": None, "core_papers": _core_papers()})

    assert "RAG Paper" in objective
