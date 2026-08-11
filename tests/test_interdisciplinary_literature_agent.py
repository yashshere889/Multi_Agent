import json
from types import SimpleNamespace

import pytest

from research_pipeline.agents.interdisciplinary_literature.interdisciplinary_literature_agent import (
    InterdisciplinaryLiteratureAgent,
    InterdisciplinaryLiteratureAgentError,
)
from research_pipeline.agents.interdisciplinary_literature.schema import SchemaValidationError, validate_output


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


def _agent(tmp_path, chat_model, arxiv_results=None, s2_results=None, **kwargs):
    """Every network call this agent makes goes through the two injected search
    functions, so a test never reaches arXiv or Semantic Scholar."""
    calls = {"arxiv": [], "s2": []}

    def fake_arxiv(queries, max_results):
        calls["arxiv"].append((list(queries), max_results))
        return [dict(p) for p in (arxiv_results or [])]

    def fake_s2(queries, max_results):
        calls["s2"].append((list(queries), max_results))
        return [dict(p) for p in (s2_results or [])]

    agent = InterdisciplinaryLiteratureAgent(
        chat_model=chat_model,
        output_dir=tmp_path,
        search_arxiv_fn=fake_arxiv,
        search_semantic_scholar_fn=fake_s2,
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
