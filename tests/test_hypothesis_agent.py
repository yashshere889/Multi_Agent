import json
from types import SimpleNamespace

import pytest

from research_pipeline.agents.hypothesis.hypothesis_agent import HypothesisAgent, HypothesisAgentError
from research_pipeline.agents.hypothesis.papers import chunk_papers, normalize_paper, normalize_papers
from research_pipeline.agents.hypothesis.schema import SchemaValidationError, validate_output


# -- papers.py: normalization + chunking -------------------------------------------------


def test_normalize_paper_prefers_arxiv_id_then_paper_id_then_doi():
    assert normalize_paper({"title": "T", "abstract": "A", "arxiv_id": "123"}, 0)["id"] == "123"
    assert normalize_paper({"title": "T", "abstract": "A", "paper_id": "abc"}, 0)["id"] == "abc"
    assert normalize_paper({"title": "T", "abstract": "A", "doi": "10.1/x"}, 0)["id"] == "10.1/x"
    assert normalize_paper({"title": "T", "abstract": "A"}, 3)["id"] == "paper_3"


def test_normalize_paper_drops_papers_with_no_usable_content():
    assert normalize_paper({"title": "", "abstract": ""}, 0) is None
    assert normalize_paper("not a dict", 0) is None
    assert normalize_paper({}, 0) is None


def test_normalize_paper_prefers_full_text_over_abstract_when_present():
    paper = normalize_paper({"title": "T", "abstract": "short", "full_text": "the whole paper"}, 0)
    assert paper["full_text"] == "the whole paper"


def test_normalize_papers_drops_bad_entries_but_keeps_good_ones():
    raw = [{"title": "Good", "abstract": "A"}, {"title": "", "abstract": ""}, None]
    normalized = normalize_papers(raw)
    assert len(normalized) == 1
    assert normalized[0]["title"] == "Good"


def test_normalize_papers_empty_input_returns_empty_list():
    assert normalize_papers([]) == []


def test_chunk_papers_splits_on_char_budget():
    papers = [normalize_paper({"title": f"P{i}", "abstract": "x" * 100}, i) for i in range(5)]
    batches = chunk_papers(papers, max_chars_per_batch=250)
    assert sum(len(b) for b in batches) == 5
    assert all(len(b) >= 1 for b in batches)


def test_chunk_papers_gives_oversized_single_paper_its_own_batch():
    huge = normalize_paper({"title": "Huge", "abstract": "x" * 10000}, 0)
    small = normalize_paper({"title": "Small", "abstract": "y" * 10}, 1)
    batches = chunk_papers([huge, small], max_chars_per_batch=500)
    assert len(batches) == 2
    assert batches[0] == [huge]


# -- schema.py: output validation ------------------------------------------------------


def _valid_output() -> dict:
    return {
        "literature_summary": "summary",
        "methods_overview": [{"method": "RAG", "papers_using_it": ["1"], "notes": "common"}],
        "gaps": [{"gap": "unstudied X", "supporting_evidence": ["1"], "notes": "n"}],
        "hypotheses": [
            {
                "id": f"H{i}",
                "statement": "s",
                "rationale": "r",
                "related_gaps": ["unstudied X"],
                "related_methods": ["RAG"],
                "suggested_variables": {"independent": ["a"], "dependent": ["b"]},
            }
            for i in (1, 2, 3)
        ],
        "source_paper_ids": ["1"],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def test_validate_output_accepts_well_formed_result():
    validate_output(_valid_output())  # should not raise


def test_validate_output_rejects_wrong_hypothesis_count():
    data = _valid_output()
    data["hypotheses"] = data["hypotheses"][:2]
    with pytest.raises(SchemaValidationError, match="exactly 3"):
        validate_output(data)


def test_validate_output_rejects_missing_field():
    data = _valid_output()
    del data["literature_summary"]
    with pytest.raises(SchemaValidationError, match="literature_summary"):
        validate_output(data)


def test_validate_output_rejects_malformed_suggested_variables():
    data = _valid_output()
    data["hypotheses"][0]["suggested_variables"] = {"independent": ["a"]}  # missing "dependent"
    with pytest.raises(SchemaValidationError, match="dependent"):
        validate_output(data)


# -- hypothesis_agent.py: orchestration, with a fake chat model ------------------------


class FakeChatModel:
    """Stands in for a real ChatOpenAI: returns canned JSON responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=next(self._responses))


def _batch_response() -> str:
    return json.dumps({
        "batch_summary": "batch covers X",
        "methods": [{"method": "RAG", "paper_ids": ["1"], "notes": "used once"}],
        "observations": [{"observation": "no long-context eval", "paper_ids": ["1"], "type": "gap"}],
    })


def _synthesis_response() -> str:
    return json.dumps({
        "literature_summary": "the field collectively finds X",
        "methods_overview": [{"method": "RAG", "papers_using_it": ["1"], "notes": "common"}],
        "gaps": [{"gap": "no long-context eval", "supporting_evidence": ["1"], "notes": "repeated"}],
    })


def _hypotheses_response() -> str:
    return json.dumps({
        "hypotheses": [
            {
                "id": f"H{i}",
                "statement": f"statement {i}",
                "rationale": "grounded in the summary above",
                "related_gaps": ["no long-context eval"],
                "related_methods": ["RAG"],
                "suggested_variables": {"independent": ["context length"], "dependent": ["accuracy"]},
            }
            for i in (1, 2, 3)
        ]
    })


def test_run_end_to_end_with_single_batch(tmp_path):
    fake_model = FakeChatModel([_batch_response(), _synthesis_response(), _hypotheses_response()])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    papers = [{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}]
    result = agent.run(papers, research_question="how to improve RAG?")

    assert result["literature_summary"] == "the field collectively finds X"
    assert len(result["hypotheses"]) == 3
    assert result["source_paper_ids"] == ["1"]

    written = list(tmp_path.glob("hypotheses_*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["hypotheses"][0]["id"] == "H1"


def test_run_raises_on_no_usable_papers(tmp_path):
    agent = HypothesisAgent(chat_model=FakeChatModel([]), output_dir=tmp_path)
    with pytest.raises(HypothesisAgentError, match="No usable papers"):
        agent.run([{"title": "", "abstract": ""}])


def test_call_json_recovers_via_repair_prompt(tmp_path):
    fake_model = FakeChatModel([
        "not json at all",  # first attempt, malformed
        _batch_response(),  # repair attempt succeeds
        _synthesis_response(),
        _hypotheses_response(),
    ])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)
    result = agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])
    assert len(result["hypotheses"]) == 3
    assert fake_model.calls[1][-1][0] == "human"  # repair prompt was sent


def test_run_raises_and_dumps_debug_file_on_schema_failure(tmp_path):
    bad_hypotheses = json.dumps({"hypotheses": []})  # violates "exactly 3"
    fake_model = FakeChatModel([_batch_response(), _synthesis_response(), bad_hypotheses])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    with pytest.raises(HypothesisAgentError, match="schema validation"):
        agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])

    assert list(tmp_path.glob("hypotheses_*_invalid.json"))
