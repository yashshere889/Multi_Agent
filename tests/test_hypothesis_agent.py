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
        "ranking": [
            {"hypothesis_id": f"H{i}", "rank": i, "score": 9 - i, "justification": "j"} for i in (1, 2, 3)
        ],
        "selected_hypothesis_id": "H1",
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


# -- schema.py: the ranking + selection rules ------------------------------------------


def test_validate_output_rejects_missing_ranking():
    data = _valid_output()
    del data["ranking"]
    with pytest.raises(SchemaValidationError, match="ranking is missing"):
        validate_output(data)


def test_validate_output_rejects_ranking_ids_that_dont_match_the_hypotheses():
    data = _valid_output()
    data["ranking"][2]["hypothesis_id"] = "H99"
    with pytest.raises(SchemaValidationError, match="don't match"):
        validate_output(data)


def test_validate_output_rejects_ranks_that_arent_a_permutation_of_1_to_3():
    data = _valid_output()
    data["ranking"][2]["rank"] = 2  # now 1, 2, 2
    with pytest.raises(SchemaValidationError, match="ranks should be exactly 1..3"):
        validate_output(data)


def test_validate_output_rejects_non_numeric_score():
    data = _valid_output()
    data["ranking"][0]["score"] = "high"
    with pytest.raises(SchemaValidationError, match="score should be a number"):
        validate_output(data)


def test_validate_output_rejects_selected_id_that_isnt_ranked_first():
    data = _valid_output()
    data["selected_hypothesis_id"] = "H2"  # H1 holds rank 1
    with pytest.raises(SchemaValidationError, match="ranked 1 is 'H1'"):
        validate_output(data)


def test_validate_output_rejects_selected_id_that_isnt_a_generated_hypothesis():
    data = _valid_output()
    data["ranking"][0]["rank"], data["ranking"][2]["rank"] = 3, 1
    data["ranking"][2]["hypothesis_id"] = "H3"
    data["selected_hypothesis_id"] = "H99"
    with pytest.raises(SchemaValidationError, match="isn't one of the generated hypothesis ids"):
        validate_output(data)


def test_validate_output_accepts_a_winner_that_isnt_the_first_hypothesis():
    data = _valid_output()
    data["ranking"] = [
        {"hypothesis_id": "H1", "rank": 3, "score": 4.0, "justification": "j"},
        {"hypothesis_id": "H2", "rank": 1, "score": 8.5, "justification": "j"},
        {"hypothesis_id": "H3", "rank": 2, "score": 6.0, "justification": "j"},
    ]
    data["selected_hypothesis_id"] = "H2"
    validate_output(data)  # should not raise


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


def _ranking_response(winner: str = "H2") -> str:
    ranks = {winner: 1}
    for hid in ("H1", "H2", "H3"):
        if hid not in ranks:
            ranks[hid] = len(ranks) + 1
    return json.dumps({
        "ranking": [
            {"hypothesis_id": hid, "rank": ranks[hid], "score": 10 - ranks[hid], "justification": "j"}
            for hid in ("H1", "H2", "H3")
        ]
    })


def test_run_end_to_end_with_single_batch(tmp_path):
    fake_model = FakeChatModel([_batch_response(), _synthesis_response(), _hypotheses_response(), _ranking_response()])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    papers = [{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}]
    result = agent.run(papers, research_question="how to improve RAG?")

    assert result["literature_summary"] == "the field collectively finds X"
    assert len(result["hypotheses"]) == 3
    assert result["source_paper_ids"] == ["1"]

    written = list(tmp_path.glob("hypotheses_*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["hypotheses"][0]["id"] == "H1"


# -- ranking, end to end ----------------------------------------------------------------


def test_run_ranks_all_three_and_selects_the_one_ranked_first(tmp_path):
    fake_model = FakeChatModel([
        _batch_response(), _synthesis_response(), _hypotheses_response(), _ranking_response(winner="H3"),
    ])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])

    # all 3 are still generated and returned — ranking is additive
    assert [h["id"] for h in result["hypotheses"]] == ["H1", "H2", "H3"]
    assert sorted(r["rank"] for r in result["ranking"]) == [1, 2, 3]
    assert result["selected_hypothesis_id"] == "H3"


def test_run_derives_the_winner_from_the_ranking_not_a_model_assertion(tmp_path):
    # The model also volunteers a (contradictory) selected_hypothesis_id; the
    # agent must take the id holding rank 1 instead.
    ranking = json.loads(_ranking_response(winner="H2"))
    ranking["selected_hypothesis_id"] = "H1"
    fake_model = FakeChatModel([
        _batch_response(), _synthesis_response(), _hypotheses_response(), json.dumps(ranking),
    ])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])

    assert result["selected_hypothesis_id"] == "H2"


def test_run_rejects_a_ranking_that_isnt_a_permutation(tmp_path):
    bad_ranking = json.dumps({
        "ranking": [{"hypothesis_id": hid, "rank": 1, "score": 5, "justification": "j"} for hid in ("H1", "H2", "H3")]
    })
    fake_model = FakeChatModel([_batch_response(), _synthesis_response(), _hypotheses_response(), bad_ranking])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    with pytest.raises(HypothesisAgentError, match="schema validation"):
        agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])


def test_run_passes_bridge_insights_into_both_prompts_when_given(tmp_path):
    fake_model = FakeChatModel([_batch_response(), _synthesis_response(), _hypotheses_response(), _ranking_response()])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    agent.run(
        [{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}],
        interdisciplinary_context=[{"insight": "ecology's rarefaction curves", "source_field": "ecology"}],
    )

    hypothesis_prompt = fake_model.calls[2][-1][1]
    ranking_prompt = fake_model.calls[3][-1][1]
    assert "rarefaction curves" in hypothesis_prompt
    assert "rarefaction curves" in ranking_prompt


def test_run_omits_the_bridge_insight_block_entirely_when_not_given(tmp_path):
    fake_model = FakeChatModel([_batch_response(), _synthesis_response(), _hypotheses_response(), _ranking_response()])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])

    assert "bridge insights" not in fake_model.calls[2][-1][1].lower()


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
        _ranking_response(),
    ])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)
    result = agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])
    assert len(result["hypotheses"]) == 3
    assert fake_model.calls[1][-1][0] == "human"  # repair prompt was sent


def test_a_doubled_rationale_key_no_longer_loses_the_whole_run(tmp_path):
    """Barkla job 10334292: the model wrote `rationationale` on two of three
    hypotheses. The rationales were complete and correct; only the key was
    doubled. Schema validation rejected the assembled output three stages later
    and the question was lost, along with its literature search and cross-field
    search. The key is now repaired deterministically before anything
    downstream sees it — no extra model call."""
    mangled = json.dumps({
        "hypotheses": [
            {
                "id": f"H{i}",
                "statement": f"statement {i}",
                # H1 got it right; H2 and H3 did not — exactly what the run produced.
                ("rationale" if i == 1 else "rationationale"): f"grounded rationale {i}",
                "related_gaps": ["no long-context eval"],
                "related_methods": ["RAG"],
                "suggested_variables": {"independent": ["context length"], "dependent": ["accuracy"]},
            }
            for i in (1, 2, 3)
        ]
    })
    fake_model = FakeChatModel([
        _batch_response(), _synthesis_response(), mangled, _ranking_response(),
    ])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])

    assert len(result["hypotheses"]) == 3
    assert [h["rationale"] for h in result["hypotheses"]] == [
        "grounded rationale 1", "grounded rationale 2", "grounded rationale 3"
    ]
    assert not any("rationationale" in h for h in result["hypotheses"])
    # Four calls: two batch/synthesis, one generation, one ranking. The repair
    # cost no extra round-trip.
    assert len(fake_model.calls) == 4
    assert not list(tmp_path.glob("hypotheses_*_invalid.json"))


def test_run_raises_and_dumps_debug_file_on_schema_failure(tmp_path):
    bad_hypotheses = json.dumps({"hypotheses": []})  # violates "exactly 3"
    fake_model = FakeChatModel([_batch_response(), _synthesis_response(), bad_hypotheses])
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    with pytest.raises(HypothesisAgentError, match="schema validation"):
        agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])

    assert list(tmp_path.glob("hypotheses_*_invalid.json"))


# -- graph.py: RetryPolicy on the LLM-calling nodes ------------------------------------


class _TransientBackendError(Exception):
    """Stands in for what actually escapes the layers below a node: a connection
    error that outlived the client's own max_retries, say. Deliberately *not*
    derived from ValueError/RuntimeError/OSError — LangGraph's default retry_on
    classifies those as persistent and won't retry them, which is the behaviour
    the schema-failure tests above depend on."""


class FlakyChatModel(FakeChatModel):
    """A FakeChatModel that raises on its first N invocations before behaving.
    A failed invocation consumes no canned response, so the retried node sees
    exactly the responses it would have seen first time round."""

    def __init__(self, responses: list[str], failures: int = 1):
        super().__init__(responses)
        self._remaining_failures = failures
        self.attempts = 0

    def invoke(self, messages):
        self.attempts += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise _TransientBackendError("connection reset by peer")
        return super().invoke(messages)


def test_retry_policy_recovers_from_one_transient_llm_failure(tmp_path):
    fake_model = FlakyChatModel(
        [_batch_response(), _synthesis_response(), _hypotheses_response(), _ranking_response()],
        failures=1,
    )
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])

    # Without RetryPolicy on analyze_batch this run dies on the first call.
    assert len(result["hypotheses"]) == 3
    assert fake_model.attempts == 5  # 1 failed + 4 that produced the canned responses
    assert len(fake_model.calls) == 4


def test_retry_policy_is_bounded_and_gives_up_after_max_attempts(tmp_path):
    fake_model = FlakyChatModel([_batch_response()], failures=99)
    agent = HypothesisAgent(chat_model=fake_model, output_dir=tmp_path)

    with pytest.raises(_TransientBackendError):
        agent.run([{"title": "Paper One", "abstract": "about RAG", "arxiv_id": "1"}])

    # max_attempts=2 means one retry, not an unbounded loop against a dead backend.
    assert fake_model.attempts == 2
