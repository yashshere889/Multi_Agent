import json
from types import SimpleNamespace

import pytest

from research_pipeline.agents.experiment_planner.experiment_planner_agent import (
    ExperimentPlannerAgent,
    ExperimentPlannerAgentError,
)
from research_pipeline.agents.experiment_planner.schema import SchemaValidationError, validate_output


# -- schema.py: output validation ------------------------------------------------------


def _valid_plan(hypothesis_id: str) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "feasible": True,
        "feasibility_notes": "fits within a single GPU job",
        "objective": "test whether X improves Y",
        "variables": {"independent": ["method"], "dependent": ["accuracy"]},
        "design": "comparative benchmark",
        "data_requirements": {"source": "public dataset", "description": "desc", "preprocessing_steps": ["clean"]},
        "methods": [{"name": "RAG", "description": "used as baseline", "reused_from_literature": True}],
        "evaluation": {"metrics": ["F1"], "baseline": "vanilla RAG", "success_criteria": "F1 improves by 5pts"},
        "implementation_steps": [{"step": 1, "description": "load dataset X"}],
        "estimated_complexity": "medium",
        "risks": ["insufficient data"],
    }


def _valid_output() -> dict:
    return {
        "experiment_plans": [_valid_plan("H1"), _valid_plan("H2"), _valid_plan("H3")],
        "shared_infrastructure": ["shared eval harness"],
        "priority_order": [
            {"hypothesis_id": "H1", "rank": 1, "justification": "highest info gain"},
            {"hypothesis_id": "H2", "rank": 2, "justification": "cheap to implement"},
            {"hypothesis_id": "H3", "rank": 3, "justification": "addresses a minor gap"},
        ],
        "source_hypothesis_ids": ["H1", "H2", "H3"],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def test_validate_output_accepts_well_formed_result():
    validate_output(_valid_output())  # should not raise


def test_validate_output_rejects_bad_complexity_value():
    data = _valid_output()
    data["experiment_plans"][0]["estimated_complexity"] = "extreme"
    with pytest.raises(SchemaValidationError, match="estimated_complexity"):
        validate_output(data)


def test_validate_output_rejects_priority_order_id_mismatch():
    data = _valid_output()
    data["priority_order"][0]["hypothesis_id"] = "H99"
    with pytest.raises(SchemaValidationError, match="don't match"):
        validate_output(data)


def test_validate_output_rejects_non_contiguous_ranks():
    data = _valid_output()
    data["priority_order"][0]["rank"] = 5
    with pytest.raises(SchemaValidationError, match="ranks should be exactly"):
        validate_output(data)


def test_validate_output_expected_ids_catches_missing_plan():
    data = _valid_output()
    data["experiment_plans"] = data["experiment_plans"][:2]
    data["priority_order"] = data["priority_order"][:2]
    with pytest.raises(SchemaValidationError, match="missing plan"):
        validate_output(data, expected_hypothesis_ids=["H1", "H2", "H3"])


# -- experiment_planner_agent.py: orchestration, with a fake chat model ----------------


class FakeChatModel:
    """Stands in for a real ChatOpenAI: returns canned JSON responses in call order.
    Since per-hypothesis planning runs concurrently, responses are looked up by a
    keyword found in the prompt rather than strictly by call order."""

    def __init__(self, response_by_keyword: dict[str, str]):
        self._response_by_keyword = response_by_keyword
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        prompt_text = messages[-1][1]
        for keyword, response in self._response_by_keyword.items():
            if keyword in prompt_text:
                return SimpleNamespace(content=response)
        raise AssertionError(f"No fake response configured for prompt: {prompt_text[:200]!r}")


def _hypothesis_output_fixture() -> dict:
    def _hyp(hid):
        return {
            "id": hid,
            "statement": f"statement for {hid}",
            "rationale": "grounded rationale",
            "related_gaps": ["gap A"],
            "related_methods": ["RAG"],
            "suggested_variables": {"independent": ["context length"], "dependent": ["accuracy"]},
        }

    return {
        "literature_summary": "the field collectively finds X",
        "methods_overview": [{"method": "RAG", "papers_using_it": ["1"], "notes": "common"}],
        "gaps": [{"gap": "gap A", "supporting_evidence": ["1"], "notes": "repeated"}],
        "hypotheses": [_hyp("H1"), _hyp("H2"), _hyp("H3")],
        "ranking": [
            {"hypothesis_id": "H2", "rank": 1, "score": 8.5, "justification": "most testable"},
            {"hypothesis_id": "H1", "rank": 2, "score": 6.0, "justification": "feasible but broad"},
            {"hypothesis_id": "H3", "rank": 3, "score": 4.0, "justification": "hardest to run"},
        ],
        "selected_hypothesis_id": "H2",
        "source_paper_ids": ["1"],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "model": "test-model",
    }


def _plan_response(hypothesis_id: str) -> str:
    plan = _valid_plan(hypothesis_id)
    return json.dumps(plan)


def _cross_cutting_response() -> str:
    return json.dumps({
        "shared_infrastructure": ["shared eval harness"],
        "priority_order": [
            {"hypothesis_id": "H1", "rank": 1, "justification": "highest info gain"},
            {"hypothesis_id": "H2", "rank": 2, "justification": "cheap"},
            {"hypothesis_id": "H3", "rank": 3, "justification": "minor gap"},
        ],
    })


def test_run_end_to_end(tmp_path):
    fake_model = FakeChatModel({
        '"id": "H1"': _plan_response("H1"),
        '"id": "H2"': _plan_response("H2"),
        '"id": "H3"': _plan_response("H3"),
        "independently drafted": _cross_cutting_response(),
    })
    agent = ExperimentPlannerAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(_hypothesis_output_fixture())

    assert [p["hypothesis_id"] for p in result["experiment_plans"]] == ["H1", "H2", "H3"]  # input order preserved
    assert result["source_hypothesis_ids"] == ["H1", "H2", "H3"]
    assert len(result["priority_order"]) == 3

    written = list(tmp_path.glob("experiment_plan_*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["experiment_plans"][0]["hypothesis_id"] == "H1"


def _single_plan_cross_cutting_response(hypothesis_id: str) -> str:
    return json.dumps({
        "shared_infrastructure": ["shared eval harness"],
        "priority_order": [{"hypothesis_id": hypothesis_id, "rank": 1, "justification": "the only plan"}],
    })


def test_run_with_hypothesis_ids_plans_only_the_named_subset(tmp_path):
    fake_model = FakeChatModel({
        '"id": "H2"': _plan_response("H2"),
        "independently drafted": _single_plan_cross_cutting_response("H2"),
    })
    agent = ExperimentPlannerAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(_hypothesis_output_fixture(), hypothesis_ids=["H2"])

    # H1/H3 never reached the fan-out: FakeChatModel would have raised on an
    # unconfigured prompt if they had.
    assert [p["hypothesis_id"] for p in result["experiment_plans"]] == ["H2"]
    assert [e["hypothesis_id"] for e in result["priority_order"]] == ["H2"]
    assert result["source_hypothesis_ids"] == ["H2"]


def test_run_without_hypothesis_ids_still_plans_all_three(tmp_path):
    fake_model = FakeChatModel({
        '"id": "H1"': _plan_response("H1"),
        '"id": "H2"': _plan_response("H2"),
        '"id": "H3"': _plan_response("H3"),
        "independently drafted": _cross_cutting_response(),
    })
    agent = ExperimentPlannerAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(_hypothesis_output_fixture())

    assert [p["hypothesis_id"] for p in result["experiment_plans"]] == ["H1", "H2", "H3"]


def test_run_still_requires_the_full_hypothesis_output_even_when_filtering(tmp_path):
    """The filter narrows what's planned, never what's validated — a truncated
    hypothesis set is still rejected."""
    agent = ExperimentPlannerAgent(chat_model=FakeChatModel({}), output_dir=tmp_path)
    truncated = _hypothesis_output_fixture()
    truncated["hypotheses"] = truncated["hypotheses"][:1]

    with pytest.raises(ExperimentPlannerAgentError, match="Hypothesis Agent's output schema"):
        agent.run(truncated, hypothesis_ids=["H1"])


def test_run_rejects_a_hypothesis_id_that_isnt_in_the_input(tmp_path):
    agent = ExperimentPlannerAgent(chat_model=FakeChatModel({}), output_dir=tmp_path)
    with pytest.raises(ExperimentPlannerAgentError, match=r"aren't in the input's hypotheses"):
        agent.run(_hypothesis_output_fixture(), hypothesis_ids=["H99"])


def test_run_rejects_malformed_hypothesis_input(tmp_path):
    agent = ExperimentPlannerAgent(chat_model=FakeChatModel({}), output_dir=tmp_path)
    bad_input = {"hypotheses": "not a list"}  # doesn't match HypothesisAgentOutput at all
    with pytest.raises(ExperimentPlannerAgentError, match="Hypothesis Agent's output schema"):
        agent.run(bad_input)


def test_run_overrides_model_echoed_hypothesis_id(tmp_path):
    # model returns the wrong id in its own JSON; the agent must not trust it
    fake_model = FakeChatModel({
        '"id": "H1"': json.dumps({**_valid_plan("WRONG_ID")}),
        '"id": "H2"': _plan_response("H2"),
        '"id": "H3"': _plan_response("H3"),
        "independently drafted": _cross_cutting_response(),
    })
    agent = ExperimentPlannerAgent(chat_model=fake_model, output_dir=tmp_path)
    result = agent.run(_hypothesis_output_fixture())
    assert result["experiment_plans"][0]["hypothesis_id"] == "H1"


def test_run_raises_and_dumps_debug_file_on_schema_failure(tmp_path):
    # A malformed *plan* (missing "objective") is unrelated to priority_order
    # and has no repair/coercion path — this must still raise and dump the
    # invalid file, same as before.
    broken_plan = _valid_plan("H1")
    del broken_plan["objective"]
    fake_model = FakeChatModel({
        '"id": "H1"': json.dumps(broken_plan),
        '"id": "H2"': _plan_response("H2"),
        '"id": "H3"': _plan_response("H3"),
        "independently drafted": _cross_cutting_response(),
    })
    agent = ExperimentPlannerAgent(chat_model=fake_model, output_dir=tmp_path)

    with pytest.raises(ExperimentPlannerAgentError, match="schema validation"):
        agent.run(_hypothesis_output_fixture())

    assert list(tmp_path.glob("experiment_plan_*_invalid.json"))


def _mismatched_cross_cutting_response() -> str:
    # Ranks a hypothesis (H99) that was never in the plans shown to this
    # call — the exact 2026-08-12 production failure (schema.py rejected a
    # priority_order referencing H2 when only H1 had been planned).
    return json.dumps({
        "shared_infrastructure": ["shared eval harness"],
        "priority_order": [
            {"hypothesis_id": "H1", "rank": 1, "justification": "highest info gain"},
            {"hypothesis_id": "H2", "rank": 2, "justification": "cheap"},
            {"hypothesis_id": "H99", "rank": 3, "justification": "hallucinated hypothesis"},
        ],
    })


def test_run_recovers_priority_order_via_repair_prompt(tmp_path):
    fake_model = FakeChatModel({
        '"id": "H1"': _plan_response("H1"),
        '"id": "H2"': _plan_response("H2"),
        '"id": "H3"': _plan_response("H3"),
        "independently drafted": _mismatched_cross_cutting_response(),
        # Matches CROSS_CUTTING_REPAIR_PROMPT's fixed wording — the retry call.
        "did not match the hypotheses actually planned": _cross_cutting_response(),
    })
    agent = ExperimentPlannerAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(_hypothesis_output_fixture())  # must not raise

    assert [e["hypothesis_id"] for e in result["priority_order"]] == ["H1", "H2", "H3"]
    assert [e["rank"] for e in result["priority_order"]] == [1, 2, 3]


def test_run_deterministically_coerces_priority_order_when_repair_also_fails(tmp_path):
    # The model repeats the same broken priority_order on the repair turn too
    # (an empty list, missing every hypothesis) — the run must still complete
    # rather than crash the whole orchestrator, per the 2026-08-12 incident.
    broken = json.dumps({"shared_infrastructure": [], "priority_order": []})
    fake_model = FakeChatModel({
        '"id": "H1"': _plan_response("H1"),
        '"id": "H2"': _plan_response("H2"),
        '"id": "H3"': _plan_response("H3"),
        "independently drafted": broken,
        "did not match the hypotheses actually planned": broken,
    })
    agent = ExperimentPlannerAgent(chat_model=fake_model, output_dir=tmp_path)

    result = agent.run(_hypothesis_output_fixture())  # must not raise

    assert sorted(e["hypothesis_id"] for e in result["priority_order"]) == ["H1", "H2", "H3"]
    assert sorted(e["rank"] for e in result["priority_order"]) == [1, 2, 3]
    assert not list(tmp_path.glob("experiment_plan_*_invalid.json"))
