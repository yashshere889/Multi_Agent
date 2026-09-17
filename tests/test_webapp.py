import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import get_args
from urllib.parse import unquote

import pytest

from research_pipeline.orchestrator.nodes import finalize_node
from research_pipeline.orchestrator.state import EndStage
from research_pipeline.webapp import app as app_module
from research_pipeline.webapp import events, runs, stages

fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="needs `uv sync --extra webapp`")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def store(tmp_path):
    return runs.RunStore(tmp_path / "runs")


@pytest.fixture
def launched(monkeypatch):
    """Records launches instead of spawning a real pipeline subprocess. Uses
    the test process's own pid, because the real launch() always writes a live
    pid together with the RUNNING status — a RUNNING run with a dead pid is
    exactly what RunStore._reconcile is supposed to convert into a failure."""
    calls = []

    def fake_launch(self, run_id):
        calls.append(run_id)
        # Mirrors every field the real launch() writes, including the cleared
        # terminal ones — a fake that drifts from it hides exactly the bugs
        # these tests exist to catch.
        return self.update(
            run_id,
            status=runs.RUNNING,
            pid=os.getpid(),
            started_at=events.utc_now(),
            error=None,
            finished_at=None,
        )

    monkeypatch.setattr(runs.RunStore, "launch", fake_launch)
    return calls


def _mark_running(store, run_id):
    return store.update(run_id, status=runs.RUNNING, pid=os.getpid())


@pytest.fixture
def client(store, launched):
    return TestClient(app_module.create_app(store))


def _seed_stage_events(store, run_id, *stage_summaries):
    log = events.EventLog(store.run_dir(run_id))
    log.append(events.RUN_STARTED)
    for stage, summary in stage_summaries:
        log.append(events.STAGE_COMPLETED, stage=stage, summary=summary)
    return log


# -- starting runs ---------------------------------------------------------------------


def test_post_runs_creates_a_run_directory_and_launches_it(client, store, launched):
    response = client.post(
        "/runs",
        # What the untouched form actually submits: the default start mode, the
        # default end stage, and a checked interdisciplinary box.
        data={
            "question": "  does retrieval reduce hallucination?  ",
            "max_results": 3,
            "max_iterations": 2,
            "start_mode": "search",
            "end_stage": "writer_reviewer",
            "include_interdisciplinary": "1",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    run_id = response.headers["location"].rsplit("/", 1)[-1]
    assert launched == [run_id]

    record = store.get(run_id)
    assert record["question"] == "does retrieval reduce hallucination?"  # stripped
    assert record["params"] == {
        "max_results_per_query": 3,
        "max_iterations": 2,
        "quality_threshold": None,
        "end_stage": "writer_reviewer",
        "include_interdisciplinary": True,
    }
    # A searched run stays uncustomized: runner.py reads a *present* key as
    # "the user chose this", so the seeded-papers keys must be absent, not null.
    assert "start_stage" not in record["params"]
    assert "seed_papers" not in record["params"]
    assert record["status"] == runs.RUNNING
    for sub in ("outputs", "papers", "experiments"):
        assert (store.run_dir(run_id) / sub).is_dir()


def test_post_runs_accepts_papers_the_user_already_has(client, store, launched):
    papers = [
        {"title": "Retrieval-Augmented Generation", "authors": ["Lewis, P."], "year": 2020, "doi": "10.1/rag"},
        {"title": "Dense Passage Retrieval", "abstract": "..."},
    ]

    response = client.post(
        "/runs",
        data={
            "question": "does retrieval reduce hallucination?",
            "start_mode": "own_papers",
            "seed_papers": json.dumps(papers),
            "end_stage": "hypothesis",
            "include_interdisciplinary": "1",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    run_id = response.headers["location"].rsplit("/", 1)[-1]
    assert launched == [run_id]

    params = store.get(run_id)["params"]
    assert params["start_stage"] == "own_papers"
    assert params["seed_papers"] == papers


@pytest.mark.parametrize(
    ("seed_papers", "expected"),
    [
        ("[{'title': 'not json'}]", "not valid JSON"),
        ("", "Paste your papers as JSON"),
        ("[]", "non-empty JSON array"),
        ('{"title": "an object, not an array"}', "non-empty JSON array"),
        ('["just a string"]', "must be a JSON object"),
        ('[{"year": 2020}, {"title": "   "}]', "needs a non-empty title"),
    ],
)
def test_post_runs_rejects_unusable_seed_papers(client, store, launched, seed_papers, expected):
    """Refused before store.create, so a bad paste leaves nothing behind — the
    title check included, since paper_seed drops title-less papers silently and
    a run seeded with none of them would fail several stages later."""
    response = client.post(
        "/runs",
        data={"question": "q", "start_mode": "own_papers", "seed_papers": seed_papers},
    )

    assert response.status_code == 200
    assert expected in response.text
    assert launched == []
    assert store.list_runs() == []


def test_post_runs_records_a_custom_end_stage(client, store):
    response = client.post(
        "/runs",
        data={"question": "q", "end_stage": "coder"},
        follow_redirects=False,
    )

    run_id = response.headers["location"].rsplit("/", 1)[-1]
    assert store.get(run_id)["params"]["end_stage"] == "coder"


def test_post_runs_rejects_an_end_stage_the_pipeline_has_no_such_stage_for(client, store, launched):
    response = client.post("/runs", data={"question": "q", "end_stage": "publish_to_nature"})

    assert response.status_code == 200
    assert "not a stage this pipeline can stop at" in response.text
    assert launched == []
    assert store.list_runs() == []


@pytest.mark.parametrize(("checkbox", "expected"), [({"include_interdisciplinary": "1"}, True), ({}, False)])
def test_post_runs_reads_the_interdisciplinary_checkbox(client, store, checkbox, expected):
    """HTML sends a checkbox field only when it is checked, so an absent field
    is the user unticking it — recorded explicitly either way."""
    response = client.post("/runs", data={"question": "q", **checkbox}, follow_redirects=False)

    run_id = response.headers["location"].rsplit("/", 1)[-1]
    assert store.get(run_id)["params"]["include_interdisciplinary"] is expected


def test_index_offers_every_end_stage_the_orchestrator_accepts(client):
    body = client.get("/").text

    for stage in get_args(EndStage):
        assert f'value="{stage}"' in body
    assert 'value="writer_reviewer" selected' in body


def test_post_runs_rejects_a_blank_question(client, store, launched):
    response = client.post("/runs", data={"question": "   "})

    assert response.status_code == 200
    assert "A research question is required" in response.text
    assert launched == []
    assert store.list_runs() == []


def test_post_runs_refuses_to_exceed_the_concurrency_cap(client, store, launched, monkeypatch):
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, webapp_max_concurrent_runs=1))
    client.post("/runs", data={"question": "first"}, follow_redirects=False)

    response = client.post("/runs", data={"question": "second"})

    assert response.status_code == 200
    assert "already in progress" in response.text
    assert len(launched) == 1
    assert len(store.list_runs()) == 1


# -- watching runs ---------------------------------------------------------------------


def test_progress_fragment_renders_completed_stages_and_keeps_polling(client, store):
    record = store.create("q", {})
    _mark_running(store, record["run_id"])
    _seed_stage_events(
        store,
        record["run_id"],
        ("literature", stages.summarize("literature", {"literature_output": {"merged_papers": [{"title": "RAG Paper", "local_path": "a.pdf"}], "search_queries": ["rag"]}})),
    )

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert "Literature" in body
    assert "1 unique paper(s), 1 downloaded" in body
    assert "RAG Paper" in body
    assert "Hypothesis" in body  # still-pending stages are shown too
    assert f'data-poll-url="/runs/{record["run_id"]}/progress"' in body
    assert "Cancel" in body


def test_progress_fragment_stops_polling_once_the_run_is_terminal(client, store):
    record = store.create("q", {})
    store.update(record["run_id"], status=runs.COMPLETED, final_result={"converged": True, "iterations_run": 2, "unresolved_issues": []})

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert "data-poll-url" not in body
    assert "Passed review" in body
    assert "Cancel" not in body


def test_progress_fragment_shows_unresolved_issues_when_review_never_passed(client, store):
    record = store.create("q", {})
    store.update(
        record["run_id"],
        status=runs.COMPLETED,
        final_result={"converged": False, "iterations_run": 3, "unresolved_issues": ["Results > H2 overstates accuracy"]},
    )

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert "Finished without passing review" in body
    assert "Results &gt; H2 overstates accuracy" in body


def test_progress_fragment_summarizes_a_run_that_stopped_at_a_custom_end_stage(client, store):
    """A partial run's final_result has no Writer/Reviewer fields at all, so the
    full-run wording would render as 'after  iteration(s)'. Built by the real
    finalize_node so the template is tested against the shape it will meet."""
    final_result = finalize_node(
        {
            "literature_output": {"merged_papers": [{"title": "RAG Paper"}]},
            "hypothesis_output": {"hypotheses": [{"id": "H1"}]},
        }
    )["final_result"]
    record = store.create("q", {"end_stage": "hypothesis"})
    store.update(record["run_id"], status=runs.COMPLETED, final_result=final_result)

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert "Stopped after: Hypothesis" in body
    assert "Stages completed: Literature → Hypothesis" in body
    assert "iteration(s)" not in body
    assert "Passed review" not in body and "Finished without passing review" not in body


# -- continuing a stopped run ----------------------------------------------------------


def _stopped_after_hypothesis(store, **params):
    """A completed run that stopped at a custom end_stage, with its final_result
    built by the real finalize_node so the route meets the shape it will meet in
    production."""
    final_result = finalize_node(
        {
            "literature_output": {"merged_papers": [{"title": "RAG Paper"}]},
            "hypothesis_output": {"hypotheses": [{"id": "H1"}], "selected_hypothesis_id": "H1"},
        }
    )["final_result"]
    record = store.create(
        "does retrieval reduce hallucination?",
        {"end_stage": "hypothesis", "max_results_per_query": 3, "max_iterations": 2, **params},
    )
    return store.update(record["run_id"], status=runs.COMPLETED, final_result=final_result)


def test_continue_starts_a_new_run_seeded_with_the_stopped_run_s_output(client, store, launched):
    record = _stopped_after_hypothesis(store)

    response = client.post(
        f"/runs/{record['run_id']}/continue", data={"end_stage": "coder"}, follow_redirects=False
    )

    assert response.status_code == 303
    new_id = response.headers["location"].rsplit("/", 1)[-1]
    assert new_id != record["run_id"]
    assert launched == [new_id]

    new_record = store.get(new_id)
    assert new_record["question"] == record["question"]
    params = new_record["params"]
    assert params["resume_from"] == "hypothesis"  # the stage the last output key came from
    assert params["resumed_from_run_id"] == record["run_id"]
    assert params["end_stage"] == "coder"
    # every finished stage travels with it, or the run would enter the planner
    # with nothing to plan
    assert params["hypothesis_output"] == {"hypotheses": [{"id": "H1"}], "selected_hypothesis_id": "H1"}
    assert params["literature_output"] == {"merged_papers": [{"title": "RAG Paper"}]}
    # the settings the user picked for this question are kept, not re-asked
    assert params["max_results_per_query"] == 3
    assert params["max_iterations"] == 2
    # the cross-field stage is behind this resume point, so it is not this
    # form's to answer
    assert "include_interdisciplinary" not in params
    # and the run being continued is left exactly as it was
    assert store.get(record["run_id"])["status"] == runs.COMPLETED
    assert store.get(record["run_id"])["final_result"] == record["final_result"]


@pytest.mark.parametrize(("checkbox", "expected"), [({"include_interdisciplinary": "1"}, True), ({}, False)])
def test_continue_from_the_literature_stage_asks_about_the_cross_field_stage(
    client, store, checkbox, expected
):
    """The one resume point with the cross-field stage still ahead of it."""
    final_result = finalize_node({"literature_output": {"merged_papers": [{"title": "RAG Paper"}]}})["final_result"]
    record = store.create("q", {"end_stage": "literature"})
    store.update(record["run_id"], status=runs.COMPLETED, final_result=final_result)

    response = client.post(
        f"/runs/{record['run_id']}/continue",
        data={"end_stage": "hypothesis", **checkbox},
        follow_redirects=False,
    )

    new_id = response.headers["location"].rsplit("/", 1)[-1]
    params = store.get(new_id)["params"]
    assert params["resume_from"] == "literature"
    assert params["include_interdisciplinary"] is expected


def test_continue_refuses_a_run_with_nothing_to_continue_from(client, store, launched):
    """A converged full run has no stages_completed bundle — only the partial
    branch of finalize_node writes one."""
    record = store.create("q", {})
    store.update(
        record["run_id"],
        status=runs.COMPLETED,
        final_result={"converged": True, "iterations_run": 1, "unresolved_issues": []},
    )

    response = client.post(f"/runs/{record['run_id']}/continue", data={"end_stage": "coder"})

    assert response.status_code == 200
    assert "no stopped-at stage to continue from" in response.text
    assert launched == []
    assert len(store.list_runs()) == 1  # nothing new was created


def test_continue_refuses_a_run_that_has_not_finished(client, store, launched):
    record = store.create("q", {"end_stage": "hypothesis"})
    _mark_running(store, record["run_id"])

    response = client.post(f"/runs/{record['run_id']}/continue", data={"end_stage": "coder"})

    assert response.status_code == 200
    assert "no stopped-at stage to continue from" in response.text
    assert launched == []


@pytest.mark.parametrize("end_stage", ["literature", "hypothesis", "publish_to_nature"])
def test_continue_refuses_an_end_stage_that_is_not_ahead_of_the_resume_point(
    client, store, launched, end_stage
):
    """At or before the resume point there is nothing left to run — the graph
    would route from its entry point straight to finalize."""
    record = _stopped_after_hypothesis(store)

    response = client.post(f"/runs/{record['run_id']}/continue", data={"end_stage": end_stage})

    assert response.status_code == 200
    assert "not a stage it can continue into" in response.text
    assert launched == []
    assert len(store.list_runs()) == 1


def test_continue_refuses_to_exceed_the_concurrency_cap(client, store, launched, monkeypatch):
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, webapp_max_concurrent_runs=1))
    record = _stopped_after_hypothesis(store)
    other = store.create("something else", {})
    _mark_running(store, other["run_id"])

    response = client.post(f"/runs/{record['run_id']}/continue", data={"end_stage": "coder"})

    assert response.status_code == 200
    assert "already in progress" in response.text
    assert launched == []


def test_progress_fragment_offers_to_continue_a_stopped_run(client, store):
    record = _stopped_after_hypothesis(store)

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert f'action="/runs/{record["run_id"]}/continue"' in body
    # only the stages still ahead of the resume point are offered
    for ahead in ("experiment_planner", "coder", "writer_reviewer"):
        assert f'value="{ahead}"' in body
    for behind in ("literature", "interdisciplinary_literature", "hypothesis"):
        assert f'value="{behind}"' not in body
    # the cross-field stage is already behind this run, so it is not asked about
    assert "include_interdisciplinary" not in body


def test_progress_fragment_does_not_offer_to_continue_a_full_run(client, store):
    record = store.create("q", {})
    store.update(
        record["run_id"],
        status=runs.COMPLETED,
        final_result={"converged": True, "iterations_run": 1, "unresolved_issues": []},
    )

    assert "/continue" not in client.get(f"/runs/{record['run_id']}/progress").text


def test_progress_fragment_does_not_offer_to_continue_a_run_still_in_flight(client, store):
    """Its stopping point is not settled yet — and the form would be offering to
    resume from a bundle that does not exist."""
    record = _stopped_after_hypothesis(store)
    _mark_running(store, record["run_id"])

    assert "/continue" not in client.get(f"/runs/{record['run_id']}/progress").text


def test_progress_fragment_labels_carried_over_stages_and_links_where_they_came_from(client, store):
    origin = _stopped_after_hypothesis(store)
    record = store.create(
        "q", {"end_stage": "coder", "resume_from": "hypothesis", "resumed_from_run_id": origin["run_id"]}
    )
    _mark_running(store, record["run_id"])
    log = events.EventLog(store.run_dir(record["run_id"]))
    log.append(
        events.STAGE_COMPLETED,
        stage="hypothesis",
        summary=stages.summarize("hypothesis", {"hypothesis_output": {"hypotheses": [{"id": "H1"}]}}),
        carried_over=True,
    )

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert "from a previous run" in body
    assert f'href="/runs/{origin["run_id"]}"' in body
    assert "Continued from" in body


def test_log_fragment_shows_the_tail_of_the_log(client, store, monkeypatch):
    monkeypatch.setattr(app_module, "LOG_TAIL", 2)
    record = store.create("q", {})
    _mark_running(store, record["run_id"])
    log = events.EventLog(store.run_dir(record["run_id"]))
    for index in range(4):
        log.log("INFO", "research_pipeline.agents.coder", f"line {index}")

    body = client.get(f"/runs/{record['run_id']}/log").text

    assert "line 3" in body and "line 2" in body
    assert "line 0" not in body
    assert "agents.coder" in body  # the research_pipeline. prefix is trimmed for width


def test_api_run_status_reports_stage_progress(client, store):
    record = store.create("q", {})
    _mark_running(store, record["run_id"])
    _seed_stage_events(store, record["run_id"], ("literature", {"papers_found": 2}), ("hypothesis", {"gaps": 1}))

    payload = client.get(f"/api/runs/{record['run_id']}").json()

    assert payload["active"] is True
    assert payload["stages_completed"] == ["literature", "hypothesis"]
    assert [s["status"] for s in payload["stages"][:3]] == ["done", "done", "running"]


def test_index_lists_runs_newest_first(client, store):
    older = store.create("older question", {})
    newer = store.create("newer question", {})
    store.update(older["run_id"], created_at="2026-01-01T00:00:00+00:00")
    store.update(newer["run_id"], created_at="2026-06-01T00:00:00+00:00")

    body = client.get("/").text

    assert body.index("newer question") < body.index("older question")


# -- cancelling ------------------------------------------------------------------------


def test_cancel_sigterms_the_runner_and_marks_the_run_cancelled(client, store, monkeypatch):
    record = store.create("q", {})
    _mark_running(store, record["run_id"])
    # Intercepted rather than delivered: the fixture uses this process's own
    # pid, so a real SIGTERM here would kill the test session.
    signalled = []
    monkeypatch.setattr(runs.os, "kill", lambda pid, sig: signalled.append((pid, sig)))

    response = client.post(f"/runs/{record['run_id']}/cancel", follow_redirects=False)

    assert response.status_code == 303
    # Liveness probes also go through os.kill (with signal 0), so look for the
    # SIGTERM among them rather than asserting on the whole call list.
    assert (os.getpid(), signal.SIGTERM) in signalled
    assert store.get(record["run_id"])["status"] == runs.CANCELLED


def test_cancel_still_records_a_run_whose_process_is_already_gone(client, store):
    record = store.create("q", {})
    store.update(record["run_id"], status=runs.RUNNING, pid=999_999)

    client.post(f"/runs/{record['run_id']}/cancel")

    # _reconcile gets there first and calls it failed; either way the run is
    # terminal and the UI stops polling it.
    assert store.get(record["run_id"])["status"] in runs.TERMINAL_STATUSES


def test_cancel_leaves_a_finished_run_alone(client, store):
    record = store.create("q", {})
    store.update(record["run_id"], status=runs.COMPLETED)

    client.post(f"/runs/{record['run_id']}/cancel")

    assert store.get(record["run_id"])["status"] == runs.COMPLETED


# -- resuming --------------------------------------------------------------------------


@pytest.fixture
def durable(monkeypatch):
    """Resume is only offered when checkpoints outlive the process that wrote
    them, which the default in-memory backend does not."""
    monkeypatch.setattr(app_module.checkpointer, "is_durable", lambda: True)


def _stopped(store, status=runs.FAILED):
    record = store.create("q", {})
    return store.update(record["run_id"], status=status, error="died", finished_at=events.utc_now())


def test_resume_relaunches_a_failed_run(client, store, launched, durable):
    record = _stopped(store)

    response = client.post(f"/runs/{record['run_id']}/resume", follow_redirects=False)

    assert response.status_code == 303
    assert launched == [record["run_id"]]
    assert store.get(record["run_id"])["status"] == runs.RUNNING


def test_resume_clears_the_failure_it_was_resumed_from(client, store, launched, durable):
    """Otherwise the run shows a running badge next to the error that stopped
    its previous attempt."""
    record = _stopped(store)

    client.post(f"/runs/{record['run_id']}/resume")

    resumed = store.get(record["run_id"])
    assert resumed["error"] is None
    assert resumed["finished_at"] is None


def test_real_launch_clears_the_previous_attempts_terminal_fields(store, monkeypatch):
    """The same assertion as above against the real launch(), since every other
    test in this module replaces it with a fake."""
    record = _stopped(store)
    monkeypatch.setattr(runs.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=os.getpid(), poll=lambda: None))

    launched_record = store.launch(record["run_id"])

    assert launched_record["status"] == runs.RUNNING
    assert launched_record["error"] is None
    assert launched_record["finished_at"] is None


def test_resume_works_for_a_cancelled_run_too(client, store, launched, durable):
    record = _stopped(store, status=runs.CANCELLED)

    client.post(f"/runs/{record['run_id']}/resume")

    assert launched == [record["run_id"]]


def test_resume_refuses_a_completed_run(client, store, launched, durable):
    record = store.create("q", {})
    store.update(record["run_id"], status=runs.COMPLETED)

    response = client.post(f"/runs/{record['run_id']}/resume", follow_redirects=False)

    assert launched == []
    assert "cannot be resumed" in unquote(response.headers["location"])
    assert store.get(record["run_id"])["status"] == runs.COMPLETED


def test_resume_refuses_a_run_that_is_already_going(client, store, launched, durable):
    """The guard that matters: a second runner on one run directory would
    interleave its events and fight over the same thread_id's checkpoints."""
    record = store.create("q", {})
    _mark_running(store, record["run_id"])

    response = client.post(f"/runs/{record['run_id']}/resume", follow_redirects=False)

    assert launched == []
    assert "cannot be resumed" in unquote(response.headers["location"])


def test_resume_respects_the_concurrency_limit(client, store, launched, durable, monkeypatch):
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, webapp_max_concurrent_runs=1))
    busy = store.create("busy", {})
    _mark_running(store, busy["run_id"])
    record = _stopped(store)

    response = client.post(f"/runs/{record['run_id']}/resume", follow_redirects=False)

    assert launched == []
    assert "Too many runs" in unquote(response.headers["location"])


def test_resume_button_is_offered_for_a_stopped_run(client, store, durable):
    record = _stopped(store)

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert f"/runs/{record['run_id']}/resume" in body


def test_no_resume_button_without_a_durable_checkpointer(client, store):
    """Under the default backend there is nothing to resume to, and relaunching
    would quietly redo the whole pipeline behind a button labelled Resume."""
    record = _stopped(store)

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert "/resume" not in body


def test_no_resume_button_while_a_run_is_still_going(client, store, durable):
    record = store.create("q", {})
    _mark_running(store, record["run_id"])

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert "/resume" not in body
    assert "/cancel" in body


def test_a_refused_resume_is_shown_on_the_run_page(client, store, durable):
    record = _stopped(store)

    body = client.get(f"/runs/{record['run_id']}?error=Nope+not+that").text

    assert "Nope not that" in body


# -- serving papers, and refusing to serve anything else -------------------------------


def test_paper_serves_the_latest_draft_while_a_run_is_still_going(client, store):
    record = store.create("q", {})
    outputs = store.run_dir(record["run_id"]) / "outputs"
    (outputs / "v1.pdf").write_bytes(b"%PDF-1.4 first")
    (outputs / "v2.pdf").write_bytes(b"%PDF-1.4 second")

    response = client.get(f"/runs/{record['run_id']}/paper")

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 second"
    assert response.headers["content-type"] == "application/pdf"


def test_paper_serves_a_named_iteration(client, store):
    record = store.create("q", {})
    (store.run_dir(record["run_id"]) / "outputs" / "v1.pdf").write_bytes(b"%PDF-1.4 first")

    response = client.get(f"/runs/{record['run_id']}/paper", params={"file": "v1.pdf"})

    assert response.content == b"%PDF-1.4 first"


def test_paper_refuses_to_escape_the_run_directory(client, store, tmp_path):
    record = store.create("q", {})
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(b"%PDF-1.4 secret")

    for attempt in ("../../secret.pdf", "/etc/passwd", str(secret)):
        response = client.get(f"/runs/{record['run_id']}/paper", params={"file": attempt})
        assert response.status_code == 404, attempt


def test_paper_refuses_a_non_pdf_inside_the_run_directory(client, store):
    record = store.create("q", {})
    (store.run_dir(record["run_id"]) / "outputs" / "v1_summary.json").write_text("{}")

    response = client.get(f"/runs/{record['run_id']}/paper", params={"file": "v1_summary.json"})

    assert response.status_code == 404


@pytest.mark.parametrize("run_id", ["not-a-uuid", "../../etc", "00000000-0000-0000-0000-000000000000"])
def test_unknown_or_malformed_run_ids_are_404(client, run_id):
    for path in (f"/runs/{run_id}", f"/runs/{run_id}/progress", f"/runs/{run_id}/log", f"/api/runs/{run_id}"):
        assert client.get(path).status_code == 404, path


@pytest.mark.parametrize("run_id", ["..", "../secrets", "", "a/b", "00000000-0000-0000-0000-00000000000"])
def test_run_dir_refuses_anything_that_is_not_a_uuid(store, run_id):
    """The guard is tested here rather than through a request because HTTP
    clients normalize `..` out of the path before it ever reaches a route."""
    with pytest.raises(runs.RunNotFound):
        store.run_dir(run_id)


# -- run state recovery ----------------------------------------------------------------


def test_a_run_whose_process_vanished_is_reported_as_failed(store):
    record = store.create("q", {})
    # A pid that is almost certainly not running, standing in for a runner that
    # was SIGKILLed or whose node was pre-empted.
    store.update(record["run_id"], status=runs.RUNNING, pid=999_999)

    recovered = store.get(record["run_id"])

    assert recovered["status"] == runs.FAILED
    assert "without reporting a result" in recovered["error"]


def test_a_runner_that_exited_is_reaped_rather_than_read_as_a_live_zombie(store):
    """An unwaited-for child becomes a zombie, and os.kill(pid, 0) succeeds for
    a zombie — so without reaping, every finished runner would read as alive
    and no run would ever be reconciled."""
    record = store.create("q", {})
    child = subprocess.Popen([sys.executable, "-c", ""])
    store._children[record["run_id"]] = child
    store.update(record["run_id"], status=runs.RUNNING, pid=child.pid)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and store.get(record["run_id"])["status"] == runs.RUNNING:
        time.sleep(0.05)

    recovered = store.get(record["run_id"])
    assert recovered["status"] == runs.FAILED
    assert "without reporting a result" in recovered["error"]
    assert record["run_id"] not in store._children


def test_a_terminal_event_wins_over_a_stale_run_json(store):
    record = store.create("q", {})
    store.update(record["run_id"], status=runs.RUNNING, pid=999_999)
    log = events.EventLog(store.run_dir(record["run_id"]))
    log.append(events.RUN_COMPLETED, final_result={"converged": True, "iterations_run": 1})

    recovered = store.get(record["run_id"])

    assert recovered["status"] == runs.COMPLETED
    assert recovered["final_result"]["converged"] is True


def test_a_malformed_event_line_does_not_break_reading_the_rest(store):
    record = store.create("q", {})
    run_dir = store.run_dir(record["run_id"])
    log = events.EventLog(run_dir)
    log.append(events.STAGE_COMPLETED, stage="literature", summary={})
    with open(run_dir / events.EVENTS_NAME, "a") as handle:
        handle.write('{"seq": 1, "type": "stage_com\n')  # a torn write
    log2 = events.EventLog(run_dir)
    log2.append(events.STAGE_COMPLETED, stage="hypothesis", summary={})

    stage_events = events.read_events(run_dir, types=[events.STAGE_COMPLETED])

    assert [e["stage"] for e in stage_events] == ["literature", "hypothesis"]


def test_run_json_survives_a_server_restart(store, tmp_path):
    record = store.create("q", {"max_results_per_query": 4})
    store.update(record["run_id"], status=runs.COMPLETED)

    reopened = runs.RunStore(tmp_path / "runs")

    assert reopened.get(record["run_id"])["params"]["max_results_per_query"] == 4
    assert json.loads((store.run_dir(record["run_id"]) / "run.json").read_text())["status"] == runs.COMPLETED


# -- the run artifact browser -------------------------------------------------


def _write_artifacts(store, run_id):
    run_dir = store.run_dir(run_id)
    (run_dir / "outputs" / "hypotheses.json").write_text('{"hypotheses":[{"id":"H1"}]}')
    (run_dir / "outputs" / "v1.pdf").write_bytes(b"%PDF-1.4 fake")
    (run_dir / "experiments" / "H1").mkdir(parents=True, exist_ok=True)
    (run_dir / "experiments" / "H1" / "run.py").write_text("print('hi')\n")
    return run_dir


def test_files_page_lists_what_the_run_wrote(client, store):
    record = store.create("q", {})
    _write_artifacts(store, record["run_id"])

    body = client.get(f"/runs/{record['run_id']}/files").text

    assert "outputs/hypotheses.json" in body
    assert "experiments/H1/run.py" in body
    assert "Stage outputs and drafts" in body
    assert "Generated experiments" in body


def test_run_page_links_to_the_files_page(client, store):
    record = store.create("q", {})

    assert f'/runs/{record["run_id"]}/files' in client.get(f"/runs/{record['run_id']}").text


def test_viewing_a_json_file_pretty_prints_it(client, store):
    record = store.create("q", {})
    _write_artifacts(store, record["run_id"])

    body = client.get(
        f"/runs/{record['run_id']}/files/view", params={"path": "outputs/hypotheses.json"}
    ).text

    assert '&#34;id&#34;: &#34;H1&#34;' in body  # re-indented, and autoescaped


def test_viewing_a_pdf_redirects_to_the_raw_bytes(client, store):
    record = store.create("q", {})
    _write_artifacts(store, record["run_id"])

    response = client.get(
        f"/runs/{record['run_id']}/files/view", params={"path": "outputs/v1.pdf"}, follow_redirects=False
    )

    assert response.status_code == 303
    assert "/files/raw?path=outputs%2Fv1.pdf" in response.headers["location"]


def test_raw_serves_a_pdf_inline_and_everything_else_as_a_download(client, store):
    record = store.create("q", {})
    _write_artifacts(store, record["run_id"])
    run_id = record["run_id"]

    pdf = client.get(f"/runs/{run_id}/files/raw", params={"path": "outputs/v1.pdf"})
    code = client.get(f"/runs/{run_id}/files/raw", params={"path": "experiments/H1/run.py"})

    assert pdf.headers["content-type"] == "application/pdf"
    assert "attachment" not in pdf.headers.get("content-disposition", "")
    assert run_id[:8] in code.headers["content-disposition"]


@pytest.mark.parametrize("route", ["view", "raw"])
@pytest.mark.parametrize("escape", ["../../../etc/passwd", "/etc/passwd", "outputs/../../secret"])
def test_a_path_outside_the_run_directory_is_refused(client, store, route, escape):
    """resolve_inside is the boundary, not the fact that the UI only ever builds
    links to files it listed: a bookmarked URL or a curl reaches these too."""
    record = store.create("q", {})
    _write_artifacts(store, record["run_id"])

    response = client.get(f"/runs/{record['run_id']}/files/{route}", params={"path": escape})

    assert response.status_code == 404


@pytest.mark.parametrize("route", ["view", "raw"])
def test_a_path_inside_the_run_that_does_not_exist_is_a_404(client, store, route):
    record = store.create("q", {})

    response = client.get(f"/runs/{record['run_id']}/files/{route}", params={"path": "outputs/nope.json"})

    assert response.status_code == 404


def test_files_page_for_a_run_that_has_only_just_started(client, store):
    """create() writes run.json and the empty stage directories, so even a run
    that has produced nothing lists its own record rather than an empty page."""
    record = store.create("q", {})

    body = client.get(f"/runs/{record['run_id']}/files").text

    assert "Run files" in body
    assert "run.json" in body
    assert "Stage outputs and drafts" not in body


# -- the experiment inspector -------------------------------------------------


def _coder_summary(store, run_id, experiments_list):
    run_dir = store.run_dir(run_id)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs" / "coder_agent_summary_20260820T120000Z.json").write_text(
        json.dumps({"experiments": experiments_list, "generated_at": "2026-08-20T12:00:00+00:00"})
    )
    return run_dir


def test_run_page_links_to_the_experiments_page(client, store):
    record = store.create("q", {})

    assert f'/runs/{record["run_id"]}/experiments' in client.get(f"/runs/{record['run_id']}").text


def test_experiments_page_renders_metrics_fixes_and_files(client, store):
    record = store.create("q", {})
    run_id = record["run_id"]
    run_dir = store.run_dir(run_id)
    code_dir = run_dir / "experiments" / "H1"
    (code_dir / "fix_attempts" / "attempt_1").mkdir(parents=True)
    (code_dir / "run.py").write_text("def main(): pass\n")
    _coder_summary(
        store,
        run_id,
        [
            {
                "hypothesis_id": "H1",
                "status": "completed",
                "code_path": str(code_dir),
                "results": {"metrics": {"f1": 0.71}, "meets_success_criteria": True, "notes": ""},
                "fix_attempts": 1,
                "fix_history": [
                    {
                        "attempt": 1,
                        "error_source": "missing_data_fallback",
                        "error_summary": "load_data reads reviews.csv unguarded",
                        "code_path": str(code_dir / "fix_attempts" / "attempt_1"),
                        "resolved": True,
                        "same_error_streak": 1,
                    }
                ],
                "assumptions_made": ["treated missing labels as negatives"],
                "data_provenance": {},
            }
        ],
    )

    body = client.get(f"/runs/{run_id}/experiments").text

    assert "H1" in body
    assert "0.71" in body
    assert "missing_data_fallback" in body
    assert "load_data reads reviews.csv unguarded" in body
    assert "treated missing labels as negatives" in body
    assert "experiments/H1/run.py" in body


def test_experiments_page_says_loudly_when_a_verdict_was_withheld(client, store):
    record = store.create("q", {})
    run_id = record["run_id"]
    _coder_summary(
        store,
        run_id,
        [
            {
                "hypothesis_id": "H1",
                "status": "completed",
                "code_path": "",
                "results": {
                    "metrics": {"rmse": 2.1},
                    "meets_success_criteria": "unknown",
                    "model_reported_meets_success_criteria": False,
                    "verdict_withheld_because": "One or more inputs are synthetic surrogates",
                    "notes": "",
                },
                "fix_attempts": 0,
                "fix_history": [],
                "data_provenance": {
                    "inputs": [{"name": "CMS claims", "kind": "synthetic_surrogate", "reason": "needs a DUA"}],
                    "surrogate_count": 1,
                },
            }
        ],
    )

    body = client.get(f"/runs/{run_id}/experiments").text

    assert "Verdict withheld" in body
    assert "synthetic surrogates" in body
    assert "synthetic_surrogate" in body
    assert "needs a DUA" in body
    # the metrics still stand, only the verdict is withheld
    assert "2.1" in body


def test_experiments_page_for_a_run_that_never_reached_the_coder(client, store):
    record = store.create("q", {"end_stage": "hypothesis"})

    body = client.get(f"/runs/{record['run_id']}/experiments").text

    assert "No experiments to inspect" in body


# -- re-running one experiment ------------------------------------------------


def _plan(hypothesis_id):
    return {
        "hypothesis_id": hypothesis_id,
        "feasible": True,
        "feasibility_notes": "n",
        "objective": "o",
        "variables": {"independent": ["x"], "dependent": ["y"]},
        "design": "d",
        "data_requirements": {"source": "s", "description": "d", "preprocessing_steps": []},
        "methods": [{"name": "m", "description": "d", "reused_from_literature": False}],
        "evaluation": {"metrics": ["F1"], "baseline": "b", "success_criteria": "c"},
        "implementation_steps": [{"step": 1, "description": "go"}],
        "estimated_complexity": "low",
        "risks": [],
    }


def _finished_run_with_a_plan(store, ids=("H1", "H2", "H3"), **params):
    """A run that got as far as planning and is over. Terminal on purpose: a
    freshly created run is still 'active' and would trip the concurrency cap
    before the re-run is even considered."""
    record = store.create("q", params)
    _planner_output_file(store, record["run_id"], ids=ids)
    return store.update(record["run_id"], status=runs.COMPLETED)


def _planner_output_file(store, run_id, ids=("H1", "H2", "H3"), name="experiment_plan_20260820T100000Z.json"):
    run_dir = store.run_dir(run_id)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs" / name).write_text(
        json.dumps(
            {
                "experiment_plans": [_plan(i) for i in ids],
                "shared_infrastructure": ["harness"],
                "priority_order": [
                    {"hypothesis_id": i, "rank": n, "justification": "j"} for n, i in enumerate(ids, start=1)
                ],
                "source_hypothesis_ids": list(ids),
                "generated_at": "2026-08-20T10:00:00+00:00",
                "model": "test-model",
            }
        )
    )


def test_rerun_starts_a_new_run_seeded_with_just_that_plan(client, store, launched):
    record = _finished_run_with_a_plan(store, max_results_per_query=4, quality_threshold=5)

    response = client.post(
        f"/runs/{record['run_id']}/experiments/H2/rerun", follow_redirects=False
    )

    assert response.status_code == 303
    new_id = response.headers["location"].rsplit("/", 1)[-1]
    assert new_id in launched

    params = store.get(new_id)["params"]
    # Only H2's plan travels, and it is still a valid planner output — the Coder
    # Agent validates what it is handed.
    assert [p["hypothesis_id"] for p in params["planner_output"]["experiment_plans"]] == ["H2"]
    assert params["planner_output"]["priority_order"] == [
        {"hypothesis_id": "H2", "rank": 1, "justification": "j"}
    ]
    # Enters at the Coder and stops there: nothing upstream runs again, and
    # nothing downstream can run without a hypothesis_output it wasn't given.
    assert params["resume_from"] == "experiment_planner"
    assert params["end_stage"] == "coder"
    assert params["resumed_from_run_id"] == record["run_id"]
    assert params["rerun_hypothesis_id"] == "H2"
    # and the run being re-run from is untouched
    assert store.get(record["run_id"])["params"] == {"max_results_per_query": 4, "quality_threshold": 5}


def test_the_narrowed_plan_a_rerun_seeds_is_a_valid_planner_output(client, store):
    """The check that matters: run_coder_agent validates its input, so a plan
    list filtered without re-ranking priority_order would be refused."""
    from research_pipeline.agents.experiment_planner.schema import validate_output

    record = _finished_run_with_a_plan(store)

    response = client.post(f"/runs/{record['run_id']}/experiments/H3/rerun", follow_redirects=False)
    new_id = response.headers["location"].rsplit("/", 1)[-1]

    validate_output(store.get(new_id)["params"]["planner_output"])  # should not raise


def test_rerun_refuses_a_run_with_no_experiment_plan(client, store):
    record = store.create("q", {})

    response = client.post(f"/runs/{record['run_id']}/experiments/H1/rerun")

    assert "no experiment plan on disk" in response.text


def test_rerun_refuses_a_hypothesis_the_plan_never_covered(client, store):
    record = _finished_run_with_a_plan(store, ids=("H1",))

    response = client.post(f"/runs/{record['run_id']}/experiments/H9/rerun")

    assert "no experiment plan for hypothesis id" in response.text


def test_rerun_ignores_an_invalid_plan_debug_file(client, store):
    """experiment_plan_<ts>_invalid.json is what the planner writes when its own
    validation failed — the one plan that must never be re-run, and the one the
    glob would otherwise sort last and pick."""
    record = _finished_run_with_a_plan(store, ids=("H1",))
    (store.run_dir(record["run_id"]) / "outputs" / "experiment_plan_20260820T110000Z_invalid.json").write_text("{}")

    response = client.post(f"/runs/{record['run_id']}/experiments/H1/rerun", follow_redirects=False)

    assert response.status_code == 303


def test_rerun_refuses_to_exceed_the_concurrency_cap(client, store, monkeypatch):
    record = _finished_run_with_a_plan(store)
    monkeypatch.setattr(runs.RunStore, "active_count", lambda self: 99)

    response = client.post(f"/runs/{record['run_id']}/experiments/H1/rerun")

    assert "already in progress" in response.text


def test_experiments_page_offers_a_rerun_per_experiment(client, store):
    record = store.create("q", {})
    run_id = record["run_id"]
    _coder_summary(
        store,
        run_id,
        [
            {"hypothesis_id": "H1", "status": "completed", "code_path": "", "results": None,
             "fix_attempts": 0, "fix_history": [], "data_provenance": {}},
            {"hypothesis_id": "H2", "status": "skipped", "code_path": "", "results": None,
             "fix_attempts": 0, "fix_history": [], "data_provenance": {}},
        ],
    )

    body = client.get(f"/runs/{run_id}/experiments").text

    assert f'action="/runs/{run_id}/experiments/H1/rerun"' in body
    assert f'action="/runs/{run_id}/experiments/H2/rerun"' in body


def test_a_rerun_run_says_what_it_is_re_running(client, store):
    record = store.create("q", {"resumed_from_run_id": "abc-123", "rerun_hypothesis_id": "H2"})

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert "Re-running" in body
    assert "H2" in body
    assert "/runs/abc-123" in body


# -- steering which hypothesis goes forward -----------------------------------


def _stopped_with_three_hypotheses(store, **params):
    """A run stopped at the hypothesis stage with a full ranked set, which is
    what makes continuing it a choice rather than just a range."""
    hypothesis_output = {
        "hypotheses": [
            {"id": "H1", "statement": "Retrieval helps only with sentence-level attribution.", "rationale": "r1"},
            {"id": "H2", "statement": "Bigger retrieval budgets hurt past saturation.", "rationale": "r2"},
            {"id": "H3", "statement": "Reranking matters more than recall.", "rationale": "r3"},
        ],
        "ranking": [
            {"hypothesis_id": "H1", "rank": 1, "score": 8.4, "justification": "highest information gain"},
            {"hypothesis_id": "H2", "rank": 2, "score": 7.1, "justification": "cheap to test"},
            {"hypothesis_id": "H3", "rank": 3, "score": 6.0, "justification": "narrower"},
        ],
        "selected_hypothesis_id": "H1",
    }
    final_result = finalize_node(
        {"literature_output": {"merged_papers": [{"title": "p"}]}, "hypothesis_output": hypothesis_output}
    )["final_result"]
    record = store.create("q", {"end_stage": "hypothesis", **params})
    return store.update(record["run_id"], status=runs.COMPLETED, final_result=final_result)


def test_start_form_offers_to_steer(client):
    assert 'name="steer_hypothesis"' in client.get("/").text


def test_steering_stops_the_run_at_the_hypothesis_stage_and_remembers_the_destination(client, store, launched):
    response = client.post(
        "/runs",
        data={"question": "q", "end_stage": "writer_reviewer", "steer_hypothesis": "1"},
        follow_redirects=False,
    )

    params = store.get(response.headers["location"].rsplit("/", 1)[-1])["params"]
    assert params["end_stage"] == "hypothesis"
    # Where the user actually wanted to end up, so continuing doesn't ask twice.
    assert params["steer_to_end_stage"] == "writer_reviewer"


def test_steering_a_run_that_already_stops_at_or_before_the_choice_is_a_no_op(client, store, launched):
    """The run already ends where the choice would be made — nothing to remember,
    and nothing to refuse."""
    response = client.post(
        "/runs",
        data={"question": "q", "end_stage": "literature", "steer_hypothesis": "1"},
        follow_redirects=False,
    )

    params = store.get(response.headers["location"].rsplit("/", 1)[-1])["params"]
    assert params["end_stage"] == "literature"
    assert "steer_to_end_stage" not in params


def test_a_run_stopped_at_hypothesis_shows_every_hypothesis_to_choose_from(client, store):
    record = _stopped_with_three_hypotheses(store)

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert 'name="planned_hypothesis_ids"' in body
    assert "Retrieval helps only with sentence-level attribution." in body
    assert "Bigger retrieval budgets hurt past saturation." in body
    assert "highest information gain" in body
    assert "rank 1" in body and "score 8.4" in body


def test_the_remembered_destination_is_preselected_on_the_continue_form(client, store):
    record = _stopped_with_three_hypotheses(store, steer_to_end_stage="coder")

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert '<option value="coder" selected>' in body


def test_continuing_with_a_chosen_hypothesis_carries_it_to_the_planner(client, store, launched):
    record = _stopped_with_three_hypotheses(store)

    response = client.post(
        f"/runs/{record['run_id']}/continue",
        data={"end_stage": "coder", "planned_hypothesis_ids": ["H3"]},
        follow_redirects=False,
    )

    new_id = response.headers["location"].rsplit("/", 1)[-1]
    assert store.get(new_id)["params"]["planned_hypothesis_ids"] == ["H3"]


def test_several_hypotheses_can_be_taken_forward_at_once(client, store, launched):
    record = _stopped_with_three_hypotheses(store)

    response = client.post(
        f"/runs/{record['run_id']}/continue",
        data={"end_stage": "coder", "planned_hypothesis_ids": ["H2", "H3"]},
        follow_redirects=False,
    )

    new_id = response.headers["location"].rsplit("/", 1)[-1]
    assert store.get(new_id)["params"]["planned_hypothesis_ids"] == ["H2", "H3"]


def test_choosing_nothing_leaves_the_ranking_s_own_pick_intact(client, store, launched):
    """Absent, not empty: the orchestrator reads this key with state.get, so an
    explicit empty list would still have to mean the same thing — not setting it
    at all is what makes 'unchanged' unambiguous."""
    record = _stopped_with_three_hypotheses(store)

    response = client.post(
        f"/runs/{record['run_id']}/continue", data={"end_stage": "coder"}, follow_redirects=False
    )

    new_id = response.headers["location"].rsplit("/", 1)[-1]
    assert "planned_hypothesis_ids" not in store.get(new_id)["params"]


def test_continuing_refuses_a_hypothesis_id_the_run_never_generated(client, store):
    record = _stopped_with_three_hypotheses(store)

    response = client.post(
        f"/runs/{record['run_id']}/continue",
        data={"end_stage": "coder", "planned_hypothesis_ids": ["H9"]},
    )

    assert "no hypothesis with id(s): H9" in response.text


def test_a_run_stopped_elsewhere_is_not_offered_a_hypothesis_choice(client, store):
    """Past the planner the plans already exist and are carried forward, so a
    choice there would name hypotheses nothing downstream would consult."""
    record = _stopped_after_hypothesis(store)
    store.update(
        record["run_id"],
        final_result={**record["final_result"], "stages_completed": ["literature_output"]},
    )

    body = client.get(f"/runs/{record['run_id']}/progress").text

    assert 'name="planned_hypothesis_ids"' not in body
