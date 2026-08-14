import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

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
        data={"question": "  does retrieval reduce hallucination?  ", "max_results": 3, "max_iterations": 2},
        follow_redirects=False,
    )

    assert response.status_code == 303
    run_id = response.headers["location"].rsplit("/", 1)[-1]
    assert launched == [run_id]

    record = store.get(run_id)
    assert record["question"] == "does retrieval reduce hallucination?"  # stripped
    assert record["params"] == {"max_results_per_query": 3, "max_iterations": 2, "quality_threshold": None}
    assert record["status"] == runs.RUNNING
    for sub in ("outputs", "papers", "experiments"):
        assert (store.run_dir(run_id) / sub).is_dir()


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
