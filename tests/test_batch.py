import json
from pathlib import Path

import pytest

from research_pipeline import batch


class FakeGraph:
    """Stands in for the compiled pipeline graph. Raises for any question
    listed in `fail_for`, otherwise returns a minimal final_result."""

    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.invoked = []

    def invoke(self, state, config=None):
        question = state["research_question"]
        self.invoked.append(question)
        if question in self.fail_for:
            raise RuntimeError(f"pipeline blew up on {question!r}")
        return {
            "final_result": {
                "final_paper_path": f"{state['output_dir']}/v1.pdf",
                "iterations_run": 1,
                "converged": True,
                "unresolved_issues": [],
                "review_history_path": f"{state['output_dir']}/review_log.json",
            }
        }


@pytest.fixture
def graph(monkeypatch):
    built = FakeGraph()
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: built)
    return built


def _questions_file(tmp_path, questions) -> Path:
    path = tmp_path / "questions.txt"
    path.write_text("\n".join(questions) + "\n")
    return path


def _manifest(output_root: Path) -> dict:
    return json.loads((output_root / batch.MANIFEST_NAME).read_text())


# -- load_questions -----------------------------------------------------------------------


def test_load_questions_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("# a comment\n\nfirst question\n\n  second question  \n")
    assert batch.load_questions(path) == ["first question", "second question"]


# -- run_batch ----------------------------------------------------------------------------


def test_run_batch_records_every_question_in_the_manifest(tmp_path, graph):
    questions_file = _questions_file(tmp_path, ["q one", "q two", "q three"])
    summary = batch.run_batch(questions_file, output_root=tmp_path / "out")

    assert summary["completed"] == 3
    assert summary["failed"] == 0
    entries = _manifest(tmp_path / "out")["entries"]
    assert [e["question"] for e in entries] == ["q one", "q two", "q three"]
    assert all(e["status"] == "completed" for e in entries)
    assert all(e["final_paper_path"] for e in entries)


def test_run_batch_gives_each_question_its_own_output_dir(tmp_path, graph):
    questions_file = _questions_file(tmp_path, ["alpha beta", "gamma delta"])
    batch.run_batch(questions_file, output_root=tmp_path / "out")

    dirs = [e["output_dir"] for e in _manifest(tmp_path / "out")["entries"]]
    assert len(set(dirs)) == 2
    assert "q000_alpha-beta" in dirs[0]
    assert "q001_gamma-delta" in dirs[1]


def test_run_batch_continues_past_a_failing_question(tmp_path, monkeypatch):
    built = FakeGraph(fail_for=["q two"])
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: built)
    questions_file = _questions_file(tmp_path, ["q one", "q two", "q three"])

    summary = batch.run_batch(questions_file, output_root=tmp_path / "out")

    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert built.invoked == ["q one", "q two", "q three"]  # never stopped
    entries = {e["question"]: e for e in _manifest(tmp_path / "out")["entries"]}
    assert entries["q two"]["status"] == "failed"
    assert "blew up" in entries["q two"]["error"]
    assert entries["q three"]["status"] == "completed"


def test_run_batch_circuit_breaker_stops_after_consecutive_failures(tmp_path, monkeypatch):
    questions = [f"q {i}" for i in range(6)]
    built = FakeGraph(fail_for=questions)
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: built)

    summary = batch.run_batch(
        _questions_file(tmp_path, questions), output_root=tmp_path / "out", max_consecutive_failures=2
    )

    assert summary["failed"] == 2
    assert built.invoked == ["q 0", "q 1"]  # halted, didn't burn the rest
    statuses = {e["question"]: e["status"] for e in _manifest(tmp_path / "out")["entries"]}
    assert statuses == {"q 0": "failed", "q 1": "failed"}  # the rest were never started


def test_run_batch_circuit_breaker_resets_after_a_success(tmp_path, monkeypatch):
    questions = ["q 0", "q 1", "q 2", "q 3"]
    built = FakeGraph(fail_for=["q 0", "q 2"])  # never two in a row
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: built)

    summary = batch.run_batch(
        _questions_file(tmp_path, questions), output_root=tmp_path / "out", max_consecutive_failures=2
    )

    assert built.invoked == questions
    assert summary["completed"] == 2
    assert summary["failed"] == 2


def test_run_batch_resume_skips_completed_questions(tmp_path, monkeypatch):
    questions = ["q one", "q two"]
    questions_file = _questions_file(tmp_path, questions)
    output_root = tmp_path / "out"

    first = FakeGraph()
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: first)
    batch.run_batch(questions_file, output_root=output_root)

    second = FakeGraph()
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: second)
    summary = batch.run_batch(questions_file, output_root=output_root)

    assert second.invoked == []  # nothing re-run
    assert summary["skipped_already_done"] == 2


def test_run_batch_without_resume_redoes_everything(tmp_path, monkeypatch):
    questions_file = _questions_file(tmp_path, ["q one"])
    output_root = tmp_path / "out"

    first = FakeGraph()
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: first)
    batch.run_batch(questions_file, output_root=output_root)

    second = FakeGraph()
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: second)
    batch.run_batch(questions_file, output_root=output_root, resume=False)

    assert second.invoked == ["q one"]


def test_run_batch_retries_a_previously_failed_question_on_resume(tmp_path, monkeypatch):
    questions_file = _questions_file(tmp_path, ["q one", "q two"])
    output_root = tmp_path / "out"

    first = FakeGraph(fail_for=["q two"])
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: first)
    batch.run_batch(questions_file, output_root=output_root)

    second = FakeGraph()
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: second)
    batch.run_batch(questions_file, output_root=output_root)

    assert second.invoked == ["q two"]  # only the failure is retried


# -- run_batch_slice ----------------------------------------------------------------------


def test_run_batch_slice_processes_only_its_share(tmp_path, monkeypatch):
    questions = [f"q {i}" for i in range(10)]
    built = FakeGraph()
    monkeypatch.setattr(batch, "build_pipeline_graph", lambda: built)

    batch.run_batch_slice(_questions_file(tmp_path, questions), 1, 5, output_root=tmp_path / "out")

    assert built.invoked == ["q 2", "q 3"]


def test_run_batch_slices_are_disjoint_and_cover_everything(tmp_path, monkeypatch):
    questions = [f"q {i}" for i in range(10)]
    questions_file = _questions_file(tmp_path, questions)
    seen = []

    for slice_index in range(4):
        built = FakeGraph()
        monkeypatch.setattr(batch, "build_pipeline_graph", lambda b=built: b)
        batch.run_batch_slice(questions_file, slice_index, 4, output_root=tmp_path / f"out{slice_index}")
        seen.append(built.invoked)

    flattened = [q for chunk in seen for q in chunk]
    assert sorted(flattened) == sorted(questions)
    assert len(flattened) == len(set(flattened))


def test_run_batch_slice_keeps_global_indices_in_the_shared_manifest(tmp_path, monkeypatch):
    questions = [f"q {i}" for i in range(4)]
    questions_file = _questions_file(tmp_path, questions)
    output_root = tmp_path / "out"

    for slice_index in range(2):
        built = FakeGraph()
        monkeypatch.setattr(batch, "build_pipeline_graph", lambda b=built: b)
        batch.run_batch_slice(questions_file, slice_index, 2, output_root=output_root)

    entries = _manifest(output_root)["entries"]
    assert [e["index"] for e in entries] == [0, 1, 2, 3]  # merged, not clobbered
    assert [e["question"] for e in entries] == questions


def test_run_batch_slice_rejects_an_out_of_range_index(tmp_path, graph):
    with pytest.raises(ValueError, match="slice_index"):
        batch.run_batch_slice(_questions_file(tmp_path, ["q"]), 5, 3, output_root=tmp_path / "out")
