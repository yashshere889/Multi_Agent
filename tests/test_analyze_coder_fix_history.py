"""Tests for scripts/analyze_coder_fix_history.py.

scripts/ isn't a package, so the module is loaded by path rather than imported
— same pattern as tests/test_kaggle_tunnel.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_coder_fix_history.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_coder_fix_history", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_coder_fix_history"] = module
    spec.loader.exec_module(module)
    return module


analyze = _load_module()


def _summary(experiments: list[dict]) -> dict:
    return {
        "experiments": experiments,
        "shared_infrastructure_path": "",
        "source_hypothesis_ids": [e["hypothesis_id"] for e in experiments],
        "generated_at": "2026-08-19T00:00:00Z",
        "model": "test-model",
    }


def _experiment(hid: str, fix_history: list[dict], status: str = "code_generated_not_run") -> dict:
    return {
        "hypothesis_id": hid,
        "status": status,
        "reason": "x",
        "code_path": f"experiments/{hid}",
        "assumptions_made": [],
        "results": None,
        "fix_attempts": len(fix_history),
        "fix_history": fix_history,
        "slurm_job_id": None,
    }


def _fix_entry(error_source: str, resolved: bool, same_error_streak: int | None = None) -> dict:
    entry = {
        "attempt": 1,
        "error_source": error_source,
        "error_summary": f"{error_source} broke",
        "code_path": "experiments/H1/fix_attempts/attempt_1/run.py",
        "resolved": resolved,
    }
    if same_error_streak is not None:
        entry["same_error_streak"] = same_error_streak
    return entry


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_find_summary_files_globs_recursively_and_accepts_a_bare_file(tmp_path):
    nested = tmp_path / "outputs" / "q1"
    nested.mkdir(parents=True)
    a = _write(nested, "coder_agent_summary_20260101T000000Z.json", _summary([]))
    (nested / "not_a_summary.json").write_text("{}")

    found = analyze.find_summary_files([tmp_path / "outputs"])
    assert found == [a]

    # A root that's already a file (not a directory) is accepted as-is, no glob needed.
    assert analyze.find_summary_files([a]) == [a]


def test_find_summary_files_ignores_a_missing_root(tmp_path):
    assert analyze.find_summary_files([tmp_path / "does_not_exist"]) == []


def test_load_skips_unparseable_json_with_a_warning(tmp_path, caplog):
    path = tmp_path / "coder_agent_summary_bad.json"
    path.write_text("not json")
    assert analyze._load(path) is None
    assert "could not parse" in caplog.text


def test_load_skips_a_file_with_no_experiments_list(tmp_path):
    path = _write(tmp_path, "coder_agent_summary_bad.json", {"model": "x"})
    assert analyze._load(path) is None


def test_aggregate_counts_occurrences_resolved_and_unresolved_at_end(tmp_path):
    # H1: one fix attempt, resolved — the sbatch-deferral-but-fixed shape seen
    # in real output (status can still be code_generated_not_run even though
    # the one fix attempt succeeded), so it must NOT count as unresolved_at_end.
    # H2: two fix attempts on the same error_source, neither resolved — the
    # second (last) one is what should count as unresolved_at_end.
    summary = _summary(
        [
            _experiment("H1", [_fix_entry("compile_check", resolved=True)]),
            _experiment(
                "H2",
                [
                    _fix_entry("run_experiment", resolved=False, same_error_streak=1),
                    _fix_entry("run_experiment", resolved=False, same_error_streak=2),
                ],
            ),
        ]
    )
    _write(tmp_path, "coder_agent_summary_1.json", summary)

    report = analyze.aggregate(analyze.find_summary_files([tmp_path]))

    assert report.files_scanned == 1
    assert report.files_skipped == 0
    assert report.experiments_scanned == 2
    assert report.fix_attempts_scanned == 3

    compile_stats = report.by_error_source["compile_check"]
    assert (compile_stats.occurrences, compile_stats.resolved, compile_stats.unresolved_at_end) == (
        1,
        1,
        0,
    )

    run_stats = report.by_error_source["run_experiment"]
    assert run_stats.occurrences == 2
    assert run_stats.resolved == 0
    assert run_stats.unresolved_at_end == 1  # only the last entry counts
    assert run_stats.stuck == 1  # only the streak>=2 entry counts
    assert run_stats.resolved_rate == 0.0
    assert compile_stats.resolved_rate == 1.0


def test_aggregate_treats_missing_same_error_streak_as_not_stuck(tmp_path):
    # Real pre-existing summaries on disk predate same_error_streak entirely —
    # this must not crash or be treated as "definitely stuck".
    summary = _summary([_experiment("H1", [_fix_entry("compile_check", resolved=True)])])
    _write(tmp_path, "coder_agent_summary_1.json", summary)

    report = analyze.aggregate(analyze.find_summary_files([tmp_path]))
    assert report.by_error_source["compile_check"].stuck == 0


def test_aggregate_skips_a_corrupt_file_but_keeps_scanning_the_rest(tmp_path):
    good = _summary([_experiment("H1", [_fix_entry("compile_check", resolved=True)])])
    _write(tmp_path, "coder_agent_summary_good.json", good)
    (tmp_path / "coder_agent_summary_bad.json").write_text("{not json")

    report = analyze.aggregate(analyze.find_summary_files([tmp_path]))
    assert report.files_scanned == 1
    assert report.files_skipped == 1
    assert report.experiments_scanned == 1


def test_aggregate_ignores_experiments_with_no_fix_history(tmp_path):
    summary = _summary([_experiment("H1", []), _experiment("H2", [])])
    _write(tmp_path, "coder_agent_summary_1.json", summary)

    report = analyze.aggregate(analyze.find_summary_files([tmp_path]))
    assert report.experiments_scanned == 0
    assert report.by_error_source == {}


def test_render_ranks_error_sources_by_occurrence_descending(tmp_path):
    summary = _summary(
        [
            _experiment(
                "H1",
                [
                    _fix_entry("run_experiment", resolved=False),
                    _fix_entry("run_experiment", resolved=False),
                ],
            ),
            _experiment("H2", [_fix_entry("static_lint", resolved=True)]),
        ]
    )
    _write(tmp_path, "coder_agent_summary_1.json", summary)

    report = analyze.aggregate(analyze.find_summary_files([tmp_path]))
    text = analyze.render(report)
    # run_experiment (count 2) must be listed before static_lint (count 1).
    assert text.index("run_experiment") < text.index("static_lint")


def test_render_with_no_data_says_so_instead_of_an_empty_table():
    report = analyze.Report()
    assert "No fix_history entries found." in analyze.render(report)


def test_report_to_json_round_trips_through_json_dumps(tmp_path):
    summary = _summary([_experiment("H1", [_fix_entry("compile_check", resolved=True)])])
    _write(tmp_path, "coder_agent_summary_1.json", summary)

    report = analyze.aggregate(analyze.find_summary_files([tmp_path]))
    payload = analyze._report_to_json(report)
    reloaded = json.loads(json.dumps(payload))  # must be plain JSON-serializable
    assert reloaded["by_error_source"]["compile_check"]["occurrences"] == 1


def test_main_writes_json_report_when_requested(tmp_path, capsys):
    summary = _summary([_experiment("H1", [_fix_entry("compile_check", resolved=True)])])
    _write(tmp_path, "coder_agent_summary_1.json", summary)
    json_out = tmp_path / "report.json"

    exit_code = analyze.main([str(tmp_path), "--json", str(json_out)])

    assert exit_code == 0
    assert json_out.exists()
    payload = json.loads(json_out.read_text())
    assert payload["files_scanned"] == 1
    captured = capsys.readouterr()
    assert "compile_check" in captured.out
    assert "Wrote" in captured.out


def test_main_accepts_a_root_with_no_summary_files(tmp_path, capsys):
    exit_code = analyze.main([str(tmp_path)])
    assert exit_code == 0
    assert "No fix_history entries found." in capsys.readouterr().out
