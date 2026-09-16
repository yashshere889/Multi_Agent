"""Tests for scripts/export_preference_data.py.

scripts/ isn't a package, so the module is loaded by path rather than imported
— same pattern as tests/test_analyze_coder_fix_history.py.

The slicer gets a *round-trip* test against the real renderer rather than a
fixture: it exists to recover the sections a rendered run.py was built from, so
the only assertion worth making is that it returns what was put in. That is also
the test that fails the day someone edits run.py.template's banners, which is the
one change it can't survive.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from research_pipeline.agents.coder import sandbox, transcript
from research_pipeline.llm_sections import render_sections

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_preference_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_preference_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_preference_data"] = module
    spec.loader.exec_module(module)
    return module


export = _load_module()

SECTIONS = {
    "imports": "import json\nimport numpy as np",
    "configuration": "SEED = 7\nSTEPS = 3",
    "load_data_function": 'def load_data():\n    """Doc."""\n    return {"x": [1, 2]}',
    "build_model_function": "def build_model(data):\n    return {'w': 0.0}",
    "run_experiment_function": "def run_experiment(data, model):\n    return {'loss': 0.5}",
    "evaluate_function": "def evaluate(experiment_output, data):\n    return {'acc': 1.0}",
    "helpers": "def _helper():\n    return 1",
}


def _render(sections: dict[str, str]) -> str:
    return sandbox.render_experiment_template(
        hypothesis_id="H1",
        objective="o",
        design="d",
        data_description="dd",
        baseline="b",
        success_criteria="sc",
        agent_imports=sections.get("imports", ""),
        agent_configuration=sections.get("configuration", ""),
        load_data_function=sections["load_data_function"],
        build_model_function=sections["build_model_function"],
        run_experiment_function=sections["run_experiment_function"],
        evaluate_function=sections["evaluate_function"],
        agent_helpers=sections.get("helpers", ""),
    )


def test_slicing_a_rendered_run_py_recovers_every_section_exactly():
    assert export.slice_run_py(_render(SECTIONS)) == SECTIONS


def test_slicing_reports_an_omitted_section_as_empty_not_as_the_template_placeholder():
    """render_experiment_with_spans substitutes '# (no helper functions needed)'
    for a section the model left blank. Attributing that to the model would put
    a line it never wrote into a training target."""
    sliced = export.slice_run_py(_render({**SECTIONS, "helpers": "", "imports": ""}))
    assert sliced["helpers"] == ""
    assert sliced["imports"] == ""


def test_parse_sections_round_trips_the_wire_format():
    assert export.parse_sections(render_sections(SECTIONS)) == SECTIONS


# ---------------------------------------------------------------------------
# Building rows from a run's artefacts
# ---------------------------------------------------------------------------


def _transcript(kind: str, prompt: str, response: str, field_names=None) -> dict:
    return {
        "schema": transcript.SCHEMA_VERSION,
        "kind": kind,
        "recorded_at": "2026-09-16T00:00:00+00:00",
        "model": "test-model",
        "system_prompt": "system",
        "user_prompt": prompt,
        "raw_responses": [response],
        "field_names": field_names,
        "temperature": 0.0,
        "max_tokens": 1000,
        "error": "",
    }


def _write_attempt(directory: Path, sections: dict[str, str], record: dict | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.py").write_text(_render(sections))
    (directory / "requirements.txt").write_text("numpy\n")
    if record is not None:
        (directory / transcript.TRANSCRIPT_FILENAME).write_text(json.dumps(record))


def _run_tree(tmp_path: Path, *, with_transcripts: bool, targeted: bool = False) -> Path:
    """One experiment that failed once and was fixed: attempt_1 holds the broken
    code, the experiment directory holds what replaced it."""
    root = tmp_path / "outputs"
    experiment_dir = root / "experiments" / "H1"
    broken = {**SECTIONS, "evaluate_function": "def evaluate(experiment_output, data):\n    pass"}

    _write_attempt(
        experiment_dir / "fix_attempts" / "attempt_1",
        broken,
        _transcript("generate", "GENERATE PROMPT", render_sections(broken))
        if with_transcripts
        else None,
    )
    fixed_response = (
        render_sections({"evaluate_function": SECTIONS["evaluate_function"]})
        if targeted
        else render_sections(SECTIONS)
    )
    _write_attempt(
        experiment_dir,
        SECTIONS,
        _transcript(
            "fix",
            "FIX PROMPT: empty_body",
            fixed_response,
            ["evaluate_function"] if targeted else None,
        )
        if with_transcripts
        else None,
    )

    (root / "coder_agent_summary_20260916T000000Z.json").write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "hypothesis_id": "H1",
                        "status": "completed",
                        "reason": "",
                        "code_path": str(experiment_dir),
                        "assumptions_made": [],
                        "results": {"metrics": {}, "meets_success_criteria": True, "notes": ""},
                        "fix_attempts": 1,
                        "fix_history": [
                            {
                                "attempt": 1,
                                "error_source": "empty_body",
                                "error_summary": "evaluate() has no implementation",
                                "code_path": str(
                                    experiment_dir / "fix_attempts" / "attempt_1" / "run.py"
                                ),
                                "resolved": True,
                                "same_error_streak": 1,
                                "regenerated_sections": ["evaluate_function"] if targeted else [],
                                "assumptions_made": [],
                            }
                        ],
                        "slurm_job_id": None,
                        "starter_used": "",
                    }
                ],
                "shared_infrastructure_path": "",
                "source_hypothesis_ids": ["H1"],
                "generated_at": "2026-09-16T00:00:00Z",
                "model": "test-model",
            }
        )
    )
    return root


def _export(root: Path, **overrides):
    args = {
        "roots": [str(root)],
        "format": "sft-fix",
        "experiments_root": [],
        "error_source": [],
        "status": [],
        "require_prompt": False,
        "exclude_partial": False,
        "skip_targeted": False,
    }
    args.update(overrides)
    return export.export(type("Args", (), args)())


def test_sft_fix_row_pairs_the_fix_prompt_with_the_response_that_cleared_it(tmp_path):
    rows, stats = _export(_run_tree(tmp_path, with_transcripts=True))
    assert stats["rows"] == 1
    (row,) = rows
    assert row["prompt"] == "FIX PROMPT: empty_body"
    assert row["completion"] == render_sections(SECTIONS)
    assert row["error_source"] == "empty_body"
    # Verbatim on both sides — nothing rebuilt, nothing inferred.
    assert row["prompt_source"] == row["completion_source"] == "transcript"
    assert row["partial"] is False


def test_an_unresolved_fix_is_never_a_training_row(tmp_path):
    """`resolved` is a real check's verdict that the regeneration got past the
    failure. Without it there is no evidence the 'chosen' side is better."""
    root = _run_tree(tmp_path, with_transcripts=True)
    summary = next(root.glob("coder_agent_summary_*.json"))
    loaded = json.loads(summary.read_text())
    loaded["experiments"][0]["fix_history"][0]["resolved"] = False
    summary.write_text(json.dumps(loaded))

    rows, stats = _export(root)
    assert rows == []
    assert stats["skipped"]["fix_not_resolved"] == 1


def test_dpo_row_pairs_the_failed_attempt_against_what_replaced_it(tmp_path):
    rows, _ = _export(_run_tree(tmp_path, with_transcripts=True), format="dpo")
    (row,) = rows
    assert row["prompt"] == "GENERATE PROMPT"
    assert "def evaluate(experiment_output, data):\n    pass" in row["rejected"]
    assert row["chosen"] == render_sections(SECTIONS)
    assert row["chosen_reconstructed"] is False


def test_dpo_merges_a_targeted_regeneration_over_the_attempt_it_fixed(tmp_path):
    """A targeted regeneration answers only the sections it was asked for, so on
    its own it is not an answer to the prompt the rejected attempt saw. Merging
    reproduces the program that was then checked — the same merge
    _assemble_generation does with previous=."""
    rows, _ = _export(_run_tree(tmp_path, with_transcripts=True, targeted=True), format="dpo")
    (row,) = rows
    assert row["chosen_reconstructed"] is True
    chosen = export.parse_sections(row["chosen"])
    # The fixed section came from the regeneration...
    assert chosen["evaluate_function"] == SECTIONS["evaluate_function"]
    # ...and everything it wasn't asked for was carried over untouched.
    assert chosen["load_data_function"] == SECTIONS["load_data_function"]


def test_skip_targeted_drops_those_pairs_instead(tmp_path):
    rows, stats = _export(
        _run_tree(tmp_path, with_transcripts=True, targeted=True),
        format="dpo",
        skip_targeted=True,
    )
    assert rows == []
    assert stats["skipped"]["dpo_targeted"] == 1


def test_artefacts_with_no_transcript_still_yield_code_pairs(tmp_path):
    """Everything this project ran before transcript.py existed. The prompt is
    gone and cannot be reconstructed; the code on both sides is recoverable, and
    the row says so rather than pretending otherwise."""
    rows, stats = _export(_run_tree(tmp_path, with_transcripts=False), format="dpo")
    (row,) = rows
    assert row["prompt"] is None
    assert row["prompt_source"] is None
    assert row["chosen_source"] == row["rejected_source"] == "sliced"
    assert row["partial"] is True
    assert export.parse_sections(row["chosen"])["evaluate_function"] == (
        SECTIONS["evaluate_function"]
    )
    assert stats["rows_with_prompt"] == 0


def test_require_prompt_drops_promptless_rows(tmp_path):
    rows, stats = _export(
        _run_tree(tmp_path, with_transcripts=False), format="dpo", require_prompt=True
    )
    assert rows == []
    assert stats["skipped"]["filtered_no_prompt"] == 1


def test_sft_fix_needs_a_transcript_and_says_so_when_there_is_none(tmp_path):
    rows, stats = _export(_run_tree(tmp_path, with_transcripts=False))
    assert rows == []
    assert stats["skipped"]["fix_no_transcript"] == 1


def test_sft_exports_the_generation_that_ended_a_completed_experiment(tmp_path):
    rows, _ = _export(_run_tree(tmp_path, with_transcripts=True), format="sft", status=["completed"])
    (row,) = rows
    assert row["prompt"] == "FIX PROMPT: empty_body"
    assert row["completion"] == render_sections(SECTIONS)
    assert row["fix_attempts"] == 1


def test_sft_skips_an_experiment_whose_status_was_not_asked_for(tmp_path):
    root = _run_tree(tmp_path, with_transcripts=True)
    summary = next(root.glob("coder_agent_summary_*.json"))
    loaded = json.loads(summary.read_text())
    loaded["experiments"][0]["status"] = "code_generated_not_run"
    summary.write_text(json.dumps(loaded))

    rows, stats = _export(root, format="sft", status=["completed"])
    assert rows == []
    assert stats["skipped"]["sft_status"] == 1


# ---------------------------------------------------------------------------
# Locating an experiment directory that has moved
# ---------------------------------------------------------------------------


def test_an_absolute_code_path_from_another_machine_resolves_by_its_tail(tmp_path):
    """`code_path` is recorded as it was at run time — routinely
    /kaggle/working/experiments/H1, or a scratch mount — and the artefacts are
    then copied somewhere local. The tail from `experiments` onwards is what
    survives that."""
    root = _run_tree(tmp_path, with_transcripts=True)
    summary = next(root.glob("coder_agent_summary_*.json"))
    loaded = json.loads(summary.read_text())
    loaded["experiments"][0]["code_path"] = "/kaggle/working/experiments/H1"
    summary.write_text(json.dumps(loaded))

    rows, stats = _export(root)
    assert stats["rows"] == 1
    assert rows[0]["resolved_by"] == "summary"


def test_a_directory_found_only_by_name_search_is_flagged_not_trusted(tmp_path):
    """Several runs' outputs side by side all contain an `experiments/H1`.
    Pairing one run's summary with another run's code would attach an error to
    code it never described, so the guess is labelled and counted."""
    root = _run_tree(tmp_path, with_transcripts=True)
    summary = next(root.glob("coder_agent_summary_*.json"))
    loaded = json.loads(summary.read_text())
    loaded["experiments"][0]["code_path"] = "/elsewhere/runs/job123/H1"
    summary.write_text(json.dumps(loaded))

    rows, stats = _export(root)
    assert rows[0]["resolved_by"] == "search"
    assert stats["rows_resolved_by_search"] == 1


def test_an_experiment_directory_that_is_gone_is_counted_not_guessed(tmp_path):
    root = _run_tree(tmp_path, with_transcripts=True)
    summary = next(root.glob("coder_agent_summary_*.json"))
    loaded = json.loads(summary.read_text())
    loaded["experiments"][0]["code_path"] = "/elsewhere/experiments/H9"
    summary.write_text(json.dumps(loaded))

    rows, stats = _export(root)
    assert rows == []
    assert stats["unresolved_experiment_dirs"] == 1
