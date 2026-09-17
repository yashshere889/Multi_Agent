"""Wrapping an unguarded read is mechanical, so Python does it.

Barkla job 10537067 spent two of its fix attempts on `missing_data_fallback`:
the fix prompt named the remedy and the line number, and the model wrote the
same bare `read_csv` back both times. Same shape as the pandas patch — the
guidance was right and the model did not apply it, so this stops asking.
"""

import ast

from research_pipeline.agents.coder import repair, sandbox

UNGUARDED = '''def load_data():
    """Read the staged Fama-French factors."""
    path = DATA_PATH
    df = pd.read_csv(path)
    return df
'''

GUARDED = '''def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return _synth()
'''

COLUMNS = ["date", "Mkt-RF", "SMB", "HML", "RF"]


def test_the_patched_source_satisfies_the_gate_that_rejected_it():
    assert sandbox.check_data_fallback(UNGUARDED)

    patched, changes = repair.guard_data_read(UNGUARDED, COLUMNS)

    assert changes == ["wrapped load_data's read in try/except with a synthesized fallback"]
    assert sandbox.check_data_fallback(patched) == []


def test_the_original_body_is_preserved_inside_the_try():
    patched, _ = repair.guard_data_read(UNGUARDED, COLUMNS)

    assert "    try:\n" in patched
    for line in ("path = DATA_PATH", "df = pd.read_csv(path)", "return df"):
        assert line in patched
    # The docstring stays outside the try — wrapping it changes nothing.
    assert patched.index('"""Read the staged') < patched.index("try:")


def test_the_fallback_frame_uses_the_real_column_names():
    patched, _ = repair.guard_data_read(UNGUARDED, COLUMNS)

    for column in COLUMNS:
        assert repr(column) in patched
    # A date-named column gets dates; a measurement gets draws.
    assert "date_range" in patched
    assert "'Mkt-RF': [_rng.gauss" in patched


def test_no_known_columns_still_yields_a_usable_frame():
    patched, changes = repair.guard_data_read(UNGUARDED, [])

    assert changes
    assert "'feature'" in patched and "'target'" in patched


def test_an_already_guarded_read_is_left_alone():
    assert repair.guard_data_read(GUARDED, COLUMNS) == (GUARDED, [])


def test_applying_it_twice_changes_nothing_the_second_time():
    once, _ = repair.guard_data_read(UNGUARDED, COLUMNS)
    twice, changes = repair.guard_data_read(once, COLUMNS)

    assert changes == []
    assert twice == once


def test_unparseable_or_absent_load_data_degrades_to_a_no_op():
    broken = "def load_data(:\n  bad"
    assert repair.guard_data_read(broken, COLUMNS) == (broken, [])
    other = "def evaluate(x):\n    return pd.read_csv('x.csv')\n"
    assert repair.guard_data_read(other, COLUMNS)[1] == []


def test_the_fallback_adds_no_dependency_the_original_did_not_have():
    """numpy would fail provisioning offline where the original needed only pandas."""
    patched, _ = repair.guard_data_read(UNGUARDED, COLUMNS)

    assert "import numpy" not in patched
    assert "import random as _random" in patched


def test_the_patch_always_compiles():
    patched, _ = repair.guard_data_read(UNGUARDED, COLUMNS)
    ast.parse(patched)
