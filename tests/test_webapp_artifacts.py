import json

import pytest

from research_pipeline.webapp import artifacts


def _populate(run_dir):
    """A run directory shaped like a real one: each stage's output plus the
    drafts under outputs/, downloaded PDFs under papers/, generated code and its
    fix attempts under experiments/, and the loose run files at the top."""
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "papers").mkdir(parents=True)
    (run_dir / "experiments" / "H1" / "fix_attempts" / "attempt_1").mkdir(parents=True)
    (run_dir / "experiments" / "H1" / "__pycache__").mkdir(parents=True)

    (run_dir / "run.json").write_text(json.dumps({"run_id": "x"}))
    (run_dir / "events.jsonl").write_text('{"seq": 0}\n')
    (run_dir / "outputs" / "hypotheses.json").write_text(json.dumps({"hypotheses": [{"id": "H1"}]}))
    (run_dir / "outputs" / "v1.pdf").write_bytes(b"%PDF-1.4 fake")
    (run_dir / "papers" / "metadata.json").write_text("[]")
    (run_dir / "papers" / "2005.11401.pdf").write_bytes(b"%PDF-1.4 fake")
    (run_dir / "experiments" / "H1" / "run.py").write_text("print('hi')\n")
    (run_dir / "experiments" / "H1" / "results.json").write_text('{"metrics": {"f1": 0.7}}')
    (run_dir / "experiments" / "H1" / "fix_attempts" / "attempt_1" / "run.py").write_text("broken\n")
    (run_dir / "experiments" / "H1" / "__pycache__" / "run.cpython-314.pyc").write_bytes(b"\x00\x01")


def test_lists_every_file_and_skips_build_residue(tmp_path):
    _populate(tmp_path)
    found, truncated = artifacts.list_artifacts(tmp_path)

    rels = {a.rel for a in found}
    assert "outputs/hypotheses.json" in rels
    assert "experiments/H1/fix_attempts/attempt_1/run.py" in rels
    assert "run.json" in rels
    # __pycache__ is residue from running the generated code, not a result.
    assert not any("__pycache__" in rel for rel in rels)
    assert not truncated


def test_groups_by_top_level_directory(tmp_path):
    _populate(tmp_path)
    found, _ = artifacts.list_artifacts(tmp_path)
    grouped = artifacts.group_artifacts(found)

    keys = [key for key, _heading, _files in grouped]
    # GROUP_ORDER, not directory order: outputs first, loose run files last.
    assert keys == ["outputs", "experiments", "papers", ""]
    headings = {key: heading for key, heading, _ in grouped}
    assert headings[""] == "Run files"


def test_an_unknown_directory_still_appears(tmp_path):
    (tmp_path / "somewhere_new").mkdir()
    (tmp_path / "somewhere_new" / "thing.txt").write_text("hi")
    found, _ = artifacts.list_artifacts(tmp_path)

    keys = [key for key, _heading, _files in artifacts.group_artifacts(found)]
    assert keys == ["somewhere_new"]


def test_listing_is_capped_and_says_so(tmp_path):
    (tmp_path / "papers").mkdir()
    for i in range(12):
        (tmp_path / "papers" / f"p{i}.pdf").write_bytes(b"%PDF")

    found, truncated = artifacts.list_artifacts(tmp_path, limit=5)
    assert len(found) == 5
    assert truncated


def test_a_symlinked_directory_is_not_followed(tmp_path):
    """A symlink out of the run would produce links resolve_inside then refuses,
    so the listing must not offer them in the first place."""
    outside = tmp_path.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")

    run_dir = tmp_path / "run"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "outputs" / "real.json").write_text("{}")
    (run_dir / "escape").symlink_to(outside, target_is_directory=True)

    found, _ = artifacts.list_artifacts(run_dir)
    assert {a.rel for a in found} == {"outputs/real.json"}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("v1.pdf", artifacts.PDF),
        ("hypotheses.json", artifacts.JSON_KIND),
        ("events.jsonl", artifacts.JSON_KIND),
        ("run.py", artifacts.TEXT),
        ("run.sbatch", artifacts.TEXT),
        ("stdout.log", artifacts.TEXT),
        ("model.bin", artifacts.BINARY),
    ],
)
def test_kind_for(tmp_path, name, expected):
    assert artifacts.kind_for(tmp_path / name) == expected


def test_only_text_kinds_are_viewable_in_the_page(tmp_path):
    _populate(tmp_path)
    found, _ = artifacts.list_artifacts(tmp_path)
    by_rel = {a.rel: a for a in found}

    assert by_rel["outputs/hypotheses.json"].viewable
    assert by_rel["experiments/H1/run.py"].viewable
    # A PDF is served raw and opened by the browser's own viewer.
    assert not by_rel["outputs/v1.pdf"].viewable


def test_preview_reindents_compact_json(tmp_path):
    path = tmp_path / "out.json"
    path.write_text('{"a":1,"b":[2,3]}')

    preview = artifacts.read_preview(path)
    assert "\n" in preview.text
    assert '"a": 1' in preview.text
    assert not preview.truncated


def test_preview_shows_unparseable_json_verbatim(tmp_path):
    """A half-written output file, mid-run, is exactly when someone looks."""
    path = tmp_path / "out.json"
    path.write_text('{"a": 1, "b": [2,')

    preview = artifacts.read_preview(path)
    assert preview.text == '{"a": 1, "b": [2,'
    assert preview.error is None


def test_preview_truncates_a_large_file(tmp_path):
    path = tmp_path / "big.log"
    path.write_text("x" * 5000)

    preview = artifacts.read_preview(path, max_bytes=1000)
    assert len(preview.text) == 1000
    assert preview.truncated


def test_preview_replaces_undecodable_bytes_rather_than_raising(tmp_path):
    path = tmp_path / "weird.txt"
    path.write_bytes(b"ok \xff\xfe done")

    preview = artifacts.read_preview(path)
    assert "ok " in preview.text
    assert preview.error is None


def test_missing_run_directory_lists_nothing(tmp_path):
    assert artifacts.list_artifacts(tmp_path / "nope") == ([], False)


@pytest.mark.parametrize(
    "size, expected",
    [(0, "0 B"), (512, "512 B"), (1536, "1.5 KB"), (20 * 1024, "20 KB"), (5 * 1024**2, "5.0 MB")],
)
def test_human_size(size, expected):
    assert artifacts.human_size(size) == expected
