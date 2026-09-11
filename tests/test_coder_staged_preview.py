"""A staged file is described to the model from its bytes, first rows and last.

Before this, a file under CODER_DATA_DIR reached the prompt as a bare path, and
the Shiller S&P 500 series staged on Barkla ends in rows whose CPI, dividend and
long rate are all 0.0 — placeholders for months not yet published.
"""

from research_pipeline.agents.coder import acquire, provenance

SHILLER_LIKE = (
    "Date,SP500,Dividend,Consumer Price Index,PE10,Flag\n"
    "1871-01-01,4.44,0.26,12.46,0.0,0\n"
    "1871-02-01,4.50,0.26,12.84,0.0,0\n"
    "1990-01-01,339.97,11.14,127.4,17.05,0\n"
    "2026-06-01,7300.10,74.1,331.2,38.10,0\n"
    "2026-07-01,7481.34,0.0,0.0,0.0,0\n"
    "2026-08-01,7711.32,,0.0,0.0,0\n"
)


def test_a_staged_csv_is_described_with_its_first_and_last_rows(tmp_path):
    path = tmp_path / "historical_stock_market.csv"
    path.write_text(SHILLER_LIKE)

    preview = acquire.describe_local(path)

    assert preview["columns"] == ["Date", "SP500", "Dividend", "Consumer Price Index", "PE10", "Flag"]
    assert preview["row_count"] == 6
    assert preview["sample_rows"][0]["Date"] == "1871-01-01"
    assert [row["Date"] for row in preview["last_rows"]] == ["2026-06-01", "2026-07-01", "2026-08-01"]


def test_columns_ending_in_placeholders_are_named_with_their_run_length(tmp_path):
    path = tmp_path / "historical_stock_market.csv"
    path.write_text(SHILLER_LIKE)

    placeholders = acquire.describe_local(path)["trailing_placeholders"]

    assert placeholders == {"Dividend": 2, "Consumer Price Index": 2}


def test_zero_throughout_and_zero_only_at_the_start_are_not_placeholders():
    rows = [{"pe10": "0.0", "flag": "0"}, {"pe10": "0.0", "flag": "0"}] + [
        {"pe10": "17.1", "flag": "0"} for _ in range(4)
    ]
    assert acquire.trailing_placeholders(rows) == {}


def test_a_large_file_is_described_from_its_head_without_claiming_a_row_count(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "STAGED_FULL_READ_BYTES", 64)
    monkeypatch.setattr(acquire, "STAGED_HEAD_BYTES", 120)
    path = tmp_path / "big.csv"
    path.write_text(SHILLER_LIKE * 10)

    preview = acquire.describe_local(path)

    assert preview["columns"][0] == "Date"
    assert "row_count" not in preview
    assert "last_rows" not in preview


def test_an_unreadable_staged_file_is_described_as_nothing(tmp_path):
    assert acquire.describe_local(tmp_path / "missing.csv") == {}
    (tmp_path / "notes.txt").write_text("just one line of prose\n")
    assert acquire.describe_local(tmp_path / "notes.txt") == {}


def test_the_prompt_states_the_placeholders_as_fact(tmp_path):
    path = tmp_path / "historical_stock_market.csv"
    path.write_text(SHILLER_LIKE)
    source = provenance.DataSource(
        name="historical US stock market and CPI",
        kind=provenance.KIND_REAL_LOCAL,
        local_path=str(path),
        preview=acquire.describe_local(path),
    )

    block = provenance.prompt_block([source])

    assert "Columns: Date, SP500, Dividend, Consumer Price Index, PE10, Flag" in block
    assert "Last rows:" in block and "2026-08-01" in block
    assert "PLACEHOLDERS" in block
    assert "Consumer Price Index (last 2 rows)" in block


def test_preview_is_recorded_only_when_there_is_one():
    bare = provenance.DataSource(name="x", kind=provenance.KIND_REAL_LOCAL, local_path="/d/x.csv")
    assert "preview" not in bare.to_dict()
    described = provenance.DataSource(
        name="x", kind=provenance.KIND_REAL_LOCAL, local_path="/d/x.csv", preview={"columns": ["a", "b"]}
    )
    assert described.to_dict()["preview"] == {"columns": ["a", "b"]}
