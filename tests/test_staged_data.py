"""The staged-data inventory the Hypothesis and Planner agents are shown."""

from dataclasses import replace

from research_pipeline import staged_data
from research_pipeline.agents.coder import acquire
from research_pipeline.agents.experiment_planner import prompts as planner_prompts
from research_pipeline.agents.hypothesis import prompts as hypothesis_prompts

README = """Staged data. Real data, not synthetic.

fama_french_three_factor_daily_returns.csv    1926-07-01 .. 2026-06-30
fama_french_three_factor_monthly_returns.csv  1926-07 .. 2026-06

  Source: Kenneth R. French Data Library.
  UNITS: percent per period, NOT decimals.

historical_stock_market.csv
  Shiller S&P 500 monthly series.

--- text corpora ---
"""


def _stage(tmp_path):
    (tmp_path / "README.txt").write_text(README)
    (tmp_path / "fama_french_three_factor_monthly_returns.csv").write_text(
        "date,Mkt-RF,SMB,HML,RF\n1926-07,2.96,-2.56,-2.43,0.22\n1926-08,2.64,-1.17,3.82,0.25\n"
    )
    (tmp_path / "historical_stock_market.csv").write_text("Date,SP500,CPI\n1871-01-01,4.44,12.46\n2026-08-01,7711.3,0.0\n")
    (tmp_path / "keyword_packed_alias_returns.csv").symlink_to(tmp_path / "historical_stock_market.csv")
    return tmp_path


def test_the_inventory_lists_each_real_file_once_and_skips_aliases_and_prose(tmp_path):
    entries = staged_data.inventory(_stage(tmp_path))

    assert [entry["file"] for entry in entries] == [
        "fama_french_three_factor_monthly_returns.csv",
        "historical_stock_market.csv",
    ]
    assert entries[0]["columns"] == ["date", "Mkt-RF", "SMB", "HML", "RF"]
    assert entries[0]["row_count"] == 2
    assert entries[0]["last_row"]["date"] == "1926-08"


def test_readme_notes_carry_the_units_for_files_grouped_on_consecutive_lines(tmp_path):
    staged = _stage(tmp_path)
    daily = staged / "fama_french_three_factor_daily_returns.csv"
    daily.write_text("date,Mkt-RF,RF\n1926-07-01,0.1,0.01\n")

    for path in (daily, staged / "fama_french_three_factor_monthly_returns.csv"):
        notes = acquire.readme_notes(path)
        assert "UNITS: percent per period" in notes
        assert "Shiller" not in notes

    assert acquire.readme_notes(staged / "historical_stock_market.csv").startswith("historical_stock_market.csv")
    assert "text corpora" not in acquire.readme_notes(staged / "historical_stock_market.csv")


def test_an_alias_symlink_finds_its_target_s_readme_entry(tmp_path):
    staged = _stage(tmp_path)
    assert "Shiller" in acquire.readme_notes(staged / "keyword_packed_alias_returns.csv")


def test_the_block_names_files_columns_notes_and_the_instruction(tmp_path):
    block = staged_data.prompt_block("USE THEM", _stage(tmp_path))

    assert "- fama_french_three_factor_monthly_returns.csv, 2 rows; columns: date, Mkt-RF, SMB, HML, RF" in block
    assert "UNITS: percent per period" in block
    assert block.rstrip().endswith("USE THEM")


def test_nothing_staged_renders_nothing(tmp_path, monkeypatch):
    assert staged_data.prompt_block("USE THEM", tmp_path) == ""
    monkeypatch.setattr(staged_data, "settings", replace(staged_data.settings, coder_data_dir=""))
    assert staged_data.prompt_block("USE THEM") == ""


def test_prompts_are_unchanged_when_nothing_is_staged():
    plan = planner_prompts.PLAN_PROMPT.format(
        hypothesis_block="{}", hypothesis_id="H1", literature_summary="s",
        methods_overview_block="[]", gaps_block="[]", staged_data_block="",
    )
    assert "Gaps (JSON):\n[]\n\nReturn ONLY a JSON object" in plan

    ranking = hypothesis_prompts.RANKING_PROMPT.format(
        research_question_line="", hypotheses_block="[]", literature_summary="s",
        gaps_block="[]", interdisciplinary_block="", staged_data_block="",
    )
    assert "Gaps (JSON):\n[]\n\nRank all 3" in ranking
