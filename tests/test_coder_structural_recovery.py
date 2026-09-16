"""A malformed response the model recovered from must not spend the budget.

Barkla job 10537067 produced `invalid_format` at fix attempts 1 and 3, cleared
both on the next regeneration, and was killed anyway: the two recovered hiccups
spent the whole default structural budget of 2, ending the plan with ten fix
attempts untouched while the failure actually blocking it — an unguarded
`read_csv` the model would not wrap — was never the reason it stopped.
"""

from research_pipeline.agents.coder.coder_agent import (
    _STRUCTURAL_ERROR_SOURCES,
    _structural_retries_spent,
)

# r16's fix_history, error_source and resolved only.
R16_HISTORY = [
    {"error_source": "invalid_format", "resolved": True},
    {"error_source": "missing_data_fallback", "resolved": False},
    {"error_source": "invalid_format", "resolved": True},
    {"error_source": "missing_data_fallback", "resolved": False},
]


def test_r16_s_recovered_hiccups_cost_nothing():
    assert _structural_retries_spent(R16_HISTORY) == 0


def test_a_model_that_never_returns_a_program_still_exhausts_the_budget():
    never = [{"error_source": "invalid_format", "resolved": False} for _ in range(3)]
    assert _structural_retries_spent(never) == 3


def test_only_structural_sources_count():
    history = [
        {"error_source": "compile_check", "resolved": False},
        {"error_source": "run_experiment", "resolved": False},
        {"error_source": "implausible_results", "resolved": False},
    ]
    assert _structural_retries_spent(history) == 0


def test_a_missing_resolved_field_counts_as_unrecovered():
    """Conservative: an entry with no verdict is treated as still broken."""
    assert _structural_retries_spent([{"error_source": "missing_sections"}]) == 1


def test_an_empty_history_has_spent_nothing():
    assert _structural_retries_spent([]) == 0


def test_every_counted_source_is_a_structural_one():
    assert "invalid_format" in _STRUCTURAL_ERROR_SOURCES
    assert "compile_check" not in _STRUCTURAL_ERROR_SOURCES
