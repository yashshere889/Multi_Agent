"""An excess return compounded as a total return is a units error, not a result.

Barkla job 10509628 grew a retirement portfolio on Fama-French `Mkt-RF` alone,
dropping ~3%/yr of risk-free return: its 4% withdrawal rule failed in 10,000 of
10,000 paths, where the literature reports about 95% success over 30 years.
"""

from research_pipeline.agents.coder import coder_agent, sandbox
from research_pipeline.agents.coder.schema import VALID_ERROR_SOURCES

FAMA_FRENCH_COLUMNS = ["date", "Mkt-RF", "SMB", "HML", "RF"]

JOB_10509628 = """
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Mkt-RF'] = df['Mkt-RF'] / 100.0
    df['RF'] = df['RF'] / 100.0
    return df

def run_experiment(data, model):
    monthly_returns = data['Mkt-RF'].values
    portfolio_value = PORTFOLIO_INITIAL_VALUE
    for i in range(360):
        portfolio_value *= (1 + monthly_returns[i])
        portfolio_value -= withdrawal_per_month
"""

CORRECTED = JOB_10509628.replace(
    "monthly_returns = data['Mkt-RF'].values",
    "monthly_returns = (data['Mkt-RF'] + data['RF']).values",
)


def test_the_recorded_run_is_flagged():
    findings = sandbox.check_excess_return_usage(JOB_10509628, FAMA_FRENCH_COLUMNS)
    assert len(findings) == 1
    assert "'Mkt-RF' is a return in excess of 'RF'" in findings[0]
    assert "df['Mkt-RF'] + df['RF']" in findings[0]


def test_adding_the_risk_free_rate_back_clears_it():
    assert sandbox.check_excess_return_usage(CORRECTED, FAMA_FRENCH_COLUMNS) == []


def test_an_input_without_a_risk_free_column_says_nothing():
    assert sandbox.check_excess_return_usage(JOB_10509628, ["date", "Mkt-RF", "SMB"]) == []


def test_code_that_compounds_nothing_says_nothing():
    source = "coefficients = sm.OLS(y, data[['Mkt-RF', 'SMB']]).fit().params\n"
    assert sandbox.check_excess_return_usage(source, FAMA_FRENCH_COLUMNS) == []


def test_the_sum_written_the_other_way_round_clears_it():
    source = JOB_10509628.replace(
        "monthly_returns = data['Mkt-RF'].values",
        "monthly_returns = (data['RF'] + data['Mkt-RF']).values",
    )
    assert sandbox.check_excess_return_usage(source, FAMA_FRENCH_COLUMNS) == []


def test_underscore_and_case_variants_are_recognised():
    source = JOB_10509628.replace("Mkt-RF", "mkt_rf")
    findings = sandbox.check_excess_return_usage(source, ["date", "mkt_rf", "rf"])
    assert len(findings) == 1


def test_a_column_that_only_looks_similar_is_left_alone():
    source = "portfolio *= (1 + data['RF_SPREAD'].values[i])\n"
    assert sandbox.check_excess_return_usage(source, ["RF_SPREAD", "RF"]) == []


def test_the_error_source_is_registered_everywhere_it_must_be():
    # A member missing from either list silently scores every regeneration as
    # having made no progress (_cleared_previous_error reads the stage order).
    assert "excess_return_as_total" in VALID_ERROR_SOURCES
    assert "excess_return_as_total" in coder_agent._ERROR_STAGE_ORDER
    assert set(coder_agent._ERROR_STAGE_ORDER) == set(VALID_ERROR_SOURCES)
