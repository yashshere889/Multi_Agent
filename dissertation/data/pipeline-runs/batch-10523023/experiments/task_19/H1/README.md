# Hypothesis H1: Policy Uncertainty from News Text vs Lagged Unemployment

## Objective
To determine whether policy uncertainty indices derived from news text using dictionary-based methods show a stronger correlation with subsequent regional unemployment changes compared to lagged unemployment alone, when controlling for macroeconomic factors such as GDP growth and inflation.

## Methods
This experiment implements a comparative econometric analysis using panel data regression models. It constructs two versions of the independent variable:
1. Based on lagged unemployment
2. Based on a dictionary-derived EPU index from news text

Both are regressed against the dependent variable (unemployment change) while controlling for GDP growth and inflation.

### Text Mining Method
We apply a dictionary-based approach to extract sentiment and uncertainty signals from news text. This method uses a predefined list of uncertainty-related keywords to quantify policy uncertainty levels in news articles over time.

### Econometric Modeling
We fit a linear regression model with lagged unemployment and EPU index as predictors, along with controls for GDP growth and inflation. The model is evaluated for predictive strength using adjusted R-squared and t-statistics.

## Data Requirements
- **Fama-French Three-Factor Monthly Returns**: Used to derive macroeconomic variables (GDP growth, inflation) and construct a regional unemployment proxy.
- **News Text Dataset**: Applied dictionary-based text mining to create an EPU index.

## Assumptions
- The Fama-French factors can serve as valid proxies for macroeconomic conditions.
- Dictionary-based text mining provides a reasonable approximation of policy uncertainty from news text.
- The temporal alignment between news text and macroeconomic data is sufficient for meaningful analysis.
- The relationship between policy uncertainty and unemployment is linear and stationary over the observed period.

## Implementation Details
The experiment follows these steps:
1. Load and preprocess Fama-French data to derive GDP growth and inflation proxies
2. Apply dictionary-based text mining on news text to compute EPU scores
3. Merge all time-series data to form a unified panel dataset
4. Construct two regression models: one with lagged unemployment as predictor, and another with EPU index as predictor
5. Compare adjusted R-squared, coefficient magnitudes, and statistical significance to evaluate which explains more variance in unemployment changes