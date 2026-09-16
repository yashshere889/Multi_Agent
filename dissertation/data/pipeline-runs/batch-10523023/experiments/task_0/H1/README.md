# Hypothesis H1: Transformer Models Outperform ARIMA in Volatility Forecasting

This experiment tests whether transformer-based sequence models (PatchTST-lite and Informer) outperform ARIMA models in forecasting daily volatility of equity index returns when transaction costs are explicitly incorporated into the performance metric using a Sharpe ratio adjusted for turnover and proportional transaction costs.

## How to Run

```bash
python run.py
```

## Expected Output

The experiment will produce a `results.json` file containing:
- MSE values for each model type
- Adjusted Sharpe ratios for each model type
- Whether the success criteria are met
- Training history for both models

## Interpretation

If the transformer models achieve significantly higher adjusted Sharpe ratios and/or lower MSE in volatility forecasts compared to ARIMA models, the hypothesis is supported. Conversely, if ARIMA models perform better or similarly, the hypothesis is refuted.

## Assumptions

- The Fama-French three-factor dataset provides sufficient historical data for training and validating forecasting models across multiple market conditions.
- Transaction costs are modeled as a fixed percentage (0.1%) of portfolio turnover.
- The volatility forecasting task is appropriately framed as a regression problem.
- The transformer models are properly configured with appropriate sequence lengths and embedding dimensions.
- The ARIMA models are simulated using a simple moving average approach as a proxy for full implementation.
- The Sharpe ratio adjustment accounts for both turnover and proportional transaction costs.

## Requirements

No additional packages required beyond standard Python libraries.