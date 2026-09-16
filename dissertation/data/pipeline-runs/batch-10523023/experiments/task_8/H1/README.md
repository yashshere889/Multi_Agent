# Hypothesis H1: Bayesian Quantile Regression for VaR Estimation

This experiment tests whether quantile regression models estimated using the asymmetric Laplace distribution provide more accurate Value-at-Risk (VaR) estimates at the 5th percentile compared to Gaussian error models, as measured by lower pinball loss.

## How to Run

```bash
python run.py
```

Requirements:
- pandas
- numpy
- scipy
- scikit-learn

## Expected Output

The experiment outputs a `results.json` file containing:
- `bayesian_avg_pinball_loss`: Average pinball loss for Bayesian quantile regression
- `gaussian_avg_pinball_loss`: Average pinball loss for Gaussian error model  
- `difference`: Difference in average pinball loss (Bayesian - Gaussian)
- `meets_success_criteria`: Boolean indicating if hypothesis is supported
- `success_notes`: Explanation of results

## Interpretation

If `meets_success_criteria` is `true`, the hypothesis is supported - Bayesian quantile regression provided more accurate VaR estimates. If `false`, the hypothesis is refuted - Gaussian models performed better. If `unknown`, results were inconclusive.

## Assumptions

- Daily returns data from Fama-French factors represents systematic risk factors suitable for VaR modeling
- Rolling 252-day windows provide stable return characteristics for reliable estimation
- The 5th percentile VaR is appropriately modeled using either Bayesian quantile regression or Gaussian assumptions
- Pinball loss is an appropriate metric for comparing quantile forecasts
- All data preprocessing steps are correctly implemented according to the specification

## Method Details

### Bayesian Quantile Regression
Implements a simplified version of Bayesian quantile regression using the asymmetric Laplace distribution. In a full implementation, this would involve sampling from the posterior distribution using MCMC or variational inference techniques.

### Gaussian Error Model  
Uses standard linear regression with normally distributed errors to estimate VaR at the 5th percentile by computing mean and standard deviation of returns and applying the inverse normal CDF.

### Pinball Loss
Calculated as: (1/2) * sum(max(0, τ*(y - y_pred), (1-τ)*(y_pred - y))) where τ=0.05 for 5th percentile VaR.