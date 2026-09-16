# Hypothesis H1: Feature Drift and Model Performance Decline

## Objective
To determine whether feature drift detected via Kolmogorov-Smirnov tests leads to a faster decline in model calibration compared to ranking discrimination over time in a consumer credit approval model.

## Methods

### Drift Detection Using Statistical Tests
We apply Kolmogorov-Smirnov (KS) tests to compare feature distributions between consecutive time slices to detect significant shifts. Features showing p < 0.05 are flagged as drifted.

### Dynamic Calibration Curve Analysis
We compute calibration plots and metrics (ECE - Expected Calibration Error) for each model trained on different time slices to quantify calibration quality over time.

### Logistic Regression
We train a logistic regression classifier on each time slice to predict credit risk. This serves as the base model for tracking both calibration and ranking performance.

## Design
A longitudinal experiment where a credit approval model is trained on sequential time slices of the UCI Statlog German Credit dataset. For each slice, we compute drift using Kolmogorov-Smirnov tests and track both calibration error and AUC over time. We compare the rate of change in these two metrics to test if calibration degrades faster than ranking discrimination due to drift.

## Data
The UCI Statlog German Credit dataset contains 1000 rows of consumer credit data with 20 features including categorical and numerical variables. It is ideal for studying drift effects because it has a natural temporal structure when sliced into time periods, and the target variable indicates credit risk (good/bad).

## Assumptions
- The dataset is available at `/mnt/fastscratch/users/sgyshere/coder-data/uci_statlog_german_credit.csv`
- The dataset has 1000 rows with 20 features
- The target variable is credit_risk with values 1 (good) and 2 (bad)
- We split the data chronologically into 10 time slices of 100 samples each
- We use Kolmogorov-Smirnov tests with a significance level of 0.05 to detect drift
- We calculate Expected Calibration Error (ECE) using sklearn's calibration_curve function
- We compute Area Under the ROC Curve (AUC) for ranking discrimination
- We assume that a faster decline in ECE compared to AUC supports our hypothesis
- We assume that a faster decline in AUC compared to ECE refutes our hypothesis
- We assume that similar rates of decline in both metrics would indicate no strong relationship between drift and performance degradation

## Evaluation Criteria
If the rate of increase in ECE exceeds the rate of decrease in AUC over time, then the hypothesis is supported. Conversely, if AUC declines faster than ECE, the hypothesis is refuted.