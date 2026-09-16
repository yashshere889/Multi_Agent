# Experiment H1: Calibration Performance on Imbalanced Data

This experiment tests hypothesis H1: "To determine whether probability calibration methods (Platt Scaling, Isotonic Regression) exhibit significantly higher Expected Calibration Error (ECE) when applied to heavily imbalanced real-world tabular data (UCI Credit Approval, UCI German Credit) compared to their performance on balanced benchmarks (AG News, 20 Newsgroups)."

## How to Run

```bash
python run.py
```

Requirements:
- pandas
- scikit-learn
- numpy

## Output Interpretation

The experiment outputs ECE values for:
1. Uncalibrated predictions
2. Platt Scaling calibration
3. Isotonic Regression calibration

These values can be compared against ECE values from balanced benchmarks to determine if the hypothesis is supported.

## Assumptions

- The UCI Credit Approval dataset represents a typical imbalanced real-world scenario
- Using 3-fold cross-validation for calibration is sufficient for this experiment
- Binning approach with 10 bins is adequate for ECE calculation
- The comparison is made within the same dataset (imbalanced) rather than against balanced benchmarks as originally planned, since only the imbalanced dataset is provided
- Logistic regression is an appropriate baseline model for this type of data
- The experiment focuses on demonstrating the effect of calibration on imbalanced data rather than direct comparison with balanced benchmarks

## Methodology

This experiment follows the design outlined in the plan:
1. Loads and preprocesses the UCI Credit Approval dataset
2. Trains a logistic regression classifier
3. Applies Platt Scaling and Isotonic Regression calibration
4. Computes Expected Calibration Error (ECE) for each method
5. Reports ECE values for comparison

The ECE is calculated using the standard binning approach, where predictions are grouped into bins based on their predicted probability, and the difference between average accuracy and average confidence within each bin is computed.