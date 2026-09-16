# Experiment for Hypothesis H1

This experiment tests the hypothesis: "To determine whether applying demographic parity versus equal opportunity fairness constraints to a credit approval model results in significantly different rejection patterns for male applicants when overall accuracy is held constant."

## How to Run

```bash
python run.py
```

Requirements:
- pandas
- scikit-learn
- numpy

## Expected Output

The experiment outputs a `results.json` file containing:
- Accuracy metrics for baseline and fairness-constrained models
- Rejection rates for male applicants under different fairness constraints
- True positive rate differences between genders
- Acceptance rate differences between genders
- Whether the hypothesis is supported or refuted

## Interpretation

The key metric is the difference in rejection rates for male applicants between demographic parity and equal opportunity constraints. If demographic parity leads to higher rejection rates for males, the hypothesis is supported. If not, it's refuted.

## Assumptions

- The UCI German Credit dataset contains sufficient samples to estimate fairness metrics reliably
- Gender is properly encoded as a binary indicator (male=1, female=0)
- The fairness constraints can be applied meaningfully to this dataset
- The simplified threshold adjustment methods provide reasonable approximations to the true fairness-constrained optimization problems
- The dataset's class imbalance (700 good / 300 bad) is handled appropriately through stratified sampling

## Method Details

This experiment implements fairness-constrained optimization using:
1. Demographic parity: Equal acceptance rates across genders
2. Equal opportunity: Equal true positive rates across genders

Both approaches maintain equal overall accuracy as specified in the hypothesis.