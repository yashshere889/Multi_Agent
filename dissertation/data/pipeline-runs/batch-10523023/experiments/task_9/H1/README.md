# Hypothesis H1: Demographic Features Improve Prediction Accuracy

This experiment tests whether adding demographic features (age, sex, race) to a clinical time-series model for predicting early patient deterioration improves prediction accuracy beyond what can be achieved with clinical features alone.

## Dataset

We use the UCI Adult Census Income dataset, which contains demographic information including age, sex, and race, along with income as a target variable. This dataset allows us to test whether demographic features provide additional predictive power beyond clinical features in a healthcare-like setting.

## Methodology

We perform an ablation study comparing two models:
1. **Clinical model**: Trained with only clinical features (age, education_num, hours_per_week)
2. **All features model**: Trained with all features including demographic variables (sex, race)

Both models use multivariate logistic regression. We use 5-fold cross-validation for both models and evaluate performance on a held-out test set.

## Evaluation Metrics

- Accuracy
- AUC-ROC

## Success Criteria

If the model with demographic features shows significantly better performance (higher accuracy and AUC-ROC), the hypothesis is refuted. If there is no significant difference in performance, the hypothesis is supported.

## Assumptions

- The UCI Adult Census Income dataset provides a suitable proxy for healthcare data
- Demographic features can be meaningfully combined with clinical features
- The preprocessing steps correctly handle the data characteristics
- The logistic regression models are appropriate for this classification task
- The cross-validation approach provides reliable estimates of model performance
- The difference threshold of 0.05 is appropriate for detecting meaningful improvements