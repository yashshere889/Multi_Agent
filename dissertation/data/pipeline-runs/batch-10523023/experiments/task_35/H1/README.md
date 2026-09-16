# Comparative Benchmark Experiment: GAM vs Ensemble Methods

This experiment evaluates whether Generalized Additive Models (GAMs) can achieve classification accuracy within 5% of that attained by ensemble methods (Random Forest and Gradient Boosting) when both are optimized with equal computational budgets on the UCI Adult Census Income dataset.

## Assumptions

- The UCI Adult Census Income dataset is available at `/mnt/fastscratch/users/sgyshere/coder-data/uci_adult_census_income.csv`.
- The dataset contains demographic and employment information for individuals, with a binary target variable indicating whether their income exceeds $50K/year.
- The experiment compares classification accuracy under equal computational budgets.
- All models are trained and evaluated using the same preprocessing pipeline and train/test split.
- The computational budget for hyperparameter optimization is fixed at 100 iterations for each model type.

## Methods

### Generalized Additive Models (GAMs)
We use the pygam library to fit GAMs to the preprocessed UCI Adult Census Income dataset. GAMs allow for non-linear relationships through smooth functions of predictors.

### Ensemble Methods
Specifically Random Forest and Gradient Boosting machines, which are known for strong performance on tabular data. We use scikit-learn implementations for both models.

### Hyperparameter Optimization
Bayesian optimization using scikit-optimize to tune hyperparameters for each model type within a fixed computational budget. For GAMs, we optimize smoothing parameters and regularization terms. For ensembles, we tune n_estimators, max_depth, learning_rate, etc.

## Evaluation Metrics

The primary metric is classification accuracy on the held-out test set. The success criterion is that GAM accuracy must be within 5% of the highest accuracy achieved by the ensemble methods.

## Implementation Details

The experiment follows these steps:
1. Load and preprocess the UCI Adult Census Income dataset
2. Split the dataset into training and testing sets (80/20 split)
3. Define a fixed computational budget (100 iterations of Bayesian optimization) for hyperparameter tuning
4. Implement hyperparameter optimization for GAMs using scikit-optimize
5. Implement hyperparameter optimization for Random Forest and Gradient Boosting using scikit-optimize
6. Train the optimized models and evaluate their classification accuracy
7. Compare the classification accuracies to determine if the hypothesis is supported

## Success Criteria

If the GAM achieves classification accuracy within 5% of the best-performing ensemble method (Random Forest or Gradient Boosting), the hypothesis is supported. If the GAM's accuracy falls outside this range, the hypothesis is refuted.