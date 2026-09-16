# Hypothesis H1: Gradient-Boosted Decision Trees Scaling Behavior

This experiment tests the hypothesis: "To determine whether gradient-boosted decision trees (XGBoost and LightGBM) maintain or improve their accuracy advantage over neural networks as the training set size increases from 1000 to 100,000 records on the UCI Statlog German Credit dataset."

## Running the Experiment

```bash
python run.py
```

Requirements:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm

## Output Interpretation

The experiment outputs a `results.json` file containing:
- Accuracy scores for each model at each training set size
- Baseline performance on the original 1000-record dataset
- Training histories for convergence analysis
- Whether the success criteria were met

Success criteria:
- Supporting the hypothesis means both XGBoost and LightGBM maintain or improve their accuracy advantage over neural networks as training set size increases
- Refuting the hypothesis occurs if neural networks show equal or greater accuracy than GBDTs at larger training set sizes

## Assumptions

- The UCI Statlog German Credit dataset (1000 rows) is representative of the problem domain
- Sampling with replacement from the original dataset adequately simulates larger datasets
- Default hyperparameters provide a fair comparison between models
- The 80/20 train/test split ensures reliable performance estimates
- All models are evaluated on the same test set for consistent comparison

The experiment uses a fixed set of training set sizes: [1000, 2000, 5000, 10000, 20000, 50000, 100000].