# Hypothesis H1: Ensemble Uncertainty Correlation Analysis

This experiment tests the hypothesis that ensemble-based uncertainty estimates from financial models using deep ensembles or bootstrap resampling show a stronger correlation with realized forecast error than with input variance, particularly when evaluated on daily stock return data from the Fama-French factors dataset.

## How to Run

```bash
python run.py
```

Requirements:
- Python 3.8+
- pandas
- numpy
- scikit-learn
- torch

## Expected Output

The experiment generates a results.json file containing:
- Correlation coefficients between ensemble spread and forecast error for both methods
- Correlation coefficient between input variance and forecast error
- Whether the success criteria are met

## Interpretation

Success criteria:
- If either bootstrap or deep ensemble shows a stronger correlation with forecast error than input variance, the hypothesis is supported
- If both methods show stronger correlation than input variance, the one with higher correlation supports the hypothesis
- If neither method shows stronger correlation, the hypothesis is refuted

## Assumptions

1. The Fama-French three-factor dataset provides sufficient data for meaningful ensemble analysis
2. Using Market Risk Premium as a proxy for returns is appropriate for this analysis
3. A 20-day rolling window for input variance captures relevant market volatility patterns
4. Both bootstrap resampling and deep ensembles provide valid uncertainty estimates
5. The linear regression model serves as a reasonable baseline for forecasting
6. The experiment uses a fixed number of bootstrap samples and neural networks to ensure reproducibility
7. The rolling window approach for calculating input variance is suitable for this type of analysis
8. The test set represents a reasonable sample for evaluating the correlation relationships
9. The early stopping mechanism prevents overfitting while allowing sufficient training
10. The correlation analysis is appropriate for assessing the relationship between uncertainty estimates and forecast errors
11. The experiment assumes that the financial data is stationary enough for meaningful correlation analysis over the time period covered
12. The experiment treats the Fama-French factors as exogenous variables that can be used for prediction
13. The experiment assumes that the ensemble methods are correctly implemented and provide meaningful uncertainty estimates
14. The experiment assumes that the input variance calculated using a rolling window is representative of the underlying market conditions
15. The experiment assumes that the linear regression model is appropriate for the data characteristics
16. The experiment assumes that the neural network architecture is suitable for the task at hand
17. The experiment assumes that the training process converges properly within the specified limits
18. The experiment assumes that the test set is representative of the overall data distribution
19. The experiment assumes that the correlation coefficients are meaningful measures of the relationships being tested
20. The experiment assumes that the ensemble methods are comparable in terms of computational complexity and effectiveness