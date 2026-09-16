# Hypothesis H1 Experiment

This experiment tests the hypothesis: "To empirically test whether higher feature selection stability, as measured by the Consistency Index, correlates with higher downstream test accuracy in NLP text classification tasks using the AG News dataset."

## How to Run

```bash
python run.py
```

Requirements:
- pandas
- scikit-learn
- numpy

## Expected Output

The experiment outputs a JSON file with the following keys:
- `accuracy_low_consistency`: Test accuracy of classifier trained on low-consistency features
- `accuracy_high_consistency`: Test accuracy of classifier trained on high-consistency features  
- `accuracy_full_features`: Test accuracy of classifier trained on all features
- `consistency_low`: Mean consistency score for low-consistency features
- `consistency_high`: Mean consistency score for high-consistency features
- `correlation_coefficient`: Pearson correlation coefficient between consistency scores and accuracies
- `meets_success_criteria`: Boolean indicating whether the hypothesis is supported
- `success_notes`: Explanation of results

## Interpretation

If the correlation coefficient is positive and statistically significant, it supports the hypothesis that more stable features lead to better classification performance. If negative or near-zero, it refutes the hypothesis.

## Assumptions

- The AG News dataset is representative of typical NLP text classification tasks
- Feature selection stability measured by consistency across bootstrap samples is a valid proxy for feature importance
- Logistic regression provides a reasonable baseline for comparison
- The test set remains independent and representative throughout the experiment
- Consistency index calculation using frequency of feature appearance across resampled datasets is appropriate for this context
