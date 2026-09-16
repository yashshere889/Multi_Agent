# Hypothesis H1: Cross-Validation vs Test Set Performance Rankings

This experiment tests whether model performance rankings derived from fivefold cross-validation on the AG News training set differ significantly from those based on the held-out test set, as measured by Spearman rank correlation below 0.7.

## Methods

### Fivefold Cross-Validation
Implements stratified 5-fold cross-validation to estimate model stability and performance. Each fold trains on 4/5 of the data and validates on the remaining 1/5, maintaining class distribution across splits.

### Train-Test Split
Uses the predefined train/test split of the AG News dataset to evaluate model performance on a held-out test set, providing an independent estimate of generalization capability.

## Data

The AG News dataset contains 120,000 short news headlines and descriptions labeled into four categories (World, Sports, Business, Sci/Tech). This dataset is ideal for text classification tasks and provides sufficient size for robust cross-validation while maintaining a distinct test set for comparison.

## Implementation Details

We use a simple TF-IDF vectorizer with Logistic Regression classifiers for text classification. The experiment performs:
1. 5-fold cross-validation on training data
2. Single evaluation on held-out test set
3. Ranking models based on average CV performance and test set performance
4. Computing Spearman rank correlation between these rankings

## Expected Outcome

If the Spearman rank correlation between model rankings from 5-fold CV and the held-out test set is below 0.7, the hypothesis is supported. A correlation of 0.7 or higher would refute the hypothesis.