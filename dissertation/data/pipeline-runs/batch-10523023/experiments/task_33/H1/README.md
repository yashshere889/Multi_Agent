# Hypothesis H1: Temporal Data Augmentation Performance Drop

## Objective
Test whether applying temporal data augmentation techniques to a tabular dataset with inherent time ordering results in a performance degradation of at least 20% when evaluated using strict chronological splits compared to standard (non-temporal) evaluation splits.

## Dataset
Uses the UCI Statlog German Credit dataset containing 1000 rows with temporal metadata (duration_months, credit_amount, age_years) and a binary target (credit_risk).

## Methods
Three temporal data augmentation techniques are applied:
1. **Time Warping**: Applies time warping to temporal features to introduce variability while preserving structure
2. **Jittering**: Adds small random noise to temporal features to simulate measurement errors
3. **SMOTE**: Synthetic Minority Oversampling Technique to balance classes in the dataset

## Experimental Design
- Two train/test splits: standard random splitting and strict chronological splitting
- Baseline classifiers trained on both splits without augmentation
- Augmented classifiers trained on both splits with each augmentation technique
- Performance gain calculated as difference in accuracy between augmented and non-augmented models
- Comparison of performance gains between standard and chronological splits

## Success Criteria
If the performance gain from augmentation drops by at least 20% when evaluated using strict chronological splits compared to standard splits, the hypothesis is supported.