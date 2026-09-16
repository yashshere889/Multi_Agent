# Hypothesis H1: Clustering Performance Comparison

This experiment evaluates whether K-means clustering with PCA dimensionality reduction recovers income strata more consistently with predefined income thresholds than hierarchical clustering or fuzzy c-means methods.

## Dataset

The UCI Adult Census Income dataset contains demographic and economic features of individuals including:
- Age
- Education level
- Marital status
- Occupation
- Capital gains/losses
- Hours worked per week
- Income level ('>50K' or '<=50K')

## Methods

1. **K-means clustering** - Standard K-means algorithm with k=2 clusters
2. **K-means with PCA** - K-means applied to 2-component PCA-reduced features
3. **Hierarchical clustering** - Agglomerative clustering with ward linkage
4. **Fuzzy C-Means** - Simplified approximation using dual K-means clustering

## Evaluation Metrics

- Adjusted Rand Index (ARI) comparing cluster assignments to income thresholds
- Normalized Mutual Information (NMI) between cluster assignments and income categories
- Silhouette Score for cluster cohesion and separation
- Consistency ratio measuring alignment with income thresholds

## Assumptions

- The dataset contains sufficient separation between income groups to detect meaningful clusters
- PCA dimensionality reduction improves clustering performance for this dataset
- K-means with PCA will show higher ARI and NMI scores compared to other methods
- The income threshold ('>50K' vs '<=50K') serves as a valid ground truth for cluster alignment

## Expected Outcome

K-means with PCA is expected to outperform hierarchical clustering and fuzzy c-means in terms of ARI and NMI scores when evaluated against predefined income thresholds, demonstrating better recovery of income strata consistency.