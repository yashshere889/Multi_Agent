# Hypothesis H1: Nonlinear Dimensionality Reduction for Gene Expression Data

This experiment evaluates whether nonlinear dimensionality reduction techniques (t-SNE, LLE) can preserve sufficient signal for accurate tissue-type classification when reducing the feature space to less than 10% of the original dimensions, compared to linear methods like PCA.

## Assumptions

- The Hugging Face dataset `HuggingFaceM4/general-pmd-synthetic-testing-with-embeddings` was examined but did not contain suitable gene expression data matching the experiment's requirements (10,000 features, 500 samples, 5 tissue types). Therefore, synthetic data was generated instead.
- Synthetic gene expression data is generated with class-specific means and covariances to simulate biological variability and class separability.
- The experiment uses a controlled synthetic setup that enables reproducible and comparable results.
- t-SNE and LLE are applied to reduce dimensions to less than 10% of the original feature space.
- k-NN classification is used as the downstream task to assess signal preservation.

## Methods

1. **Principal Component Analysis (PCA)**: Linear dimensionality reduction technique used as a baseline.
2. **t-distributed Stochastic Neighbor Embedding (t-SNE)**: Nonlinear dimensionality reduction technique used to preserve local neighborhood relationships.
3. **Locally Linear Embedding (LLE)**: Nonlinear dimensionality reduction technique that preserves local geometric relationships.
4. **k-Nearest Neighbors (k-NN) Classifier**: Simple, non-parametric classifier used to evaluate classification accuracy on the reduced feature representations.

## Evaluation Metrics

- **Classification Accuracy**: Measured for each method and reduction ratio combination.
- **Success Criteria**: If t-SNE or LLE achieves classification accuracy within 5% of PCA at <10% reduction ratio, the hypothesis is supported.

## Implementation Details

The experiment compares the performance of PCA, t-SNE, and LLE on synthetic gene expression data with 10,000 features, 500 samples, and 5 tissue types. For each method, various reduction ratios are tested, including full dimensionality for PCA and <10% for t-SNE and LLE. Classification accuracy is measured for each method and ratio combination to assess signal preservation.