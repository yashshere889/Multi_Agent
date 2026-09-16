# Neural Topic Models vs LDA for Interdisciplinary Research Detection

This experiment evaluates whether Neural Topic Models (NTMs) with variance-invariance-covariance regularization detect emerging interdisciplinary research areas earlier than traditional Latent Dirichlet Allocation (LDA) models.

## Methodology

We use the 20 Newsgroups dataset containing scientific discussions to train both LDA and NTM models. The NTM incorporates regularization that encourages variance-invariance-covariance properties in the latent topic representations, potentially improving topic coherence and generalization.

## Key Components

### Models Implemented
1. **Latent Dirichlet Allocation (LDA)** - Standard probabilistic topic model
2. **Neural Topic Model (NTM) with regularization** - Enhanced neural model with variance-invariance-covariance regularization

### Evaluation Metrics
- **Mean Absolute Error (MAE)** between actual topic shift times and simulated citation peak times
- **Precision at k (P@k)** for detecting topic shifts within a window of citation peaks  
- **Recall at k (R@k)** for detecting topic shifts within a window of citation peaks

### Citation Peak Simulation
Since real citation data isn't available, we simulate citation peaks by:
1. Tracking topic prevalence over time
2. Identifying periods of rapid increase in topic weights
3. Defining these as citation peaks

## Assumptions

- The 20 Newsgroups dataset contains sufficient scientific content to detect interdisciplinary trends
- Simulated citation peaks accurately represent the temporal dynamics of topic shifts
- The variance-invariance-covariance regularization in NTMs improves topic detection capabilities
- Topic prevalence changes correlate with citation patterns in scientific literature

## Expected Outcome

If NTM with regularization shows significantly lower MAE and higher P@k/R@k values compared to LDA, it supports the hypothesis that NTMs detect emerging interdisciplinary research areas earlier. Conversely, if there's no significant difference or if LDA performs better, it refutes the hypothesis.