# Hypothesis H1: Centrality Measures vs Bilateral Trade Volume for Predicting Post-Shock GDP Growth

## Objective
To determine whether centrality measures derived from the distance backbone of a weighted international trade network predict post-shock GDP growth rates more accurately than bilateral trade volume alone.

## Methods
This experiment implements a comparative benchmark study using two predictive models:
1. **Centrality-based model**: Uses centrality measures (degree, betweenness, closeness, eigenvector, PageRank) from the distance backbone of the trade network
2. **Volume-based model**: Uses bilateral trade volumes between countries as features

Both models are trained using linear regression and evaluated using Mean Absolute Error (MAE).

## Data Requirements
The experiment uses the Hugging Face dataset `lukesjordan/worldbank-project-documents` as a proxy for trade data, since the actual trade data requirements (World Bank Open Data API or UN Comtrade database) are not available. The dataset contains project information documents from World Bank projects.

## Assumptions
- The Hugging Face dataset contains sufficient information to extract country names for constructing trade networks
- Synthetic trade data can be reasonably generated to represent international trade relationships
- GDP growth data can be simulated to represent post-shock economic impacts
- Economic shocks (2008 financial crisis, 2020 pandemic) can be modeled through simulated GDP growth changes

## Implementation Details
- Distance backbone extraction is implemented as a simplified filtering of edges based on weight thresholds
- Network centrality measures are computed using simplified approximations
- Linear regression models are trained with early stopping and gradient clipping to prevent numerical overflow
- Model comparison is performed using MAE as the primary evaluation metric

## Success Criteria
If the model using centrality measures from the distance backbone achieves lower MAE than the bilateral trade volume model, the hypothesis is supported. If the reverse occurs, the hypothesis is refuted.