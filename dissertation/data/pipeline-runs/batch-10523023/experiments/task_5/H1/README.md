# Hypothesis H1: Market Regime Detection Improves Momentum Strategy Performance

This experiment tests whether using spectral clustering to identify market regimes from daily S&P 500 price and volume features improves the out-of-sample Sharpe ratio of a simple momentum strategy compared to a static momentum rule.

## How to Run

```bash
python run.py
```

Requirements:
- pandas
- scikit-learn
- numpy

## Expected Output

The experiment outputs a `results.json` file containing:
- Sharpe ratios for both strategies
- Annualized returns for both strategies  
- Maximum drawdowns for both strategies
- Whether the hypothesis is supported or refuted

## Interpretation

If the regime-aware momentum strategy achieves a higher out-of-sample Sharpe ratio than the static momentum strategy, the hypothesis is supported. Otherwise, it is refuted.

## Assumptions

- The S&P 500 data contains sufficient information to identify meaningful market regimes through spectral clustering
- A simple momentum strategy can be effectively implemented using daily returns and rolling lookbacks
- The regime-aware strategy benefits from adjusting position sizes based on detected market conditions
- Transaction costs and slippage are negligible for this analysis
- The data is representative of the broader market conditions over the analyzed period

## Method Details

### Spectral Clustering
We apply spectral clustering to group days into distinct market regimes based on normalized price-volume features. Using the graph Laplacian to find low-dimensional embeddings and KMeans clustering on the embedding space to identify 3-5 market states (bullish, bearish, neutral). The number of clusters is selected using silhouette analysis.

### Momentum Strategy
We implement a simple momentum strategy that buys assets with highest past returns and sells those with lowest returns. For the regime-aware version, we adjust momentum weights based on the detected market regime (e.g., higher exposure during bullish regimes).

## Data Source

The experiment uses the historical stock market daily closing price financial data from `/mnt/fastscratch/users/sgyshere/coder-data/historical_stock_market_daily_closing_price_financial_data.csv`.
This dataset provides daily S&P 500 index levels and related financial indicators including Real Price, Real Dividend, Real Earnings, PE10, Consumer Price Index, Long Interest Rate.