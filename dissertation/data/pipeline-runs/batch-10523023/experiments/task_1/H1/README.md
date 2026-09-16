# Hypothesis H1: BERT-based Financial Sentiment Improves Next-Day Directional Prediction

This experiment tests whether a BERT-based financial sentiment model trained on news headlines improves next-day directional prediction of S&P 500 returns compared to a model using only Fama-French three-factor risk premiums, using Granger causality tests.

## Methods

### BERT-based Models
We use a pre-trained BERT model (bert-base-uncased) to generate synthetic sentiment scores for news headlines. Since we don't have a real financial news dataset with labeled sentiment, we generate synthetic sentiment labels based on the AG News headlines using a pre-trained BERT model. This allows us to test the core idea without requiring access to a large, labeled financial news dataset.

### Granger Causality Analysis
We apply Granger causality tests to determine whether past values of one variable (sentiment scores or Fama-French factors) help predict future values of another variable (next-day S&P 500 return direction). This involves fitting autoregressive models and testing statistical significance of lagged predictors.

## Data Requirements

- `ag_news_document_text_category_classification_train.csv`: AG News dataset containing real news headlines and descriptions. We use this to generate synthetic financial sentiment labels for news headlines, simulating a financial news dataset.
- `fama_french_three_factor_daily_returns.csv`: Fama-French three-factor daily returns dataset.

## Experimental Design

We compare two models:
1. BERT-based sentiment scores predicting next-day S&P 500 return direction
2. Fama-French three-factor risk premiums predicting next-day S&P 500 return direction

Both models are evaluated using Granger causality tests to measure predictive power.

## Assumptions

- The synthetic sentiment scores generated from AG News headlines can serve as a proxy for financial sentiment
- The Fama-French factors provide a robust baseline for financial market prediction
- Granger causality tests are appropriate for evaluating the predictive relationship between sentiment and market returns
- The alignment of dates between sentiment data and financial data is sufficient for this proof-of-concept experiment
- The time series properties of the data are suitable for Granger causality analysis
- The synthetic data generation approach produces realistic enough patterns for statistical testing

## Success Criteria

If the BERT-based sentiment model shows higher Granger causality F-statistics or greater directional prediction accuracy than the Fama-French model, the hypothesis is supported. If not, the hypothesis is refuted.