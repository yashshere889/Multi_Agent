# Experiment H1: Compression Rate Effects on Readability and Factual Accuracy

## Objective
To determine whether increasing the compression rate (ratio of summary length to source document length) in news articles from the AG News corpus leads to a significant decrease in readability scores (Flesch-Kincaid Grade Level) and a moderate negative correlation with factual accuracy (zero-shot faithfulness evaluation).

## Methods

### Readability Indices
We use the Flesch-Kincaid Grade Level score to quantify readability. This index evaluates text complexity based on average sentence length and syllable count per word.

### Zero-Shot Faithfulness Evaluation
We apply a zero-shot faithfulness evaluation using a pre-trained summarization model (Facebook BART-large-CNN) to assess factual consistency of truncated summaries against the original article. This method evaluates whether the summary accurately reflects information present in the source without requiring labeled training data.

## Design
A controlled experiment varying compression rates across a range of news articles from the AG News corpus. For each article, multiple truncated versions are generated at different compression levels (0.25, 0.5, 0.75). Each truncated version is scored for readability (Flesch-Kincaid Grade Level) and factual accuracy (zero-shot faithfulness).

## Data Requirements
- Source: `ag_news_document_text_category_classification_test.csv`
- Description: A collection of 7600 real news articles from the AG News dataset, categorized into four domains (World, Sports, Business, Sci/Tech). Each article includes the full text and word count.

## Assumptions
- The AG News corpus provides real news articles with sufficient length for meaningful truncation
- The Flesch-Kincaid Grade Level is an appropriate measure of readability for short news articles
- The zero-shot faithfulness evaluation using BART can provide reliable estimates of factual accuracy
- Articles with fewer than 20 words are filtered out to ensure meaningful truncation

## Success Criteria
Supporting the hypothesis would show a statistically significant negative trend in Flesch-Kincaid Grade Level with increasing compression rate, and a moderate negative correlation with zero-shot faithfulness scores. Refuting the hypothesis would mean observing no clear trend or even positive trends in either metric.