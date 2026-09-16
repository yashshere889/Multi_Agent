# Experiment Plan: H1 - Sequence Length Effects in Transformers

This experiment investigates whether increasing sequence length beyond 512 tokens leads to diminishing returns in predictive accuracy for transformer models in text classification tasks.

## Objective
To determine whether increasing sequence length beyond 512 tokens leads to diminishing returns in predictive accuracy for transformer models in text classification tasks, and to assess whether model architecture choice has a greater impact on performance than sequence length.

## Methods
We conduct a factorial experiment comparing multiple transformer architectures (BERT, RoBERTa, DistilBERT) at various sequence lengths (128, 256, 512, 1024, 2048) on a text classification task using the 20 Newsgroups dataset.

## Data
The 20 Newsgroups dataset contains real newsgroup posts with 20 categories. Documents vary widely in length, making it suitable for exploring sequence length effects. The train set has 11014 rows and the test set has 7317 rows. Text is already tokenized and normalized.

## Assumptions
- The 20 Newsgroups dataset provides sufficient variation in document lengths to explore sequence length effects
- The experiment can be completed within the allocated computational resources
- The implemented transformer models are representative of the architectures mentioned in the plan

## Implementation Details
This experiment trains transformer models on the 20 Newsgroups dataset with varying sequence lengths and evaluates their performance using accuracy and F1-score metrics.