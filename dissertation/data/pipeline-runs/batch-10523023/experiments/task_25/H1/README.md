# Hypothesis H1 Experiment

This experiment tests whether sparse fine-tuning on multilingual pretrained language models (mPLMs) enables effective cross-lingual sentiment classification in low-resource languages with as little as 10% of the in-language training data required by full fine-tuning, while maintaining performance comparable to full fine-tuning on high-resource languages.

## Methods

### Multilingual Pretrained Language Models (mPLMs)
We use a multilingual pretrained language model (mBERT) as the base model for cross-lingual transfer. The model is initialized with pre-trained weights and fine-tuned on the AG News dataset to classify sentiment. This method is reused from literature and is suitable for zero-shot and few-shot cross-lingual transfer.

### Composable Sparse Fine-Tuning
We apply sparse fine-tuning techniques to update only a small subset of model parameters during training. This method is reused from literature and aims to reduce the need for large in-language training datasets by focusing on key parameters that contribute most to performance.

## Data

The experiment uses the AG News dataset, which contains 120,000 short news articles across 4 categories (World, Sports, Business, Sci/Tech). For this experiment, we filter for the 'Business' category to create a binary sentiment classification task (positive/negative).

## Design

This is a controlled comparative experiment with two groups:
1. Full fine-tuning on 100% of the training data (high-resource baseline)
2. Sparse fine-tuning on 10% of the training data (low-resource condition)

Performance is compared against full fine-tuning on 100% of the data (high-resource baseline).

## Assumptions

- The AG News dataset provides sufficient labeled examples for both high-resource and low-resource simulations
- The multilingual BERT model can effectively transfer knowledge across languages
- Sparse fine-tuning can maintain performance with reduced training data
- The experiment will complete within the allocated wall-clock budget
- The model will converge within 2 epochs with the specified learning rate and batch size

## Evaluation Metrics

- Accuracy
- F1 Score (weighted average)

## Success Criteria

If sparse fine-tuning achieves performance (accuracy and F1-score) comparable to full fine-tuning on 100% of the data when trained on only 10% of the data, the hypothesis is supported. If sparse fine-tuning performs significantly worse than full fine-tuning on 100% of the data, the hypothesis is refuted.