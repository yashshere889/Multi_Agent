# Experiment H1: Document Length and Vocabulary Richness Effects on BERT Classification Accuracy

## Objective
To empirically test whether increasing document length beyond 300 words significantly improves classification accuracy compared to using vocabulary-rich shorter documents with a Type-Token Ratio (TTR) above 0.75 when both are processed through a standard BERT model.

## Methods

### Type Token Ratio (TTR)
Calculates vocabulary richness by dividing the number of unique tokens by the total number of tokens in a document. Used to identify vocabulary-rich shorter documents (TTR > 0.75) versus less rich longer documents (TTR <= 0.75).

### BERT Model for Classification
Uses a standard BERT-base model fine-tuned on the 20 Newsgroups dataset to classify documents into their respective categories. The same model architecture is applied consistently across all experimental conditions to ensure fair comparison.

## Experimental Design
Controlled comparative experiment with three treatment groups:
1. **Short vocabulary-rich**: documents truncated to 300 words or less with TTR > 0.75
2. **Long low-vocabulary**: documents extended to exceed 300 words but with TTR <= 0.75  
3. **Original vocabulary-rich**: original-length documents with TTR > 0.75

All groups are classified using the same BERT model to isolate the effect of document length vs. vocabulary richness.

## Data
The 20 Newsgroups dataset contains real newsgroup posts with varying document lengths (median 86 words, p90 342, max 11765), providing ideal variation for testing document length effects. Documents are categorized into 20 classes, enabling meaningful classification accuracy measurement.

## Assumptions
- The 20 Newsgroups dataset provides sufficient variation in document length and vocabulary richness
- BERT model can be effectively fine-tuned on the provided dataset sizes
- Statistical significance testing will properly distinguish between the experimental groups
- The TTR calculation accurately reflects vocabulary richness

## Success Criteria
If the difference in classification accuracy between the vocabulary-rich short documents (TTR > 0.75, ≤300 words) and the long documents with low vocabulary richness (TTR ≤ 0.75, >300 words) is statistically insignificant (p > 0.05), the hypothesis is supported. If the long documents with low vocabulary richness show significantly lower accuracy, the hypothesis is refuted.