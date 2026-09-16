# Experiment H1: Multi-task Learning for Credit Risk Prediction

## Objective
To determine whether incorporating Fama-French three-factor returns (Mkt-RF, SMB, HML) as auxiliary prediction targets in a multi-task learning model for credit default prediction improves out-of-sample AUC performance compared to a single-task model trained only on borrower-level features.

## Hypothesis
The multi-task model that simultaneously predicts credit default and the Fama-French factors will achieve higher AUC performance than a single-task model trained only on borrower-level features.

## Methods

### Multi-task Deep Learning
A neural network architecture with shared layers followed by task-specific heads for credit default prediction and the three Fama-French factors. The shared representation learns common patterns between tasks while task-specific layers adapt to each prediction goal.

### Deep Neural Networks (DNN)
A feedforward neural network with hidden layers to learn non-linear relationships between borrower features and credit risk, as well as between macroeconomic factors and credit risk.

## Data
- **Credit Risk Dataset**: UCI Statlog (German Credit Data) containing 1000 German credit records with 20 borrower-level features and a binary credit risk outcome (good/bad).
- **Macroeconomic Factors**: Fama-French three-factor daily returns (Mkt-RF, SMB, HML) - aligned with borrower data through date matching.

## Implementation Details
This experiment implements a multi-task deep learning model that shares representations between credit default prediction and the three Fama-French factors. The model uses a shared backbone of fully connected layers followed by separate heads for each task.

## Evaluation
Primary metric: Area Under the ROC Curve (AUC) on out-of-sample test data.

## Assumptions
- The Fama-French factors provide useful auxiliary information for credit risk prediction
- The temporal alignment between borrower data and macroeconomic factors is sufficient for meaningful joint learning
- The multi-task model can effectively learn shared representations without negative transfer

## Success Criteria
If the multi-task model achieves higher AUC than the single-task model on out-of-sample test data, the hypothesis is supported. If the AUC is equal or lower, the hypothesis is refuted.