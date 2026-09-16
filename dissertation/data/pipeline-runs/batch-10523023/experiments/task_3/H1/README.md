# Credit Approval Fraud Detection Experiment (H1)

This experiment tests Hypothesis H1: "Applying SMOTE oversampling reduces fraud-detection recall at a fixed false-positive budget compared to using class-weighted models or decision-threshold tuning in a credit approval dataset with class imbalance."

## Dataset

The experiment uses the UCI Credit Approval dataset (`uci_credit_approval.csv`), which contains:
- 690 samples
- Binary target variable: approved ('+') or rejected ('-') 
- The minority class ('-') represents fraudulent/rejected applications
- Features include categorical and numerical attributes

## Methods Compared

1. **SMOTE Oversampling**: Generates synthetic samples for the minority class before training
2. **Class-weighted Random Forest**: Adjusts the learning process to penalize misclassification of the minority class more heavily  
3. **Decision Threshold Tuning**: Trains a base classifier and adjusts the decision threshold to achieve a fixed false-positive rate

## Evaluation Metric

Recall at a fixed false-positive rate (FPR = 10%) on the held-out test set.

## Success Criteria

If SMOTE results in lower recall at the fixed FPR compared to the other two methods, the hypothesis is supported. Otherwise, it is refuted.

## Assumptions

- The dataset is representative of a fraud detection scenario
- The fixed FPR of 10% reflects realistic operational constraints
- All methods are evaluated on the same test set to ensure fair comparison
- The Random Forest model parameters are kept consistent across methods
- The SMOTE implementation generates high-quality synthetic samples
- The threshold tuning approach correctly identifies the optimal decision boundary
- The class-weighted approach appropriately balances the class distribution during training