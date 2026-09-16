# Active Learning vs Random Sampling for Materials Property Prediction

This experiment evaluates whether active learning with uncertainty sampling based on Gaussian Process Regression reduces the number of required simulations by at least 30% compared to random sampling when mapping materials property landscapes on the MoleculeNet ESOL dataset.

## Methods

### Active Learning
An iterative process where a model is trained on an initial subset of data, then selects the most informative samples for labeling based on uncertainty measures before retraining. This continues until convergence or a maximum number of iterations is reached.

### Uncertainty Sampling
A query strategy within active learning that selects samples with highest prediction uncertainty (e.g., highest variance from Gaussian Process Regression) for labeling. This helps prioritize informative samples that will most improve model performance.

### Gaussian Process Regression
A probabilistic regression model that provides uncertainty estimates along with predictions. In this experiment, it serves as both the predictive model and the basis for uncertainty sampling decisions.

## Dataset
The MoleculeNet ESOL dataset provides 1128 compounds with molecular descriptors and measured log solubility values. This dataset is suitable for testing active learning strategies in materials property prediction, with the measured solubility serving as the ground truth target for model evaluation.

## Experimental Design
Two treatment groups are compared:
1. Active learning with uncertainty sampling using Gaussian Process Regression
2. Random sampling

Both strategies are applied iteratively to select molecules for solubility prediction until a convergence criterion is met. The number of simulations required for each approach is recorded and compared.

## Success Criteria
If the active learning approach requires at least 30% fewer simulations than random sampling to achieve equivalent prediction accuracy, the hypothesis is supported. If the reduction is less than 30%, the hypothesis is refuted.

## Assumptions
- The MoleculeNet ESOL dataset is representative of materials property landscapes
- Gaussian Process Regression provides reliable uncertainty estimates
- Convergence is determined by a small change in validation MSE
- The initial training set size is sufficient for both approaches
- The maximum iterations limit prevents infinite loops