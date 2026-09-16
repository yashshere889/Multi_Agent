# Comparative Benchmark Experiment: GNN vs Fingerprint Models for Solubility Prediction

This experiment compares the predictive accuracy of a Graph Neural Network (GNN) model with attention mechanisms against a fingerprint-based model on the MoleculeNet ESOL dataset, using the same number of trainable parameters.

## Objective
Determine whether a GNN model using SMILES representations and attention mechanisms achieves statistically significant higher predictive accuracy for compound solubility compared to a fingerprint-based model with an equal number of trainable parameters, as measured by RMSE on the MoleculeNet ESOL dataset.

## Methods

### Graph Neural Networks (GNNs)
A GNN model using SMILES representations and attention mechanisms to predict compound solubility. The model processes molecular structures as graphs with atoms as nodes and bonds as edges, applying message passing and attention mechanisms to learn molecular features.

### Molecular Fingerprints
A fingerprint-based model using Morgan fingerprints as input features to predict compound solubility. This serves as the baseline model for comparison with the GNN approach.

### Attention Mechanisms
Attention mechanisms integrated into the GNN model to improve interpretability and performance by focusing on relevant parts of the molecular graph during prediction.

### SMILES Representation
SMILES strings are used as the primary input format for the GNN model, converting molecular structures into a format suitable for graph-based processing.

## Data
The MoleculeNet ESOL dataset provides real-world measured solubility values and SMILES representations suitable for both model types. It contains 1128 compounds with measured log solubility in mols per litre and SMILES strings for molecular structure representation.

## Experimental Design
Comparative benchmark experiment with two treatment groups: GNN model with attention mechanisms and fingerprint-based model, each constrained to have equal trainable parameters. The models are trained and evaluated on the same dataset split to ensure fair comparison.

## Evaluation Metrics
Root Mean Square Error (RMSE) is used to measure predictive accuracy. The baseline model is the fingerprint-based model with equal number of trainable parameters.

## Success Criteria
If the GNN model achieves a lower RMSE than the fingerprint-based model, supporting the hypothesis. If the RMSE values are not significantly different, the hypothesis is refuted.

## Assumptions
- The MoleculeNet ESOL dataset is representative of the problem domain.
- Both models are trained and evaluated on the same data splits.
- The parameter count constraint is met through careful model architecture selection.
- The preprocessing steps are consistent across both models.