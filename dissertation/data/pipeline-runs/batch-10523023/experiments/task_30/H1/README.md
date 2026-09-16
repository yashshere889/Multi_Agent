# Physics-Informed Neural Networks for Molecular Solubility Prediction

## Objective
To empirically demonstrate that Physics-Informed Neural Networks (PINNs) with physical loss terms achieve better extrapolation performance on molecular solubility data compared to standard neural networks trained without physical constraints.

## Methods

### Physics-Informed Neural Networks (PINNs)
A neural network architecture modified to incorporate physical laws directly into the loss function. In this experiment, we implement a PINN that includes a loss term representing the physical constraints of solubility behavior, such as thermodynamic consistency or molecular interaction principles. This is done by adding a penalty term to the standard mean squared error that enforces adherence to known physical relationships.

### Standard Neural Network
A conventional feedforward neural network trained using standard mean squared error loss without any physical constraints. This serves as our baseline model for comparison with the PINN approach.

## Data
The experiment uses the MoleculeNet ESOL (Delaney) dataset containing:
- 1128 molecular entries
- Molecular descriptors including: Minimum Degree, Molecular Weight, Number of H-Bond Donors, Number of Rings, Number of Rotatable Bonds, Polar Surface Area
- Target variable: measured log solubility in mols per litre

## Experimental Design
We train two types of models:
1. Standard neural networks without physical constraints
2. Physics-Informed Neural Networks with embedded physical loss terms

Both models are trained on the same molecular solubility dataset, with the PINN incorporating a loss term derived from the underlying physical principles governing solubility. We then evaluate both models on out-of-distribution test points to measure extrapolation performance.

## Evaluation Metrics
- Mean Squared Error (MSE) on out-of-distribution test points

## Success Criteria
If the PINN achieves a lower MSE on out-of-distribution test points than the standard neural network, this supports the hypothesis that physics-informed loss terms improve extrapolation performance. Conversely, if the standard network performs better or similarly, the hypothesis is refuted.

## Assumptions
- The physical constraints incorporated into the PINN loss function are meaningful for solubility prediction
- The dataset provides sufficient variation to create meaningful out-of-distribution test cases
- The PINN implementation properly enforces physical constraints through its loss function