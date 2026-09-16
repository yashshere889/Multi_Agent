# Agricultural Yield Prediction Experiment (H1)

This experiment evaluates whether gradient-boosted models trained on historical weather records achieve higher seasonal crop yield prediction accuracy than a simple degree-day growth model.

## Objective
To determine whether gradient-boosted models trained on historical weather records achieve higher seasonal crop yield prediction accuracy than a simple degree-day growth model when evaluated on a real-world agricultural dataset with diverse climatic conditions and crop types.

## Methods

### Machine Learning Models (Gradient Boosted Trees)
A gradient-boosted model (specifically XGBoost) that learns complex non-linear relationships between weather variables and crop yields. Features include temperature, precipitation, humidity, and other climatic indicators. The model is trained to predict crop yield based on historical weather records.

### Glacio-hydrological Degree-day Model (GDM)
A simple degree-day model that calculates crop growth based on accumulated temperature above a base threshold. This serves as the baseline model for comparison. It uses temperature data to estimate crop development stages and final yield.

## Data Requirements
The experiment uses a synthetic agricultural yield dataset generated to mimic real-world conditions with diverse crop types, climatic conditions, and historical weather records. This includes features like temperature, precipitation, humidity, and soil conditions, along with corresponding crop yield values.

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

## Success Criteria
If the gradient-boosted model achieves lower MAE, RMSE, and higher R² scores compared to the degree-day model, the hypothesis is supported. If the degree-day model performs equally well or better, the hypothesis is refuted.

## Assumptions
- The synthetic dataset adequately represents real-world agricultural conditions for the purposes of this experiment
- The degree-day model implementation provides a reasonable baseline for comparison
- Cross-validation provides sufficient evidence for model comparison
- The dataset contains enough samples to support reliable model training and evaluation