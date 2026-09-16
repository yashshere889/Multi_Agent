# Hypothesis H1: Spatial Lag Improves PM2.5 Prediction

This experiment evaluates whether incorporating a spatial lag model into a machine learning framework for predicting PM2.5 concentrations improves prediction accuracy compared to models using only temporal features.

## Dataset

The UCI Air Quality dataset provides hourly averaged responses from a gas multisensor device located in Italy between 2004 and 2005. It includes PM2.5 concentration measurements along with various temporal and environmental variables. The dataset also contains spatial coordinates (latitude and longitude) which are essential for constructing spatial weights matrices required for the spatial lag model.

## Methods

1. **Machine Learning Model (Random Forest)**: A Random Forest regressor is used as the base machine learning model to predict PM2.5 concentrations based on temporal features. This model is chosen for its robustness to overfitting and ability to handle mixed data types.

2. **Spatial Lag Model (SAR)**: A Spatial Autoregressive (SAR) model is implemented to incorporate spatial autocorrelation into the prediction framework. The spatial weights matrix is constructed using inverse distance weighting based on geographical coordinates to reflect spatial proximity between monitoring stations.

## Experimental Design

A comparative benchmark experiment where two models are trained and evaluated:
- One using only temporal features
- Another augmented with spatial features including a spatial lag term derived from spatial econometric methods

## Evaluation Metrics

Mean Absolute Error (MAE) is used to measure prediction accuracy.

## Success Criteria

If the MAE of the model incorporating spatial lag is lower than that of the temporal-only model, the hypothesis is supported. Conversely, if the MAE is higher or similar, the hypothesis is refuted.

## Assumptions

- The UCI Air Quality dataset contains sufficient spatial variation to detect meaningful spatial autocorrelation effects
- The spatial weights matrix calculation is computationally feasible within the available resources
- The synthetic spatial coordinates are representative enough for demonstrating the concept
- The Random Forest model is appropriate for this type of prediction task