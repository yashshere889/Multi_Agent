# Temporal Resolution Impact on Forecasting Accuracy

This experiment tests whether increasing the temporal resolution of input measurements from hourly to 15-minute intervals improves forecasting accuracy by at least 5% as measured by MAE when using a Temporal Fusion Transformer (TFT) model for short-horizon electricity demand forecasting (1-hour to 6-hour ahead).

## Assumptions

- The UCI Air Quality dataset serves as a proxy for electricity demand drivers
- The Temporal Fusion Transformer model is implemented correctly according to the literature
- The experiment can be completed within the allocated compute resources
- The 15-minute resolution data is properly resampled without introducing future information leakage

## Methods

### Temporal Fusion Transformer (TFT)

This experiment implements the Temporal Fusion Transformer as described in the literature. It is a transformer-based model designed for time series forecasting that handles multiple temporal hierarchies and incorporates static and dynamic covariates. The model uses a transformer encoder with attention mechanisms to process sequential data and make multi-step forecasts.

### Data Preprocessing

1. Load the UCI Air Quality dataset (hourly resolution)
2. Parse Date and Time columns into datetime format
3. Resample the data to 15-minute intervals using forward-fill for missing values
4. Create two datasets: one with original hourly resolution and one with 15-minute resolution
5. Select relevant variables (CO(GT), NOx(GT), T, RH) as input features
6. Normalize all numerical features to [0,1] range for TFT compatibility
7. Split each dataset into train/validation/test sets (70/15/15%) maintaining temporal order

### Experimental Design

Control vs Treatment Group Comparison:
- Control group: TFT model trained on hourly resolution data
- Treatment group: TFT model trained on 15-minute resolution data
- Both models use identical architecture and hyperparameters
- MAE of forecasts from both models are compared to determine if the higher resolution improves accuracy by at least 5%

## Success Criteria

If the MAE of the model using 15-minute resolution data is at least 5% lower than the MAE of the model using hourly resolution data, the hypothesis is supported. If the improvement is less than 5% or negative, the hypothesis is refuted.

## Results

The experiment compares the mean absolute error (MAE) of predictions made by two identical TFT models trained on different temporal resolutions of the same dataset.