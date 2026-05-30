# Traffic Demand Prediction using Ensemble Learning

## Overview

This project focuses on predicting normalized traffic demand using advanced machine learning techniques on spatio-temporal traffic data.

The solution was developed for a competitive hackathon setting and includes:

- Smart missing value imputation
- Leakage-safe target encoding
- Feature engineering
- GPU-accelerated boosting
- Ensemble optimization
- Leaderboard-aware experimentation

The final solution achieved a Public Leaderboard Score of 91.16918.

---

# Problem Statement

The objective is to predict traffic demand using:
- geospatial information
- weather conditions
- road characteristics
- temporal traffic patterns

The target variable:

```python
demand
```

represents normalized traffic demand values between 0 and 1.

---

# Dataset Information

## Training Dataset

- Rows: 77,299
- Features:
  - Geohash
  - RoadType
  - NumberofLanes
  - Weather
  - Temperature
  - Landmarks
  - LargeVehicles
  - Timestamp/Day

## Test Dataset

- Rows: 41,778

---

# Exploratory Data Analysis

Key findings:

- Highly skewed target distribution
- Strong temporal traffic patterns
- Strong geospatial dependency
- Long-tail traffic demand behavior
- Significant interaction between location and traffic density

### Target Statistics

| Metric | Value |
|---|---|
| Mean | ~0.094 |
| Skewness | ~3.73 |
| Kurtosis | ~17.33 |

---

# Data Preprocessing

## Smart Missing Value Imputation

Instead of naive:
- median filling
- mode filling

context-aware imputation was used.

### Temperature Imputation

Used:
- same geohash
- same weather
- nearby timestamps

### Weather Imputation

Used:
- geohash
- temperature
- timestamp

### RoadType Imputation

Used:
- NumberofLanes
- road characteristics
- nearby road patterns

---

# Feature Engineering

## Temporal Features

- hour
- hour_sin
- hour_cos
- is_peak_hour

## Spatial Features

- geohash frequency
- geohash prefixes
- density features

## Interaction Features

- road capacity
- landmark-road interactions
- weather-temperature interactions

## Density Features

- geo_hour_density
- geo_day_density

---

# Leakage Detection & Fixing

Initial models showed:
- very high local CV
- poor leaderboard performance

This led to discovery of target leakage.

Leaky features:

```python
geohash_demand_mean
hour_demand_mean
roadtype_demand_mean
day_demand_mean
```

were removed and replaced with:
- Out-of-Fold Target Encoding

---

# Target Encoding

Leakage-safe OOF target encoding was implemented for:

- geohash
- hour
- RoadType
- day

This significantly improved generalization.

---

# Models Used

## CatBoost
- Excellent categorical handling
- GPU accelerated

## XGBoost
- Strong nonlinear structured learning
- Best-performing individual model

## LightGBM
- Efficient leaf-wise boosting
- Strong ensemble diversity

---

# Ensemble Learning

Final weighted ensemble:

```python
final_preds = (

    0.05 * cat_preds +

    0.70 * xgb_preds +

    0.25 * lgb_preds
)
```

---

# Results

| Stage | Public LB Score |
|---|---|
| Initial CatBoost | 85.72 |
| Leakage-safe OOF Model | 86.66 |
| Final Weighted Ensemble | 91.16918 |

---

# Advanced Experiments

Several advanced experiments were explored:

## Attempted Improvements

- Stacking Ensemble
- Rank Averaging
- Log-Transform Target
- Bayesian Smoothed Target Encoding
- High-Cardinality Interaction TE
- XGB Hyperparameter Tuning

## Key Learning

Higher local CV did NOT always improve leaderboard score.

The competition strongly rewarded:
- robust generalization
- calibration
- smooth feature complexity

instead of:
- aggressive memorization
- overly specific interaction encoding

---

# Final Insights

The biggest improvement came from:

- fixing target leakage
- proper validation strategy
- calibrated ensemble learning

NOT from:
- deeper models
- excessive feature interactions
- complicated stacking

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- CatBoost
- XGBoost
- LightGBM
- Google Colab GPU (Tesla T4)

---

# Project Structure

```text
Traffic-Demand-Prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Missing_Value_Imputation.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Model_Training.ipynb
│
├── submissions/
│   ├── submission_v1.csv
│   ├── submission_final.csv
│
├── README.md
│
└── requirements.txt
```

---

# Future Improvements

Potential future work:

- Adversarial Validation
- Pseudo Labeling
- Better temporal validation
- Smoothed Bayesian encoding
- Graph-based traffic modeling
- Deep learning for spatio-temporal forecasting

---

# Final Leaderboard Score

Public Score: 91.16918

---

# Author

Sahil Burnwal

Machine Learning | Ensemble Learning | Spatio-Temporal Modeling
