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

The final solution achieved a Public Leaderboard Score of 91.45390.

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

Leakage-safe Out-of-Fold (OOF) target encoding was implemented for:

- geohash
- hour
- RoadType
- day

The encodings are computed **inside each cross-validation fold** using only that
fold's training rows, then mapped onto the validation and test rows. Unseen
categories fall back to the global mean. Computing the encoding in-fold (rather than
once on the full training set) guarantees that no target information leaks from
validation into training, which significantly improved generalization.

---

# Models Used

The final solution trains three GPU-accelerated gradient boosting regressors inside a
shared 5-fold cross-validation loop. Each model uses **early stopping (200 rounds)**
so it selects its own optimal number of trees, and all models are trained on a
**log1p-transformed target** (predictions inverted with `expm1` and clipped at zero,
since demand is non-negative and right-skewed).

## CatBoost
- Native categorical handling via `cat_features` (its core strength on high-cardinality fields like geohash)
- 5000 iterations, learning rate 0.03, depth 8, l2_leaf_reg 3
- GPU accelerated

## XGBoost
- Strongest individual model on this data
- 5000 trees, learning rate 0.02, max_depth 8, subsample 0.8, colsample_bytree 0.8, min_child_weight 3, reg_lambda 1
- `hist` tree method on CUDA

## LightGBM
- Native categorical handling via `categorical_feature`; adds ensemble diversity
- 5000 trees, learning rate 0.02, num_leaves 63, subsample 0.8 (subsample_freq 1), colsample_bytree 0.8, reg_lambda 1, min_child_samples 20

Out-of-fold (OOF) predictions are stored for every model so the ensemble can be
**measured directly rather than guessed**.

---

# Ensemble Learning

Instead of hand-set blend weights, the ensemble weights are **optimised on the
out-of-fold predictions** using SLSQP (weights constrained to be non-negative and to
sum to 1) to maximise OOF R². The optimised weights are used only if they beat a
simple average on OOF, which guards against overfitting the blend. Final test
predictions are clipped at zero.

```python
from scipy.optimize import minimize

def neg_r2(w):
    return -r2_score(y, oof_stack.dot(w))

res = minimize(
    neg_r2,
    x0=[1/3, 1/3, 1/3],
    method='SLSQP',
    bounds=[(0, 1)] * 3,
    constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
)
weights = res.x   # used only if it beats the simple average on OOF
```

---

# Results

| Stage | Public LB Score |
|---|---|
| Initial CatBoost | 85.72 |
| Leakage-safe OOF Model | 86.66 |
| Fixed-weight Ensemble | 91.16918 |
| Optimised Ensemble (best) | 91.45390 |

---

# Advanced Experiments

Several advanced experiments were explored on top of the optimised ensemble.

## Improvements that worked

- Native categorical handling (CatBoost / LightGBM)
- Log-transform of the target
- Per-model early stopping
- OOF-optimised ensemble weights (SLSQP)

## Experiments that did NOT help

- Temporal lag / rolling / history features
- High-cardinality interaction target encoding (geohash x hour, geohash x day-of-week)
- Spatial cluster features

These raised local CV but reduced the leaderboard score (a temporal/interaction-heavy
variant scored 90.49 versus the best 91.45), confirming the feature set had limited
room for added complexity.

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

The biggest improvements came from:

- fixing target leakage (in-fold OOF target encoding)
- native categorical handling and a log-transformed target
- calibrated, OOF-optimised ensemble weighting

NOT from:
- deeper models
- excessive feature interactions or temporal/history features
- complicated stacking

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- SciPy
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
│   ├── EDA_for_Traffic_Prediction.ipynb
│   ├── Data_Preprocessing_Traffic_Prediction.ipynb
│   ├── Feature_Engineering_Traffic_Prediction.ipynb
│   ├── Final_Submission_Flipkart_Gridlock_Optimised.ipynb
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

- Optuna hyperparameter tuning
- Increased folds + multi-seed averaging
- Smoothed (Bayesian) target encoding
- Additional diverse base learners (e.g. HistGradientBoosting)
- Adversarial Validation
- Graph-based traffic modeling
- Deep learning for spatio-temporal forecasting

---

# Final Leaderboard Score

Public Score: 91.45390

---

# Author

Sahil Burnwal

Machine Learning | Ensemble Learning | Spatio-Temporal Modeling
