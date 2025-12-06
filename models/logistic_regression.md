# Logistic Regression Model for Diabetes Prediction

## Model overview

This is a scikit-learn Pipeline object that chains together multiple preprocessing, feature engineering, and modeling steps into a single estimator. Pipelines ensure that all transformations are applied consistently during both training and inference, preventing data leakage and simplifying deployment. The pipeline was optimized using RandomizedSearchCV with resource constraints and serialized using joblib for efficient storage and loading of the fitted transformers and model.

Key features:
- **End-to-end processing**: Automatically handles all preprocessing from raw data to predictions
- **Reproducible transformations**: All fitted parameters (scalers, encoders, PCA components) are preserved
- **Hyperparameter optimization**: Parameters across all pipeline steps were jointly optimized
- **Resource-aware training**: Model was trained within specified memory (16GB) and runtime (10 min) constraints

## Files

- **Model file**: `logistic_regression.joblib` (scikit-learn Pipeline object serialized with joblib)
- **Documentation**: `logistic_regression.md`

## Training information

- **Training date**: 2025-12-06 05:08:51
- **Training samples**: 17,041
- **Random state**: 315
- **Cross-validation score (ROC-AUC)**: 0.6662

## Hyperparameter optimization

- **Method**: Randomized Search CV
- **Cross-validation folds**: 3
- **Iterations**: 3
- **Scoring metric**: ROC-AUC
- **Optimization runtime**: 421.4 seconds (7.0 minutes)

## Pipeline components

### 1. Preprocessing

#### ID column removal
- **ID column dropper**: Automatically removes the 'id' column from input data

#### Numerical features
- **IQR clipping**: Outlier clipping using interquartile range (multiplier: 2.50) *[optimized]*
- **Standardization**: Standard scaling (mean=0, std=1)
- **Features**: age, alcohol_consumption_per_week, diet_score, physical_activity_minutes_per_week, sleep_hours_per_day, screen_time_hours_per_day, bmi, waist_to_hip_ratio, systolic_bp, diastolic_bp, heart_rate, cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides

#### Ordinal features
- **Ordinal encoding**: education_level, income_level

#### Nominal features
- **One-hot encoding**: gender, ethnicity, smoking_status, employment_status, family_history_diabetes, hypertension_history, cardiovascular_history (drop first category)

### 2. Feature engineering

- **Polynomial features**:
  - Degree: 1 *[optimized]*
  - Include bias: False *[optimized]*
  - Interaction only: True *[optimized]*

- **Constant feature removal**: Removes features with zero variance

- **Post-polynomial standardization**: Standard scaling after polynomial transformation

- **PCA dimensionality reduction**:
  - Components: 18 *[optimized]*
  - SVD solver: randomized *[optimized]*
  - Whiten: True *[optimized]*

### 3. Classifier

- **Algorithm**: Logistic regression
- **Penalty**: l2 *[optimized]*
- **Regularization (C)**: 0.1305 *[optimized]*
- **Max iterations**: 1000
- **Class weight**: balanced

## Usage

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('logistic_regression.joblib')

# Prepare test data (pipeline will automatically handle 'id' column)
X_test = pd.read_csv('test.csv')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Notes

- Input data can include the 'id' column - it will be automatically removed by the pipeline
- The pipeline handles all preprocessing and feature engineering automatically
- All transformations are applied in the correct sequence without requiring manual intervention
- Model was trained with resource constraints: 16GB memory limit, 10 minute runtime limit
