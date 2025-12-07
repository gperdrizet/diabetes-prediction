# Logistic Regression Model for Diabetes Prediction

## Model overview

This dataset contains a scikit-learn Pipeline object that chains together multiple preprocessing, feature engineering, and modeling steps into a single estimator. Pipelines ensure that all transformations are applied consistently during both training and inference, preventing data leakage and simplifying deployment. The pipeline was optimized using RandomizedSearchCV. Optimized hyperparameters are marked with *[optimized]* in the pipeline component descriptions below.

For details on model optimization and training, see the [logistic regression optimization and training notebook](https://github.com/gperdrizet/diabetes-prediction/blob/main/notebooks/01.1-logistic_regression_model.ipynb) on GitHub.

## Files

- **Model file**: `logistic_regression.joblib` (scikit-learn Pipeline object serialized with joblib)
- **Custom transformers**: `logistic_regression_transformers.py` (required for model deserialization)
- **Documentation**: `logistic_regression.md`

Key features:
- **End-to-end processing**: Automatically handles all preprocessing from raw data to predictions
- **Reproducible transformations**: All fitted parameters (scalers, encoders, PCA components) are preserved
- **Hyperparameter optimization**: Parameters across all pipeline steps were jointly optimized

## Training information

- **Training date**: 2025-12-07 10:41:29
- **Training samples**: 700,000
- **Cross-validation score (ROC-AUC)**: 0.6855

## Hyperparameter optimization

- **Method**: Randomized Search CV
- **Cross-validation folds**: 3
- **Iterations**: 63
- **Scoring metric**: ROC-AUC
- **Optimization runtime**: 1433.9 seconds (23.9 minutes)

## Inference performance

Measured on test dataset with 300,000 samples using `tracemalloc` to track peak memory allocation:

- **Inference time**: 4.6238 seconds
- **Throughput**: 64,882 samples/second
- **Peak memory**: 2.1950 GB

## Pipeline components

### 1. Preprocessing

#### ID column removal
- **ID column dropper**: Automatically removes the 'id' column from input data (custom transformer)

#### Numerical features
- **Standardization**: Standard scaling (mean=0, std=1)
- **Features**: age, alcohol_consumption_per_week, diet_score, physical_activity_minutes_per_week, sleep_hours_per_day, screen_time_hours_per_day, bmi, waist_to_hip_ratio, systolic_bp, diastolic_bp, heart_rate, cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides

#### Ordinal features
- **Ordinal encoding**: education_level, income_level

#### Nominal features
- **One-hot encoding**: gender, ethnicity, smoking_status, employment_status, family_history_diabetes, hypertension_history, cardiovascular_history (drop first category)

### 2. Feature engineering

- **Polynomial features**:
  - Degree: 2 *[optimized]*
  - Include bias: True *[optimized]*
  - Interaction only: True *[optimized]*

- **Constant feature removal**: Removes features with zero variance (custom transformer)

- **Post-polynomial standardization**: Standard scaling after polynomial transformation

- **PCA dimensionality reduction**:
  - Components: 65 *[optimized]*
  - SVD solver: randomized *[optimized]*
  - Whiten: False *[optimized]*

### 3. Classifier

- **Algorithm**: Logistic regression
- **Penalty**: l2 *[optimized]*
- **Regularization (C)**: 0.0043 *[optimized]*
- **Max iterations**: 1000
- **Class weight**: balanced

## Custom transformers

The model uses two custom scikit-learn transformers defined in `logistic_regression_transformers.py`:

### IDColumnDropper
Automatically removes the 'id' column from input DataFrames before processing. This allows the model to accept raw test data without manual preprocessing.

### ConstantFeatureRemover
Removes features with zero variance after polynomial transformation. This eliminates redundant features that don't contribute to model predictions, reducing dimensionality and improving computational efficiency.

**Important**: The `logistic_regression_transformers.py` file must be available in the Python path when loading the model, as joblib stores references to these classes and needs to import them during deserialization.

## Usage

```python
import joblib
import pandas as pd
import sys
from pathlib import Path

# Add the models directory to the path (adjust as needed)
sys.path.insert(0, str(Path('models').resolve()))

# Load the model (this will import the custom transformers)
model = joblib.load('models/logistic_regression.joblib')

# Prepare test data (pipeline will automatically handle 'id' column)
X_test = pd.read_csv('test.csv')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Notes

- Input data can include the 'id' column - it will be automatically removed by the pipeline
- The pipeline handles all preprocessing and feature engineering automatically
- The `logistic_regression_transformers.py` file must be in the Python path when loading the model
