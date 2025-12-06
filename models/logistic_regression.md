# Logistic Regression Model for Diabetes Prediction

## Model overview

This is a scikit-learn Pipeline object that chains together multiple preprocessing, feature engineering, and modeling steps into a single estimator. Pipelines ensure that all transformations are applied consistently during both training and inference, preventing data leakage and simplifying deployment. The pipeline was optimized using RandomizedSearchCV with resource constraints and serialized using joblib for efficient storage and loading of the fitted transformers and model.

Key features:
- **End-to-end processing**: Automatically handles all preprocessing from raw data to predictions
- **Reproducible transformations**: All fitted parameters (scalers, encoders, PCA components) are preserved
- **Hyperparameter optimization**: Parameters across all pipeline steps were jointly optimized
- **Resource-aware training**: Model was trained within specified memory (30GB) and runtime (1440 min) constraints

For details on model optimization and training, see the [Jupyter notebook on GitHub](https://github.com/gperdrizet/diabetes-prediction/blob/main/notebooks/01.1-logistic_regression_model.ipynb).

## Files

- **Model file**: `logistic_regression.joblib` (scikit-learn Pipeline object serialized with joblib)
- **Custom transformers**: `logistic_regression_transformers.py` (required for model deserialization)
- **Documentation**: `logistic_regression.md`

## Training information

- **Training date**: 2025-12-06 15:00:38
- **Training samples**: 700,000
- **Random state**: 315
- **Cross-validation score (ROC-AUC)**: 0.6438

## Hyperparameter optimization

- **Method**: Randomized Search CV
- **Cross-validation folds**: 3
- **Iterations**: 303
- **Scoring metric**: ROC-AUC
- **Optimization runtime**: 8156.2 seconds (135.9 minutes)

## Inference performance

Measured on test dataset with 300,000 samples using `tracemalloc` to track peak memory allocation:

- **Inference time**: 5.8265 seconds
- **Throughput**: 51,489 samples/second
- **Peak memory**: 1.9916 GB

## Pipeline components

### 1. Preprocessing

#### ID column removal
- **ID column dropper**: Automatically removes the 'id' column from input data (custom transformer)

#### Numerical features
- **IQR clipping**: Outlier clipping using interquartile range (multiplier: 1.35) *[optimized]* (custom transformer)
- **Standardization**: Standard scaling (mean=0, std=1)
- **Features**: age, alcohol_consumption_per_week, diet_score, physical_activity_minutes_per_week, sleep_hours_per_day, screen_time_hours_per_day, bmi, waist_to_hip_ratio, systolic_bp, diastolic_bp, heart_rate, cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides

#### Ordinal features
- **Ordinal encoding**: education_level, income_level

#### Nominal features
- **One-hot encoding**: gender, ethnicity, smoking_status, employment_status, family_history_diabetes, hypertension_history, cardiovascular_history (drop first category)

### 2. Feature engineering

- **Polynomial features**:
  - Degree: 2 *[optimized]*
  - Include bias: False *[optimized]*
  - Interaction only: True *[optimized]*

- **Constant feature removal**: Removes features with zero variance (custom transformer)

- **Post-polynomial standardization**: Standard scaling after polynomial transformation

- **PCA dimensionality reduction**:
  - Components: 62 *[optimized]*
  - SVD solver: randomized *[optimized]*
  - Whiten: True *[optimized]*

### 3. Classifier

- **Algorithm**: Logistic regression
- **Penalty**: None *[optimized]*
- **Regularization (C)**: N/A *[optimized]*
- **Max iterations**: 1000
- **Class weight**: balanced

## Custom transformers

The model uses three custom scikit-learn transformers defined in `logistic_regression_transformers.py`:

### IDColumnDropper
Automatically removes the 'id' column from input DataFrames before processing. This allows the model to accept raw test data without manual preprocessing.

### IQRClipper
Clips outliers in numerical features based on the interquartile range (IQR). During fitting, calculates clipping bounds as Q1 - k×IQR and Q3 + k×IQR, where k is the optimized multiplier. This reduces the impact of extreme outliers while preserving the overall distribution.

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
- All transformations are applied in the correct sequence without requiring manual intervention
- Model was trained with resource constraints: 30GB memory limit, 1440 minute runtime limit
- The `logistic_regression_transformers.py` file must be in the Python path when loading the model
