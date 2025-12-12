# Gradient boosting model for diabetes prediction

## Model overview

This dataset contains a scikit-learn Pipeline object that chains together multiple preprocessing, feature engineering, and modeling steps into a single estimator. Pipelines ensure that all transformations are applied consistently during both training and inference, preventing data leakage and simplifying deployment. The pipeline was optimized using RandomizedSearchCV with extensive hyperparameter tuning.

For details on model optimization and training, see the [gradient boosting optimization and training notebook](https://github.com/gperdrizet/diabetes-prediction/blob/main/notebooks/02.1-gradient_boosting_model.ipynb) on GitHub.

## Files

- **Model file**: `gradient_boosting.joblib` (scikit-learn Pipeline object serialized with joblib)
- **Custom transformers**: `gradient_boosting_transformers.py` (required for model deserialization)
- **Documentation**: `gradient_boosting.md`

Key features:
- **End-to-end processing**: Automatically handles all preprocessing from raw data to predictions
- **Reproducible transformations**: All fitted parameters (encoders, feature engineering, clustering) are preserved
- **Hyperparameter optimization**: Parameters across all pipeline steps were jointly optimized

## Training information

- **Training date**: 2025-12-11 18:28:35
- **Training samples**: 700,000
- **Cross-validation**: 3-fold CV with ROC-AUC scoring
- **Optimization method**: RandomizedSearchCV with 200 iterations
- **Optimization samples**: ~70,000 samples (10%) for hyperparameter tuning

## Pipeline components

### 1. Preprocessing

#### ID column removal
- **ID column dropper**: Automatically removes the 'id' column from input data (custom transformer)

#### Numerical features (14 features)
- **Features**: age, physical_activity_minutes_per_week, diet_score, sleep_hours_per_day, screen_time_hours_per_day, bmi, waist_to_hip_ratio, systolic_bp, diastolic_bp, heart_rate, cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides
- **Transformation**: Passed through for feature engineering

#### Ordinal features (6 features)
- **Ordinal encoding**: education_level, income_level, alcohol_consumption_per_week, family_history_diabetes, hypertension_history, cardiovascular_history
- **Method**: OrdinalEncoder with predefined category orders

#### Nominal features (4 features)
- **One-hot encoding**: gender, ethnicity, smoking_status, employment_status
- **Method**: OneHotEncoder (drop first category to avoid multicollinearity)

### 2. Feature engineering

This pipeline creates extensive synthetic features to capture complex relationships:

- **Binning (discretization)**:
  - Bins: 5 quantile-based bins
  - Applied to all 14 numerical features
  - Creates categorical representations of continuous features

- **KMeans clustering** (3 cluster feature sets):
  - **Heart health cluster** (4 clusters): systolic_bp, diastolic_bp, heart_rate, hypertension_history, cardiovascular_history
  - **Cholesterol cluster** (4 clusters): cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides
  - **Lifestyle cluster** (4 clusters): physical_activity_minutes_per_week, diet_score, sleep_hours_per_day, screen_time_hours_per_day, alcohol_consumption_per_week

- **Polynomial features**:
  - Degree: 2
  - Include bias: False
  - Creates interaction terms and squared features

- **Synthetic features** (custom transformers):
  - **Difference features**: All pairwise feature differences (A - B)
  - **Sum features**: All pairwise feature sums (A + B)
  - **Ratio features**: All pairwise feature ratios (A / B)
  - **Reciprocal features**: 1/feature for all features
  - **Log features**: Natural log transform of all features
  - **Square root features**: Square root of all features

### 3. Feature selection

After feature engineering, the pipeline reduces dimensionality:

- **Variance threshold**: Removes low-variance features (optimized threshold)
- **Select percentile**: Keeps top N% of features based on ANOVA F-value (optimized percentage)

### 4. Classifier

- **Algorithm**: HistGradientBoostingClassifier
- **Optimized hyperparameters**:
  - Learning rate: Optimized via log-uniform distribution (0.001 - 0.3)
  - Max iterations: Optimized (100 - 1000 range)
  - Max depth: Optimized (10 - 100 range)
  - Min samples leaf: Optimized (5 - 50 range)
  - L2 regularization: Optimized via log-uniform distribution
  - Class weight: Optimized (None or balanced)
- **Fixed parameters**:
  - Max bins: 255
  - Early stopping: Enabled
  - Validation fraction: 0.1
  - N iter no change: 50

## Custom transformers

The model uses multiple custom scikit-learn transformers defined in `gradient_boosting_transformers.py`:

### IDColumnDropper
Automatically removes the 'id' column from input DataFrames before processing.

### IQRClipper
Clips outliers using the interquartile range (IQR) method (not used in final pipeline but available).

### Feature engineering transformers
- **DifferenceFeatures**: Creates difference features between all pairs of input features
- **SumFeatures**: Creates sum features from all combinations of input features
- **RatioFeatures**: Creates ratio features between all pairs of input features (handles division by zero)
- **ReciprocalFeatures**: Creates reciprocal (1/x) features for all input features (handles division by zero)
- **LogFeatures**: Creates log-transformed features for all input features (handles negative values)
- **SquareRootFeatures**: Creates square root features for all input features (handles negative values)
- **KMeansClusterFeatures**: Creates cluster membership features using KMeans clustering

**Important**: The `gradient_boosting_transformers.py` file must be available in the Python path when loading the model, as joblib stores references to these classes and needs to import them during deserialization.

## Usage

```python
import joblib
import pandas as pd
import sys
from pathlib import Path

# Add the models directory to the path (adjust as needed)
sys.path.insert(0, str(Path('models').resolve()))

# Load the model (this will import the custom transformers)
model = joblib.load('models/gradient_boosting.joblib')

# Prepare test data (pipeline will automatically handle 'id' column)
X_test = pd.read_csv('test.csv')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Notes

- Input data can include the 'id' column - it will be automatically removed by the pipeline
- The pipeline handles all preprocessing and feature engineering automatically
- The `gradient_boosting_transformers.py` file must be in the Python path when loading the model
- This model uses extensive feature engineering including polynomial features, clustering, and synthetic features
- Feature engineering creates thousands of features which are then reduced via feature selection
