"""Custom transformers for the logistic regression model pipeline.

This file must be available when loading the logistic_regression.joblib model.
On Kaggle, upload this file along with the model as a dataset.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IDColumnDropper(BaseEstimator, TransformerMixin):
    """Drops the 'id' column from the dataframe."""
    
    def __init__(self, id_column='id'):
        self.id_column = id_column
    
    def fit(self, X, y=None):
        """No fitting necessary for dropping columns."""
        return self
    
    def transform(self, X):
        """Drop the ID column if it exists."""
        if isinstance(X, pd.DataFrame):
            if self.id_column in X.columns:
                return X.drop(columns=[self.id_column])
            return X
        else:
            # If not a DataFrame, return as is
            return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips features to a multiple of the interquartile range (IQR)."""
    
    def __init__(self, iqr_multiplier=2.0):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        """Calculate the clipping bounds based on IQR."""
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.iqr_multiplier * IQR
        self.upper_bounds_ = Q3 + self.iqr_multiplier * IQR
        return self
    
    def transform(self, X):
        """Apply clipping to the data."""
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class ConstantFeatureRemover(BaseEstimator, TransformerMixin):
    """Removes features with constant values (zero variance)."""
    
    def __init__(self):
        self.constant_features_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
    
    def fit(self, X, y=None):
        """Identify constant-valued features."""
        # Calculate variance for each feature
        variances = np.var(X, axis=0)
        # Identify constant features (variance close to zero)
        self.constant_features_ = np.where(np.isclose(variances, 0.0))[0]
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.n_features_in_ - len(self.constant_features_)
        return self
    
    def transform(self, X):
        """Remove constant-valued features."""
        # Create mask for non-constant features
        mask = np.ones(self.n_features_in_, dtype=bool)
        mask[self.constant_features_] = False
        return X[:, mask]
