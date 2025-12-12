"""Preprocessing transformers for the ensemble stage1 pipelines.

This module contains transformers for cleaning and preparing data:
- ConstantFeatureRemover: Removes features with zero variance
- IQRClipper: Clips outliers using interquartile range
- CleanNumericTransformer: Handles NaN/Inf values
- RandomFeatureSelector: Random column sampling for diversity
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class ConstantFeatureRemover(BaseEstimator, TransformerMixin):
    """Removes features with constant values (zero variance).
    
    This transformer identifies features that have the same value across all samples
    (or nearly constant with variance close to zero) and removes them, as they provide
    no discriminative information for modeling.
    """
    
    def __init__(self, variance_threshold=1e-10):
        """Initialize the constant feature remover.
        
        Parameters
        ----------
        variance_threshold : float, default=1e-10
            Features with variance below this threshold are considered constant.
        """
        self.variance_threshold = variance_threshold
        self.constant_features_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
    
    def fit(self, X, y=None):
        """Identify constant-valued features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored).
        
        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Calculate variance for each feature
        variances = np.var(X, axis=0)
        # Identify constant features (variance below threshold)
        self.constant_features_ = np.where(variances < self.variance_threshold)[0]
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.n_features_in_ - len(self.constant_features_)
        return self
    
    def transform(self, X):
        """Remove constant-valued features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_out)
            Data with constant features removed.
        """
        if self.constant_features_ is None:
            raise ValueError("Transformer must be fitted before calling transform")
        
        # Safety check: If ALL features are constant, keep the first one
        # This prevents downstream transformers from receiving 0 features
        if len(self.constant_features_) >= self.n_features_in_:
            # All features are constant - keep just the first feature
            return X[:, :1]
        
        # Create mask for non-constant features
        mask = np.ones(self.n_features_in_, dtype=bool)
        mask[self.constant_features_] = False
        return X[:, mask]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips features to a multiple of the interquartile range (IQR).
    
    This transformer identifies outliers based on the IQR and clips them to
    a specified range (Q1 - k*IQR, Q3 + k*IQR) where k is the IQR multiplier.
    """
    
    def __init__(self, iqr_multiplier=2.0):
        """Initialize the IQR clipper.
        
        Parameters
        ----------
        iqr_multiplier : float, default=2.0
            Multiplier for the IQR to determine clipping bounds.
            Standard boxplot uses 1.5, we default to 2.0 for less aggressive clipping.
        """
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        """Calculate the clipping bounds based on IQR.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored).
        
        Returns
        -------
        self : object
            Fitted transformer.
        """
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.iqr_multiplier * IQR
        self.upper_bounds_ = Q3 + self.iqr_multiplier * IQR
        return self
    
    def transform(self, X):
        """Apply clipping to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_clipped : array-like of shape (n_samples, n_features)
            Data with outliers clipped.
        """
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("Transformer must be fitted before calling transform")
        
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class CleanNumericTransformer(BaseEstimator, TransformerMixin):
    """Handles NaN and Inf values in numeric data.
    
    This transformer is designed to clean up after feature engineering
    operations that may produce NaN or Inf values (e.g., log of negative
    numbers, division by zero, etc.).
    """
    
    def __init__(self, strategy='median'):
        """Initialize the numeric cleaner.
        
        Parameters
        ----------
        strategy : str, default='median'
            Strategy for imputing NaN values. Options: 'mean', 'median', 'constant'.
            Median is robust to outliers.
        """
        self.strategy = strategy
        self.imputer_ = None
    
    def fit(self, X, y=None):
        """Fit the imputer on the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored).
        
        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Replace Inf with NaN so imputer can handle them
        X_clean = np.copy(X)
        X_clean[np.isinf(X_clean)] = np.nan
        
        # Fit imputer
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(X_clean)
        return self
    
    def transform(self, X):
        """Clean NaN and Inf values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_clean : array-like of shape (n_samples, n_features)
            Data with NaN and Inf values imputed.
        """
        if self.imputer_ is None:
            raise ValueError("Transformer must be fitted before calling transform")
        
        # Replace Inf with NaN
        X_clean = np.copy(X)
        X_clean[np.isinf(X_clean)] = np.nan
        
        # Impute NaN values
        return self.imputer_.transform(X_clean)


class RandomFeatureSelector(BaseEstimator, TransformerMixin):
    """Randomly selects a fraction of features for diversity.
    
    This transformer is used for column sampling to increase model diversity
    in ensemble methods. Each model sees a different random subset of features.
    """
    
    def __init__(self, feature_fraction=0.5, random_state=None):
        """Initialize the random feature selector.
        
        Parameters
        ----------
        feature_fraction : float, default=0.5
            Fraction of features to keep (between 0 and 1).
        random_state : int, RandomState instance or None, default=None
            Random state for reproducibility. If None, uses a different
            random subset each time (good for ensemble diversity).
        """
        self.feature_fraction = feature_fraction
        self.random_state = random_state
        self.selected_features_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """Select random features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored).
        
        Returns
        -------
        self : object
            Fitted transformer.
        """
        self.n_features_in_ = X.shape[1]
        n_features_to_select = max(1, int(self.n_features_in_ * self.feature_fraction))
        
        # Create RNG
        rng = np.random.RandomState(self.random_state)
        
        # Randomly select features
        all_features = np.arange(self.n_features_in_)
        self.selected_features_ = np.sort(
            rng.choice(all_features, size=n_features_to_select, replace=False)
        )
        return self
    
    def transform(self, X):
        """Select the chosen features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_selected : array-like of shape (n_samples, n_features_selected)
            Data with only selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer must be fitted before calling transform")
        
        return X[:, self.selected_features_]
