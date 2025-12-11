"""Custom transformers for the gradient boosting model pipeline.

This file must be available when loading the gradient_boosting.joblib model.
On Kaggle, upload this file along with the model as a dataset.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


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


class DifferenceFeatures(BaseEstimator, TransformerMixin):
    """Creates difference features for all combinations of input features.
    
    For each pair of features (A, B), creates: A - B
    """
    
    def __init__(self):
        self.feature_names_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """Store feature information."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Create difference features."""
        from itertools import combinations
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        new_features = []
        for i, j in combinations(range(self.n_features_in_), 2):
            new_features.append(X_array[:, i] - X_array[:, j])
        
        return np.column_stack(new_features)


class SumFeatures(BaseEstimator, TransformerMixin):
    """Creates sum features for all combinations of input features.
    
    For each pair of features (A, B), creates: A + B
    """
    
    def __init__(self):
        self.feature_names_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """Store feature information."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Create sum features."""
        from itertools import combinations
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        new_features = []
        for i, j in combinations(range(self.n_features_in_), 2):
            new_features.append(X_array[:, i] + X_array[:, j])
        
        return np.column_stack(new_features)


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Creates ratio features for all permutations of input features.
    
    For each ordered pair of features (A, B), creates: A / (B + offset)
    The offset prevents division by zero and is calculated during fit.
    """
    
    def __init__(self):
        self.feature_names_ = None
        self.n_features_in_ = None
        self.offsets_ = None
    
    def fit(self, X, y=None):
        """Store feature information and calculate offsets."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        self.n_features_in_ = X_array.shape[1]
        # Calculate offset as min value + 1 for each feature
        self.offsets_ = X_array.min(axis=0) + 1
        return self
    
    def transform(self, X):
        """Create ratio features."""
        from itertools import permutations
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        new_features = []
        for i, j in permutations(range(self.n_features_in_), 2):
            denominator = X_array[:, j] + self.offsets_[j]
            new_features.append(X_array[:, i] / denominator)
        
        return np.column_stack(new_features)


class ReciprocalFeatures(BaseEstimator, TransformerMixin):
    """Creates reciprocal features for all input features.
    
    For each feature A, creates: 1 / (A + offset)
    The offset prevents division by zero and is calculated during fit.
    """
    
    def __init__(self):
        self.feature_names_ = None
        self.n_features_in_ = None
        self.offsets_ = None
    
    def fit(self, X, y=None):
        """Store feature information and calculate offsets."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        self.n_features_in_ = X_array.shape[1]
        # Calculate offset as min value + 1 for each feature
        self.offsets_ = X_array.min(axis=0) + 1
        return self
    
    def transform(self, X):
        """Create reciprocal features."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        new_features = []
        for i in range(self.n_features_in_):
            denominator = X_array[:, i] + self.offsets_[i]
            new_features.append(1.0 / denominator)
        
        return np.column_stack(new_features)


class LogFeatures(BaseEstimator, TransformerMixin):
    """Creates log-transformed features for all input features.
    
    For each feature A, creates: log(A + offset)
    The offset ensures positive values and is calculated during fit.
    """
    
    def __init__(self):
        self.feature_names_ = None
        self.n_features_in_ = None
        self.offsets_ = None
    
    def fit(self, X, y=None):
        """Store feature information and calculate offsets."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        self.n_features_in_ = X_array.shape[1]
        # Calculate offset as min value + 1 for each feature
        self.offsets_ = X_array.min(axis=0) + 1
        return self
    
    def transform(self, X):
        """Create log features."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        new_features = []
        for i in range(self.n_features_in_):
            shifted = X_array[:, i] + self.offsets_[i]
            new_features.append(np.log(shifted))
        
        return np.column_stack(new_features)


class SquareRootFeatures(BaseEstimator, TransformerMixin):
    """Creates square root features for all input features.
    
    For each feature A, creates: sqrt(A + offset)
    The offset ensures non-negative values and is calculated during fit.
    """
    
    def __init__(self):
        self.feature_names_ = None
        self.n_features_in_ = None
        self.offsets_ = None
    
    def fit(self, X, y=None):
        """Store feature information and calculate offsets."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        self.n_features_in_ = X_array.shape[1]
        # Calculate offset as min value + 1 for each feature
        self.offsets_ = X_array.min(axis=0) + 1
        return self
    
    def transform(self, X):
        """Create square root features."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        new_features = []
        for i in range(self.n_features_in_):
            shifted = X_array[:, i] + self.offsets_[i]
            new_features.append(shifted ** 0.5)
        
        return np.column_stack(new_features)


class KMeansClusterFeatures(BaseEstimator, TransformerMixin):
    """Creates KMeans cluster membership feature for a group of features.
    
    This transformer is designed to work with sklearn's ColumnTransformer.
    It fits a single KMeans model on the provided features and returns
    the cluster labels as a single column.
    """
    
    def __init__(self, n_clusters=4, random_state=315):
        """
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for KMeans
        random_state : int
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """Fit KMeans model on the provided features."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X_array)
        
        return self
    
    def transform(self, X):
        """Transform data by returning cluster membership."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if not hasattr(self, 'kmeans_'):
            raise RuntimeError("This KMeansClusterFeatures instance is not fitted yet. Call 'fit' first.")
        
        cluster_labels = self.kmeans_.predict(X_array)
        return cluster_labels.reshape(-1, 1)
