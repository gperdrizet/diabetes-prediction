"""Custom feature engineering transformers for ensemble stage 1 models.

This module provides a collection of sklearn-compatible transformers for diverse
feature engineering. Each transformer can randomly select feature pairs during fit
to ensure diversity across ensemble members.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import gaussian_kde


class RandomFeatureSelector(BaseEstimator, TransformerMixin):
    """Randomly selects a subset of features (columns).
    
    Parameters
    ----------
    feature_fraction : float, default=0.75
        Fraction of features to select (0.5 to 0.95).
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, feature_fraction=0.75, random_state=None):
        self.feature_fraction = feature_fraction
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Select random feature indices."""
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        n_selected = max(1, int(n_features * self.feature_fraction))
        self.selected_indices_ = np.sort(rng.choice(n_features, size=n_selected, replace=False))
        self.n_features_in_ = n_features
        self.n_features_out_ = n_selected
        return self
    
    def transform(self, X):
        """Select the chosen features."""
        return X[:, self.selected_indices_]
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'feature_{i}' for i in self.selected_indices_])
        return np.array(input_features)[self.selected_indices_]


class RatioTransformer(BaseEstimator, TransformerMixin):
    """Creates ratio features from random pairs of input features.
    
    Parameters
    ----------
    n_features : int, default=10
        Number of ratio features to create.
    epsilon : float, default=1e-8
        Small constant to prevent division by zero.
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_features=10, epsilon=1e-8, random_state=None):
        self.n_features = n_features
        self.epsilon = epsilon
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Randomly select feature pairs for ratios."""
        rng = np.random.RandomState(self.random_state)
        n_input_features = X.shape[1]
        
        # Generate random pairs
        self.feature_pairs_ = []
        for _ in range(self.n_features):
            pair = rng.choice(n_input_features, size=2, replace=False)
            self.feature_pairs_.append(tuple(pair))
        
        self.n_features_in_ = n_input_features
        self.n_features_out_ = self.n_features
        return self
    
    def transform(self, X):
        """Create ratio features."""
        ratios = np.zeros((X.shape[0], self.n_features))
        for i, (idx1, idx2) in enumerate(self.feature_pairs_):
            ratios[:, i] = X[:, idx1] / (X[:, idx2] + self.epsilon)
        return ratios
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'ratio_{i}_{j}' for i, j in self.feature_pairs_])
        return np.array([f'{input_features[i]}/{input_features[j]}' 
                        for i, j in self.feature_pairs_])


class ProductTransformer(BaseEstimator, TransformerMixin):
    """Creates product features from random pairs of input features.
    
    Parameters
    ----------
    n_features : int, default=10
        Number of product features to create.
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_features=10, random_state=None):
        self.n_features = n_features
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Randomly select feature pairs for products."""
        rng = np.random.RandomState(self.random_state)
        n_input_features = X.shape[1]
        
        self.feature_pairs_ = []
        for _ in range(self.n_features):
            pair = rng.choice(n_input_features, size=2, replace=False)
            self.feature_pairs_.append(tuple(pair))
        
        self.n_features_in_ = n_input_features
        self.n_features_out_ = self.n_features
        return self
    
    def transform(self, X):
        """Create product features."""
        products = np.zeros((X.shape[0], self.n_features))
        for i, (idx1, idx2) in enumerate(self.feature_pairs_):
            products[:, i] = X[:, idx1] * X[:, idx2]
        return products
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'product_{i}_{j}' for i, j in self.feature_pairs_])
        return np.array([f'{input_features[i]}*{input_features[j]}' 
                        for i, j in self.feature_pairs_])


class DifferenceTransformer(BaseEstimator, TransformerMixin):
    """Creates difference features from random pairs of input features.
    
    Parameters
    ----------
    n_features : int, default=10
        Number of difference features to create.
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_features=10, random_state=None):
        self.n_features = n_features
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Randomly select feature pairs for differences."""
        rng = np.random.RandomState(self.random_state)
        n_input_features = X.shape[1]
        
        self.feature_pairs_ = []
        for _ in range(self.n_features):
            pair = rng.choice(n_input_features, size=2, replace=False)
            self.feature_pairs_.append(tuple(pair))
        
        self.n_features_in_ = n_input_features
        self.n_features_out_ = self.n_features
        return self
    
    def transform(self, X):
        """Create difference features."""
        differences = np.zeros((X.shape[0], self.n_features))
        for i, (idx1, idx2) in enumerate(self.feature_pairs_):
            differences[:, i] = X[:, idx1] - X[:, idx2]
        return differences
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'diff_{i}_{j}' for i, j in self.feature_pairs_])
        return np.array([f'{input_features[i]}-{input_features[j]}' 
                        for i, j in self.feature_pairs_])


class SumTransformer(BaseEstimator, TransformerMixin):
    """Creates sum features from random pairs of input features.
    
    Parameters
    ----------
    n_features : int, default=10
        Number of sum features to create.
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_features=10, random_state=None):
        self.n_features = n_features
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Randomly select feature pairs for sums."""
        rng = np.random.RandomState(self.random_state)
        n_input_features = X.shape[1]
        
        self.feature_pairs_ = []
        for _ in range(self.n_features):
            pair = rng.choice(n_input_features, size=2, replace=False)
            self.feature_pairs_.append(tuple(pair))
        
        self.n_features_in_ = n_input_features
        self.n_features_out_ = self.n_features
        return self
    
    def transform(self, X):
        """Create sum features."""
        sums = np.zeros((X.shape[0], self.n_features))
        for i, (idx1, idx2) in enumerate(self.feature_pairs_):
            sums[:, i] = X[:, idx1] + X[:, idx2]
        return sums
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'sum_{i}_{j}' for i, j in self.feature_pairs_])
        return np.array([f'{input_features[i]}+{input_features[j]}' 
                        for i, j in self.feature_pairs_])


class ReciprocalTransformer(BaseEstimator, TransformerMixin):
    """Creates reciprocal features (1/x) for all input features.
    
    Parameters
    ----------
    epsilon : float, default=1e-8
        Small constant to prevent division by zero.
    """
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        
    def fit(self, X, y=None):
        """No fitting necessary."""
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Create reciprocal features."""
        return 1.0 / (X + self.epsilon)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'reciprocal_{i}' for i in range(self.n_features_in_)])
        return np.array([f'1/{name}' for name in input_features])


class SquareTransformer(BaseEstimator, TransformerMixin):
    """Creates squared features (x²) for all input features."""
    
    def fit(self, X, y=None):
        """No fitting necessary."""
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Create squared features."""
        return X ** 2
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'square_{i}' for i in range(self.n_features_in_)])
        return np.array([f'{name}²' for name in input_features])


class SquareRootTransformer(BaseEstimator, TransformerMixin):
    """Creates square root features preserving sign: √|x| × sign(x)."""
    
    def fit(self, X, y=None):
        """No fitting necessary."""
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Create square root features."""
        return np.sqrt(np.abs(X)) * np.sign(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'sqrt_{i}' for i in range(self.n_features_in_)])
        return np.array([f'√{name}' for name in input_features])


class LogTransformer(BaseEstimator, TransformerMixin):
    """Creates log features preserving sign: log(|x|+1) × sign(x)."""
    
    def fit(self, X, y=None):
        """No fitting necessary."""
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Create log features."""
        return np.log(np.abs(X) + 1) * np.sign(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'log_{i}' for i in range(self.n_features_in_)])
        return np.array([f'log({name})' for name in input_features])


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Creates binned features using quantile or uniform binning.
    
    Parameters
    ----------
    n_bins : int, default=5
        Number of bins (3 to 10).
    strategy : str, default='quantile'
        Strategy for binning: 'quantile' or 'uniform'.
    encode : str, default='ordinal'
        Encoding method: 'ordinal' or 'onehot'.
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_bins=5, strategy='quantile', encode='ordinal', random_state=None):
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Fit the binning transformer."""
        self.binner_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy=self.strategy,
            random_state=self.random_state
        )
        self.binner_.fit(X)
        self.n_features_in_ = X.shape[1]
        
        if self.encode == 'onehot':
            self.n_features_out_ = sum(len(edges) - 1 for edges in self.binner_.bin_edges_)
        else:
            self.n_features_out_ = X.shape[1]
        
        return self
    
    def transform(self, X):
        """Transform features into bins."""
        return self.binner_.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if self.encode == 'ordinal':
            if input_features is None:
                return np.array([f'binned_{i}' for i in range(self.n_features_in_)])
            return np.array([f'binned_{name}' for name in input_features])
        else:
            # One-hot encoded bins
            names = []
            for i in range(self.n_features_in_):
                n_bins_feature = len(self.binner_.bin_edges_[i]) - 1
                if input_features is None:
                    names.extend([f'bin_{i}_{j}' for j in range(n_bins_feature)])
                else:
                    names.extend([f'{input_features[i]}_bin_{j}' for j in range(n_bins_feature)])
            return np.array(names)


class KDESmoothingTransformer(BaseEstimator, TransformerMixin):
    """Applies Gaussian KDE smoothing to features.
    
    Parameters
    ----------
    bandwidth : str or float, default='scott'
        Bandwidth selection method: 'scott', 'silverman', or a float value.
    n_samples : int, default=1000
        Number of samples for KDE evaluation (for efficiency).
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, bandwidth='scott', n_samples=1000, random_state=None):
        self.bandwidth = bandwidth
        self.n_samples = n_samples
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Fit KDE for each feature."""
        self.kdes_ = []
        self.feature_ranges_ = []
        
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            
            # Fit KDE
            try:
                kde = gaussian_kde(feature_data, bw_method=self.bandwidth)
                self.kdes_.append(kde)
                
                # Store range for evaluation
                self.feature_ranges_.append((feature_data.min(), feature_data.max()))
            except:
                # If KDE fails (e.g., constant feature), use None
                self.kdes_.append(None)
                self.feature_ranges_.append((0, 1))
        
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Apply KDE smoothing."""
        X_smoothed = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            if self.kdes_[i] is not None:
                # Evaluate KDE at data points
                X_smoothed[:, i] = self.kdes_[i](X[:, i])
            else:
                # If KDE failed, return original
                X_smoothed[:, i] = X[:, i]
        
        return X_smoothed
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return np.array([f'kde_{i}' for i in range(self.n_features_in_)])
        return np.array([f'kde_{name}' for name in input_features])
