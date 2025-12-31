"""Standalone transformers module for Kaggle ensemble inference.

This module contains all custom feature engineering transformers needed
to deserialize the ensemble model. Upload this file to Kaggle alongside
the model joblib files.

No dependencies on the ensemble package - all code is self-contained.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde


class CleanNumericTransformer(BaseEstimator, TransformerMixin):
    """Replace NaN and infinite values with median."""
    
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.medians_ = None
    
    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=np.float64)
        X_clean = X_array.copy()
        X_clean[~np.isfinite(X_clean)] = np.nan
        self.medians_ = np.nanmedian(X_clean, axis=0)
        self.medians_[np.isnan(self.medians_)] = 0.0
        return self
    
    def transform(self, X):
        X_array = np.asarray(X, dtype=np.float64)
        X_clean = X_array.copy()
        for i in range(X_clean.shape[1]):
            mask = ~np.isfinite(X_clean[:, i])
            if np.any(mask):
                X_clean[mask, i] = self.medians_[i]
        return X_clean


class RandomFeatureSelector(BaseEstimator, TransformerMixin):
    """Randomly selects a subset of features."""
    
    def __init__(self, feature_fraction=0.75, random_state=None):
        self.feature_fraction = feature_fraction
        self.random_state = random_state
        
    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        if n_features == 0:
            raise ValueError("RandomFeatureSelector received 0 features.")
        n_selected = max(1, int(n_features * self.feature_fraction))
        n_selected = min(n_selected, n_features)
        self.selected_indices_ = np.sort(rng.choice(n_features, size=n_selected, replace=False))
        self.n_features_in_ = n_features
        self.n_features_out_ = n_selected
        return self
    
    def transform(self, X):
        return X[:, self.selected_indices_]


class RatioTransformer(BaseEstimator, TransformerMixin):
    """Creates ratio features from random pairs."""
    
    def __init__(self, n_features=10, epsilon=1e-8, random_state=None):
        self.n_features = n_features
        self.epsilon = epsilon
        self.random_state = random_state
        
    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n_input_features = X.shape[1]
        self.feature_pairs_ = []
        if n_input_features < 2:
            for _ in range(self.n_features):
                self.feature_pairs_.append((0, 0))
        else:
            for _ in range(self.n_features):
                pair = rng.choice(n_input_features, size=2, replace=False)
                self.feature_pairs_.append(tuple(pair))
        self.n_features_in_ = n_input_features
        self.n_features_out_ = self.n_features
        return self
    
    def transform(self, X):
        ratios = np.zeros((X.shape[0], self.n_features))
        for i, (idx1, idx2) in enumerate(self.feature_pairs_):
            ratios[:, i] = X[:, idx1] / (X[:, idx2] + self.epsilon)
        return ratios


class ProductTransformer(BaseEstimator, TransformerMixin):
    """Creates product features from random pairs."""
    
    def __init__(self, n_features=10, random_state=None):
        self.n_features = n_features
        self.random_state = random_state
        
    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n_input_features = X.shape[1]
        self.feature_pairs_ = []
        if n_input_features < 2:
            for _ in range(self.n_features):
                self.feature_pairs_.append((0, 0))
        else:
            for _ in range(self.n_features):
                pair = rng.choice(n_input_features, size=2, replace=False)
                self.feature_pairs_.append(tuple(pair))
        self.n_features_in_ = n_input_features
        self.n_features_out_ = self.n_features
        return self
    
    def transform(self, X):
        products = np.zeros((X.shape[0], self.n_features))
        for i, (idx1, idx2) in enumerate(self.feature_pairs_):
            products[:, i] = X[:, idx1] * X[:, idx2]
        return products


class DifferenceTransformer(BaseEstimator, TransformerMixin):
    """Creates difference features from random pairs."""
    
    def __init__(self, n_features=10, random_state=None):
        self.n_features = n_features
        self.random_state = random_state
        
    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n_input_features = X.shape[1]
        self.feature_pairs_ = []
        if n_input_features < 2:
            for _ in range(self.n_features):
                self.feature_pairs_.append((0, 0))
        else:
            for _ in range(self.n_features):
                pair = rng.choice(n_input_features, size=2, replace=False)
                self.feature_pairs_.append(tuple(pair))
        self.n_features_in_ = n_input_features
        self.n_features_out_ = self.n_features
        return self
    
    def transform(self, X):
        differences = np.zeros((X.shape[0], self.n_features))
        for i, (idx1, idx2) in enumerate(self.feature_pairs_):
            differences[:, i] = X[:, idx1] - X[:, idx2]
        return differences


class SumTransformer(BaseEstimator, TransformerMixin):
    """Creates sum features from random pairs."""
    
    def __init__(self, n_features=10, random_state=None):
        self.n_features = n_features
        self.random_state = random_state
        
    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n_input_features = X.shape[1]
        self.feature_pairs_ = []
        if n_input_features < 2:
            for _ in range(self.n_features):
                self.feature_pairs_.append((0, 0))
        else:
            for _ in range(self.n_features):
                pair = rng.choice(n_input_features, size=2, replace=False)
                self.feature_pairs_.append(tuple(pair))
        self.n_features_in_ = n_input_features
        self.n_features_out_ = self.n_features
        return self
    
    def transform(self, X):
        sums = np.zeros((X.shape[0], self.n_features))
        for i, (idx1, idx2) in enumerate(self.feature_pairs_):
            sums[:, i] = X[:, idx1] + X[:, idx2]
        return sums


class ReciprocalTransformer(BaseEstimator, TransformerMixin):
    """Creates reciprocal features (1/x)."""
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        return 1.0 / (X + self.epsilon)


class SquareTransformer(BaseEstimator, TransformerMixin):
    """Creates squared features (x²)."""
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        return X ** 2


class SquareRootTransformer(BaseEstimator, TransformerMixin):
    """Creates square root features preserving sign."""
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        return np.sqrt(np.abs(X)) * np.sign(X)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Creates log features preserving sign."""
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        return np.log(np.abs(X) + 1) * np.sign(X)


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Creates binned features using quantile or uniform binning."""
    
    def __init__(self, n_bins=5, strategy='quantile', encode='ordinal', random_state=None):
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.random_state = random_state
        
    def fit(self, X, y=None):
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
        return self.binner_.transform(X)


class KDESmoothingTransformer(BaseEstimator, TransformerMixin):
    """Applies Gaussian KDE smoothing to features."""
    
    def __init__(self, bandwidth='scott', n_samples=1000, random_state=None):
        self.bandwidth = bandwidth
        self.n_samples = n_samples
        self.random_state = random_state
        
    def fit(self, X, y=None):
        self.kdes_ = []
        self.feature_ranges_ = []
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            try:
                kde = gaussian_kde(feature_data, bw_method=self.bandwidth)
                self.kdes_.append(kde)
                self.feature_ranges_.append((feature_data.min(), feature_data.max()))
            except:
                self.kdes_.append(None)
                self.feature_ranges_.append((0, 1))
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]
        return self
    
    def transform(self, X):
        X_smoothed = np.zeros_like(X)
        for i in range(X.shape[1]):
            if self.kdes_[i] is not None:
                X_smoothed[:, i] = self.kdes_[i](X[:, i])
            else:
                X_smoothed[:, i] = X[:, i]
        return X_smoothed


class KMeansClusterTransformer(BaseEstimator, TransformerMixin):
    """Adds K-Means cluster membership as a feature."""
    
    def __init__(self, n_clusters=5, add_distances=True, random_state=None):
        self.n_clusters = n_clusters
        self.add_distances = add_distances
        self.random_state = random_state
        
    def fit(self, X, y=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=100
        )
        self.kmeans_.fit(X)
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = 1 + (self.n_clusters if self.add_distances else 0)
        return self
    
    def transform(self, X):
        cluster_labels = self.kmeans_.predict(X).reshape(-1, 1)
        if self.add_distances:
            distances = self.kmeans_.transform(X)
            return np.hstack([cluster_labels, distances])
        else:
            return cluster_labels


class NoiseInjector(BaseEstimator, TransformerMixin):
    """Add deliberate noise to features (DISABLED during inference).
    
    During training, this adds noise for diversity.
    During inference, this is a pass-through (no noise).
    """
    
    def __init__(self, feature_fraction=None, noise_distributions=None, 
                 noise_scale_range=(0.01, 0.2), random_state=None, 
                 inference_mode=False):
        self.feature_fraction = feature_fraction
        self.noise_distributions = noise_distributions
        self.noise_scale_range = noise_scale_range
        self.random_state = random_state
        self.inference_mode = inference_mode
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Always pass-through during inference (no noise)
        return np.asarray(X, dtype=np.float64)
