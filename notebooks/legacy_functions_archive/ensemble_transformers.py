"""Custom feature engineering transformers for ensemble stage 1 models.

This module provides a collection of sklearn-compatible transformers for diverse
feature engineering. Each transformer can randomly select feature pairs during fit
to ensure diversity across ensemble members.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde


class CleanNumericTransformer(BaseEstimator, TransformerMixin):
    """Replace NaN and infinite values with median.
    
    Handles both NaN and +/- infinity values that can be introduced by
    feature engineering operations (log of negative, division by zero, etc.).
    
    Parameters
    ----------
    strategy : str, default='median'
        The imputation strategy. Currently only 'median' is supported.
    """
    
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.medians_ = None
    
    def fit(self, X, y=None):
        """Learn the median values for each column.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X_array = np.asarray(X, dtype=np.float64)
        
        # Replace infinities with NaN temporarily to compute median
        X_clean = X_array.copy()
        X_clean[~np.isfinite(X_clean)] = np.nan
        
        # Compute median for each column (ignoring NaN)
        self.medians_ = np.nanmedian(X_clean, axis=0)
        
        # Handle columns that are all NaN/inf - use 0
        self.medians_[np.isnan(self.medians_)] = 0.0
        
        return self
    
    def transform(self, X):
        """Replace NaN and infinite values with learned medians.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_clean : ndarray of shape (n_samples, n_features)
            Data with NaN and infinite values replaced.
        """
        X_array = np.asarray(X, dtype=np.float64)
        X_clean = X_array.copy()
        
        # Replace all non-finite values (NaN, +inf, -inf) with median
        for i in range(X_clean.shape[1]):
            mask = ~np.isfinite(X_clean[:, i])
            if np.any(mask):
                X_clean[mask, i] = self.medians_[i]
        
        return X_clean


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
        
        # Safety check: If input has 0 features, we can't proceed
        if n_features == 0:
            raise ValueError(f"RandomFeatureSelector received 0 features. Check upstream transformers.")
        
        n_selected = max(1, int(n_features * self.feature_fraction))
        # Ensure we don't try to sample more than available
        n_selected = min(n_selected, n_features)
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
        # Need at least 2 features to create pairs
        if n_input_features < 2:
            # Fall back to using same feature (will result in 1.0 after ratio)
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
        # Need at least 2 features to create pairs
        if n_input_features < 2:
            # Fall back to using same feature (will result in feature squared)
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
        # Need at least 2 features to create pairs
        if n_input_features < 2:
            # Fall back to using same feature (will result in 0)
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
        # Need at least 2 features to create pairs
        if n_input_features < 2:
            # Fall back to using same feature (will result in 2*feature)
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


class KMeansClusterTransformer(BaseEstimator, TransformerMixin):
    """Adds K-Means cluster membership as a feature.
    
    Creates cluster labels and optionally distances to cluster centers as new features.
    
    Parameters
    ----------
    n_clusters : int, default=5
        Number of clusters (3 to 10 recommended).
    add_distances : bool, default=True
        Whether to add distance to each cluster center as features.
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_clusters=5, add_distances=True, random_state=None):
        self.n_clusters = n_clusters
        self.add_distances = add_distances
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Fit KMeans clustering."""
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=100  # Limited for speed
        )
        self.kmeans_.fit(X)
        self.n_features_in_ = X.shape[1]
        # Output: cluster label + (optionally) distances to each cluster
        self.n_features_out_ = 1 + (self.n_clusters if self.add_distances else 0)
        return self
    
    def transform(self, X):
        """Transform by adding cluster membership and distances."""
        # Get cluster labels
        cluster_labels = self.kmeans_.predict(X).reshape(-1, 1)
        
        if self.add_distances:
            # Calculate distances to all cluster centers
            distances = self.kmeans_.transform(X)
            # Concatenate: [cluster_label, dist_to_cluster_0, dist_to_cluster_1, ...]
            return np.hstack([cluster_labels, distances])
        else:
            return cluster_labels
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        names = ['cluster_label']
        if self.add_distances:
            names.extend([f'dist_to_cluster_{i}' for i in range(self.n_clusters)])
        return np.array(names)


class NoiseInjector(BaseEstimator, TransformerMixin):
    """Add deliberate noise to features to increase ensemble diversity.
    
    Randomly selects a subset of features and adds noise from various distributions
    with different spreads. Each feature gets its own noise distribution and scale,
    creating diverse perturbations across ensemble members.
    
    This helps create diverse models even when training on similar data, as each
    model sees slightly different feature values.
    
    Parameters
    ----------
    feature_fraction : float, default=None
        Fraction of features to add noise to (0 to 1).
        If None, randomly selected during fit (0 to 1).
    
    noise_distributions : list of str, default=None
        List of allowed noise distributions: 'normal', 'uniform', 'laplace', 'exponential'
        If None, all distributions are available.
    
    noise_scale_range : tuple, default=(0.01, 0.2)
        Range for noise scale as fraction of feature standard deviation.
        Lower values = subtle noise, higher values = aggressive noise.
    
    random_state : int, default=None
        Random state for reproducibility. None for diversity.
    """
    
    def __init__(self, feature_fraction=None, noise_distributions=None, 
                 noise_scale_range=(0.01, 0.2), random_state=None):
        self.feature_fraction = feature_fraction
        self.noise_distributions = noise_distributions
        self.noise_scale_range = noise_scale_range
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Randomly select features and noise parameters for each.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        rng = np.random.RandomState(self.random_state)
        X_array = np.asarray(X, dtype=np.float64)
        n_features = X_array.shape[1]
        
        # Determine feature fraction if not set
        if self.feature_fraction is None:
            # Randomly select 0% to 100% of features
            feature_fraction = rng.uniform(0.0, 1.0)
        else:
            feature_fraction = self.feature_fraction
        
        # Select which features to add noise to
        n_noisy_features = max(0, int(n_features * feature_fraction))
        if n_noisy_features > 0:
            self.noisy_feature_indices_ = rng.choice(
                n_features, 
                size=n_noisy_features, 
                replace=False
            )
        else:
            self.noisy_feature_indices_ = np.array([], dtype=int)
        
        # Calculate feature standard deviations for noise scaling
        # Replace infinities and NaNs before computing std
        X_clean = X_array.copy()
        X_clean[~np.isfinite(X_clean)] = np.nan
        self.feature_stds_ = np.nanstd(X_clean, axis=0)
        # Replace zero/nan stds with 1.0 to avoid division issues
        self.feature_stds_[~np.isfinite(self.feature_stds_)] = 1.0
        self.feature_stds_[self.feature_stds_ == 0] = 1.0
        
        # Available distributions
        available_dists = self.noise_distributions or ['normal', 'uniform', 'laplace', 'exponential']
        
        # For each noisy feature, randomly select distribution and scale
        self.noise_configs_ = []
        for feat_idx in self.noisy_feature_indices_:
            # Random distribution
            distribution = rng.choice(available_dists)
            
            # Random noise scale (as fraction of feature std)
            noise_scale = rng.uniform(*self.noise_scale_range)
            
            # Actual noise standard deviation
            noise_std = noise_scale * self.feature_stds_[feat_idx]
            
            self.noise_configs_.append({
                'feature_idx': feat_idx,
                'distribution': distribution,
                'noise_std': noise_std
            })
        
        return self
    
    def transform(self, X):
        """Add noise to selected features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_noisy : ndarray of shape (n_samples, n_features)
            Data with noise added to selected features.
        """
        X_array = np.asarray(X, dtype=np.float64).copy()
        n_samples = X_array.shape[0]
        
        # Add noise to each selected feature
        for config in self.noise_configs_:
            feat_idx = config['feature_idx']
            distribution = config['distribution']
            noise_std = config['noise_std']
            
            # Generate noise based on distribution
            if distribution == 'normal':
                # Gaussian noise: mean=0, std=noise_std
                noise = np.random.normal(0, noise_std, size=n_samples)
            
            elif distribution == 'uniform':
                # Uniform noise: [-noise_std*sqrt(3), +noise_std*sqrt(3)]
                # (scaled so variance equals noise_std^2)
                width = noise_std * np.sqrt(3)
                noise = np.random.uniform(-width, width, size=n_samples)
            
            elif distribution == 'laplace':
                # Laplace noise: mean=0, scale parameter for std=noise_std
                # scale = noise_std / sqrt(2)
                scale = noise_std / np.sqrt(2)
                noise = np.random.laplace(0, scale, size=n_samples)
            
            elif distribution == 'exponential':
                # Exponential noise (centered): mean subtracted to center at 0
                # scale = noise_std (exponential std = scale)
                noise = np.random.exponential(noise_std, size=n_samples)
                noise = noise - noise_std  # Center at 0
            
            else:
                # Fallback to normal
                noise = np.random.normal(0, noise_std, size=n_samples)
            
            # Add noise to feature
            X_array[:, feat_idx] += noise
        
        return X_array
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names (unchanged from input)."""
        if input_features is None:
            return None
        return np.array(input_features)

