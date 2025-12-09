"""Ensemble classifier with all custom transformers bundled.

This single file contains:
1. All custom feature engineering transformers
2. EnsembleClassifier wrapper class

Upload this file to Kaggle dataset along with the ensemble_model.joblib file.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde


# ============================================================================
# CUSTOM TRANSFORMERS
# ============================================================================

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
        
        # Safety check: If input has 0 features, we can't proceed
        if n_features == 0:
            raise ValueError("RandomFeatureSelector received 0 features. Check upstream transformers.")
        
        n_selected = max(1, int(n_features * self.feature_fraction))
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
        self.inference_mode = inference_mode  # Set to True for inference
        
    def fit(self, X, y=None):
        """Fit is pass-through - noise parameters were set during training."""
        return self
    
    def transform(self, X):
        """During inference, return data unchanged (no noise)."""
        # Always return unchanged during inference
        # The fitted model already has noise_configs_ if it was trained with noise
        # But we never apply noise during inference
        return np.asarray(X, dtype=np.float64)


# ============================================================================
# ENSEMBLE CLASSIFIER WRAPPER
# ============================================================================

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for ensemble model.
    
    Combines multiple stage 1 models and an optional stage 2 aggregator
    into a single sklearn-compatible classifier.
    
    Parameters
    ----------
    ensemble_models : list of fitted sklearn pipelines
        Stage 1 models (each is a complete pipeline: preprocessing + classifier)
    stage2_model : keras.Model or None
        Optional stage 2 DNN aggregator. If None, uses simple averaging.
    aggregation : str, default='mean'
        How to aggregate stage 1 predictions when stage2_model is None.
        Options: 'mean', 'median'
    
    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels [0, 1]
    n_models_ : int
        Number of stage 1 models in the ensemble
    
    Examples
    --------
    >>> # After training ensemble
    >>> from ensemble_classifier import EnsembleClassifier
    >>> wrapper = EnsembleClassifier(ensemble_models, stage2_model)
    >>> 
    >>> # Save
    >>> import joblib
    >>> joblib.dump(wrapper, 'ensemble_model.joblib')
    >>> 
    >>> # Load and predict (in inference notebook)
    >>> from ensemble_classifier import EnsembleClassifier
    >>> model = joblib.load('ensemble_model.joblib')
    >>> predictions = model.predict(test_df)
    """
    
    def __init__(self, ensemble_models, stage2_model=None, aggregation='mean'):
        self.ensemble_models = ensemble_models
        self.stage2_model = stage2_model
        self.aggregation = aggregation
        self.classes_ = np.array([0, 1])
        self.n_models_ = len(ensemble_models)
    
    def fit(self, X, y=None):
        """Not implemented - model is already trained.
        
        Raises
        ------
        NotImplementedError
            This is a pre-trained model wrapper.
        """
        raise NotImplementedError(
            "EnsembleClassifier is a wrapper for pre-trained models. "
            "Train individual models separately, then wrap them."
        )
    
    def predict_proba(self, X):
        """Predict class probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data (must include 'id' column and all feature columns)
        
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probability estimates for each class [P(class_0), P(class_1)]
        """
        # Stage 1: Get predictions from all ensemble models
        stage1_predictions = []
        
        for model in self.ensemble_models:
            try:
                if hasattr(model, 'predict_proba'):
                    # Most classifiers have predict_proba
                    pred = model.predict_proba(X)[:, 1]  # Probability of class 1
                else:
                    # Linear models without predict_proba (e.g., SVC with decision_function)
                    pred = model.decision_function(X)
                    # Normalize decision function to [0, 1] range (rough approximation)
                    pred = 1 / (1 + np.exp(-pred))  # Sigmoid
                
                stage1_predictions.append(pred)
            except Exception as e:
                # If a model fails, skip it (shouldn't happen with well-trained models)
                print(f"Warning: Model failed during prediction: {e}")
                continue
        
        if len(stage1_predictions) == 0:
            raise RuntimeError("All ensemble models failed during prediction")
        
        # Stack predictions: shape (n_samples, n_models)
        stage1_predictions = np.column_stack(stage1_predictions)
        
        # Stage 2: Aggregate predictions
        if self.stage2_model is not None:
            # Use DNN aggregator
            try:
                proba_class1 = self.stage2_model.predict(stage1_predictions, verbose=0).flatten()
            except Exception as e:
                print(f"Warning: Stage 2 model failed, falling back to mean aggregation: {e}")
                proba_class1 = np.mean(stage1_predictions, axis=1)
        else:
            # Simple aggregation (no stage 2 model)
            if self.aggregation == 'mean':
                proba_class1 = np.mean(stage1_predictions, axis=1)
            elif self.aggregation == 'median':
                proba_class1 = np.median(stage1_predictions, axis=1)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Clip to valid probability range
        proba_class1 = np.clip(proba_class1, 0.0, 1.0)
        
        # Return full probability matrix: [P(class_0), P(class_1)]
        proba_class0 = 1.0 - proba_class1
        return np.column_stack([proba_class0, proba_class1])
    
    def predict(self, X):
        """Predict class labels.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data (must include 'id' column and all feature columns)
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def __repr__(self):
        """String representation."""
        stage2_info = "DNN aggregator" if self.stage2_model is not None else f"{self.aggregation} aggregation"
        return (f"EnsembleClassifier(n_models={self.n_models_}, "
                f"stage2={stage2_info})")
