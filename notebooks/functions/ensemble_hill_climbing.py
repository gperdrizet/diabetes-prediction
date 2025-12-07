"""Hill climbing functions for ensemble stage 1 model generation and optimization.

This module provides functions for generating diverse pipelines, evaluating diversity,
performing quick optimization, and managing the simulated annealing acceptance process.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from . import ensemble_database
import pandas as pd
from scipy.stats import uniform, loguniform, randint
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF, FactorAnalysis
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, QuantileTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler

from .ensemble_transformers import (
    CleanNumericTransformer, RandomFeatureSelector, RatioTransformer, ProductTransformer,
    DifferenceTransformer, SumTransformer, ReciprocalTransformer,
    SquareTransformer, SquareRootTransformer, LogTransformer,
    BinningTransformer, KDESmoothingTransformer, KMeansClusterTransformer, NoiseInjector
)

# Import from models directory for constant feature removal
import sys
from pathlib import Path
models_path = Path(__file__).resolve().parent.parent.parent / 'models'
sys.path.insert(0, str(models_path))
from logistic_regression_transformers import ConstantFeatureRemover


def generate_random_pipeline(
    iteration: int,
    random_state: int,
    base_preprocessor: Any,
    n_jobs: int = 1
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Generate a random stage 1 pipeline with diverse feature engineering.
    
    Parameters
    ----------
    iteration : int
        Current iteration number (affects row sampling strategy).
    random_state : int
        Random state for reproducibility and diversity.
    base_preprocessor : sklearn transformer
        Base preprocessing pipeline (column transformer for encoding).
    n_jobs : int, optional
        Number of CPU cores to allocate to this model (default: 1).
        Used for parallelizable classifiers like RandomForest, KNN, ExtraTrees.
    
    Returns
    -------
    pipeline : Pipeline
        sklearn pipeline with random configuration.
    metadata : dict
        Dictionary containing pipeline configuration details.
    """
    rng = np.random.RandomState(random_state)
    
    # Random column sampling (DIVERSITY BOOST: reduced from 50-95% to 30-70%)
    col_sample_pct = rng.uniform(0.30, 0.70)
    
    # DIVERSITY BOOST: 30% chance to skip ALL feature engineering (use raw features only)
    skip_feature_engineering = rng.random() < 0.30
    
    # Select 0-3 feature engineering transformers (0 if skipping)
    n_transformers = 0 if skip_feature_engineering else rng.randint(1, 4)
    
    # Available feature engineering transformers (excluding dimensionality reduction)
    transformer_options = [
        ('ratio', RatioTransformer),
        ('product', ProductTransformer),
        ('difference', DifferenceTransformer),
        ('sum', SumTransformer),
        ('reciprocal', ReciprocalTransformer),
        ('square', SquareTransformer),
        ('sqrt', SquareRootTransformer),
        ('log', LogTransformer),
        ('binning', BinningTransformer),
        ('kde', KDESmoothingTransformer),
        ('kmeans', KMeansClusterTransformer),
        ('nystroem', Nystroem),
        ('rbf_sampler', RBFSampler),
        # Note: SkewedChi2Sampler removed - requires X > -skewedness which conflicts with StandardScaler
        ('power_transform', PowerTransformer),
        ('quantile_transform', QuantileTransformer),
        ('standard_scaler', StandardScaler),
        ('noise_injector', NoiseInjector)  # DIVERSITY BOOST: deliberate noise injection
    ]
    
    # Dimensionality reduction options (will select one or none)
    dim_reduction_options = [
        ('pca', PCA),
        ('truncated_svd', TruncatedSVD),
        ('fast_ica', FastICA),
        ('factor_analysis', FactorAnalysis)
    ]
    
    # Randomly select transformers (only if not skipping)
    if n_transformers > 0:
        selected_transformer_indices = rng.choice(
            len(transformer_options),
            size=n_transformers,
            replace=False
        )
        selected_transformers = [transformer_options[i] for i in selected_transformer_indices]
    else:
        selected_transformers = []
    
    # Build feature engineering pipeline steps
    feature_steps = []
    transformer_names = []
    
    for name, TransformerClass in selected_transformers:
        transformer_names.append(name)
        
        # Configure transformer with random parameters
        if name in ['ratio', 'product', 'difference', 'sum']:
            # Pairwise transformers: random number of features
            n_features = rng.randint(5, 31)
            transformer = TransformerClass(
                n_features=n_features,
                random_state=None  # No random state for diversity
            )
        elif name == 'binning':
            n_bins = rng.randint(3, 8)  # Reduced max bins from 10 to 7 to avoid small bin warnings
            strategy = rng.choice(['quantile', 'uniform'])
            transformer = TransformerClass(
                n_bins=n_bins,
                strategy=strategy,
                encode='ordinal'
                # No random_state parameter for KBinsDiscretizer
            )
        elif name == 'kde':
            bandwidth = rng.choice(['scott', 'silverman'])
            transformer = TransformerClass(
                bandwidth=bandwidth,
                random_state=None  # No random state for diversity
            )
        elif name == 'kmeans':
            # K-Means clustering features (cluster label + distances)
            n_clusters = rng.randint(3, 11)  # 3 to 10 clusters
            add_distances = rng.choice([True, False])  # Randomly add distances or not
            transformer = TransformerClass(
                n_clusters=n_clusters,
                add_distances=add_distances,
                random_state=None  # No random state for diversity
            )
        elif name == 'nystroem':
            # Approximate kernel feature map using subset of training data
            kernel = rng.choice(['rbf', 'poly', 'sigmoid', 'cosine'])
            n_components = int(10 ** rng.uniform(1.5, 2.5))  # 30 to 300 components
            gamma = 10 ** rng.uniform(-3, 0) if kernel in ['rbf', 'poly', 'sigmoid'] else None
            degree = rng.randint(2, 5) if kernel == 'poly' else 3
            transformer = TransformerClass(
                kernel=kernel,
                n_components=n_components,
                gamma=gamma,
                degree=degree,
                random_state=None  # No random state for diversity
            )
        elif name == 'rbf_sampler':
            # Approximates RBF kernel feature map using random Fourier features
            n_components = int(10 ** rng.uniform(1.5, 2.5))  # 30 to 300 components
            gamma = 10 ** rng.uniform(-3, 0)  # 0.001 to 1.0
            transformer = TransformerClass(
                n_components=n_components,
                gamma=gamma,
                random_state=None  # No random state for diversity
            )
        elif name == 'power_transform':
            # Transforms data to be more Gaussian-like
            # Only use yeo-johnson as it handles negative values (box-cox requires strictly positive data)
            standardize = rng.choice([True, False])
            transformer = TransformerClass(
                method='yeo-johnson',
                standardize=standardize
                # No random_state parameter
            )
        elif name == 'quantile_transform':
            # Transform features to follow a uniform or normal distribution
            n_quantiles = rng.choice([100, 500, 1000])
            output_distribution = rng.choice(['uniform', 'normal'])
            transformer = TransformerClass(
                n_quantiles=n_quantiles,
                output_distribution=output_distribution,
                random_state=None  # No random state for diversity
            )
        elif name == 'standard_scaler':
            # Standardize features by removing mean and scaling to unit variance
            with_mean = rng.choice([True, False])
            with_std = rng.choice([True, False])
            # Ensure at least one is True
            if not with_mean and not with_std:
                with_std = True
            transformer = TransformerClass(
                with_mean=with_mean,
                with_std=with_std
            )
        elif name == 'noise_injector':
            # DIVERSITY BOOST: Add deliberate noise to features
            # Randomly select fraction of features to add noise to (0-100%)
            feature_fraction = rng.uniform(0.0, 1.0)
            # Randomly select noise scale range (fraction of feature std)
            noise_scale_min = rng.uniform(0.001, 0.05)  # Min: 0.1% to 5%
            noise_scale_max = rng.uniform(0.05, 0.3)    # Max: 5% to 30%
            noise_scale_range = (noise_scale_min, noise_scale_max)
            transformer = TransformerClass(
                feature_fraction=feature_fraction,
                noise_scale_range=noise_scale_range,
                random_state=None  # No random state - different noise each time
            )
        else:
            # Simple transformers
            transformer = TransformerClass()
        
        feature_steps.append((name, transformer))
    
    # Add initial scaling BEFORE feature engineering to prevent overflow
    # This is inserted at position 0, so it runs first after column selection
    scaler_choice = rng.choice(['standard', 'minmax', 'robust'])
    if scaler_choice == 'standard':
        initial_scaler = StandardScaler()
    elif scaler_choice == 'minmax':
        initial_scaler = MinMaxScaler()
    else:  # robust
        from sklearn.preprocessing import RobustScaler
        initial_scaler = RobustScaler()
    
    # Add NaN/Inf cleaner AFTER feature engineering (handles NaN/Inf from log, division, etc.)
    # Use median strategy as it's robust to outliers
    nan_inf_cleaner = CleanNumericTransformer(strategy='median')
    
    # Add constant feature remover (always first to clean up after preprocessing)
    feature_steps.insert(0, ('constant_remover', ConstantFeatureRemover()))
    
    # Add initial scaler (before feature engineering to prevent overflow)
    feature_steps.insert(1, ('initial_scaler', initial_scaler))
    
    # Add NaN/Inf cleaner (after all transformations, before dim reduction)
    feature_steps.append(('nan_inf_cleaner', nan_inf_cleaner))
    
    # Add column selector
    feature_steps.insert(0, ('column_selector', RandomFeatureSelector(
        feature_fraction=col_sample_pct,
        random_state=None  # No random state for diversity
    )))
    
    # Initialize flag for non-negative feature requirement
    needs_nonnegative = False
    
    # Optionally add dimensionality reduction (50% chance)
    use_dim_reduction = rng.random() < 0.5
    dim_reduction_name = None
    if use_dim_reduction:
        # Randomly select one dimensionality reduction technique
        idx = rng.randint(0, len(dim_reduction_options))
        dim_reduction_name, DimReductionClass = dim_reduction_options[idx]
        
        # Configure based on technique
        if dim_reduction_name == 'pca':
            # Use variance-based selection to avoid dimensionality issues
            pca_options = [0.90, 0.95, 0.99, 'mle']
            n_components = rng.choice(pca_options)
            # Convert numpy scalar to Python type if needed
            if n_components != 'mle':
                n_components = float(n_components)
            dim_reducer = DimReductionClass(
                n_components=n_components,
                svd_solver='full',  # More robust to edge cases (constant features, low variance)
                random_state=None  # No random state for diversity
            )
        elif dim_reduction_name == 'truncated_svd':
            # Truncated SVD (works with sparse matrices, no centering)
            n_components = int(10 ** rng.uniform(0.7, 1.7))  # 5 to 50 components
            n_iter = rng.randint(5, 21)  # 5 to 20 iterations
            dim_reducer = DimReductionClass(
                n_components=n_components,
                n_iter=n_iter,
                random_state=None  # No random state for diversity
            )
        elif dim_reduction_name == 'fast_ica':
            # Independent Component Analysis (finds independent sources)
            algorithm = rng.choice(['parallel', 'deflation'])
            fun = rng.choice(['logcosh', 'exp', 'cube'])
            max_iter = rng.randint(200, 1001)  # 200 to 1000 iterations
            # Explicit whiten setting: randomly choose from valid options
            # Valid values: 'unit-variance', 'arbitrary-variance', or False
            whiten_choice = rng.randint(0, 3)
            if whiten_choice == 0:
                whiten = 'unit-variance'
            elif whiten_choice == 1:
                whiten = 'arbitrary-variance'
            else:
                whiten = False
            
            # Only set n_components when whiten is enabled (sklearn requirement)
            if whiten is False:
                dim_reducer = DimReductionClass(
                    algorithm=algorithm,
                    fun=fun,
                    max_iter=max_iter,
                    whiten=whiten,
                    random_state=None  # No random state for diversity
                )
            else:
                n_components = int(10 ** rng.uniform(0.7, 1.7))  # 5 to 50 components
                dim_reducer = DimReductionClass(
                    n_components=n_components,
                    algorithm=algorithm,
                    fun=fun,
                    max_iter=max_iter,
                    whiten=whiten,
                    random_state=None  # No random state for diversity
                )
        elif dim_reduction_name == 'factor_analysis':
            # Factor Analysis (similar to PCA but with noise modeling)
            n_components = int(10 ** rng.uniform(0.7, 1.7))  # 5 to 50 components
            max_iter = rng.randint(100, 501)  # 100 to 500 iterations
            dim_reducer = DimReductionClass(
                n_components=n_components,
                max_iter=max_iter,
                random_state=None  # No random state for diversity
            )
        
        feature_steps.append((dim_reduction_name, dim_reducer))
    
    # Select classifier first to determine scaling strategy
    classifier_options = [
        'logistic_regression',
        'random_forest',
        'gradient_boosting',
        'linear_svc',
        'sgd_classifier',
        'mlp',
        'knn',
        'extra_trees',
        'adaboost',
        'naive_bayes',
        'lda',
        'qda'
    ]
    
    classifier_type = rng.choice(classifier_options)
    
    # For Naive Bayes, we need to check which variant will be used
    # MultinomialNB requires non-negative features
    if classifier_type == 'naive_bayes':
        nb_type = rng.choice(['gaussian', 'multinomial', 'bernoulli'])
        if nb_type == 'multinomial':
            needs_nonnegative = True
    
    # Add appropriate scaler before classifier
    if needs_nonnegative:
        # MinMaxScaler ensures non-negative features for MultinomialNB
        feature_steps.append(('scaler', MinMaxScaler()))
    else:
        # StandardScaler for all other classifiers
        feature_steps.append(('scaler', StandardScaler()))
    
    # DIVERSITY BOOST: Uniform low row sampling for ALL classifiers (2.5-30%)
    # This maximizes training data diversity - different models see very different subsets
    # Old approach: 1.25-100% varied by classifier (too much overlap)
    # New approach: 2.5-30% uniform (minimal overlap, maximum diversity)
    row_sample_pct = rng.uniform(0.025, 0.30)
    
    # Create classifier with random hyperparameters (wide distributions for diversity)
    if classifier_type == 'logistic_regression':
        # SPEED FIX: Removed saga solver (too slow), kept fast solvers only
        C = 10 ** rng.uniform(-3, 2)  # 0.001 to 100
        penalty = rng.choice(['l1', 'l2', None])
        
        # Fast solvers only - no saga or sag
        if penalty == 'l1':
            solver = 'liblinear'  # Only fast solver for l1
        elif penalty == 'l2':
            solver = rng.choice(['lbfgs', 'liblinear', 'newton-cg'])  # Fast solvers for l2
        else:  # penalty is None
            solver = rng.choice(['lbfgs', 'newton-cg'])  # Fast solvers for no penalty
        
        classifier = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=rng.choice([500, 1000]),  # Reasonable iterations
            class_weight='balanced'
            # No random_state for diversity
        )
    
    elif classifier_type == 'random_forest':
        n_estimators = int(10 ** rng.uniform(1.0, 2.0))  # ~10 to 100
        max_depth = rng.choice([3, 5, 7, 10, 15, 20, None])  # Reduced deep trees for speed
        min_samples_split = int(10 ** rng.uniform(0.3, 1.3))  # 2 to 20
        min_samples_leaf = int(10 ** rng.uniform(0, 1))  # 1 to 10
        max_features = rng.choice(['sqrt', 'log2', None])
        # Use allocated cores for parallel tree building
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            n_jobs=n_jobs if n_jobs > 1 else 1
            # No random_state for diversity
        )
    
    elif classifier_type == 'gradient_boosting':
        # SPEED FIX: Reduced max_iter, max_depth, and min learning_rate to prevent 30min+ timeouts
        max_iter = int(10 ** rng.uniform(1.0, 1.7))  # ~10 to 50 (was 10-100)
        learning_rate = 10 ** rng.uniform(-2.0, 0)  # 0.01 to 1.0 (was 0.003-1.0)
        max_depth = rng.choice([None, 3, 5, 7, 10])  # Max 10 (was up to 20)
        l2_regularization = 10 ** rng.uniform(-4, 1)  # 0.0001 to 10
        min_samples_leaf = int(10 ** rng.uniform(1, 2))  # 10 to 100
        max_bins = rng.choice([32, 64, 128, 255])  # Keep full range
        classifier = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            l2_regularization=l2_regularization,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins
            # No random_state for diversity
        )
    
    elif classifier_type == 'linear_svc':
        # SPEED FIX: Reduced C range, max_iter, and use dual=False for speed
        C = 10 ** rng.uniform(-1, 1)  # 0.1 to 10 (narrower range)
        loss = 'squared_hinge'  # Only squared_hinge (faster, works with dual=False)
        # dual=False is much faster when n_features >> n_samples (after transformations)
        # Also relaxed tolerance for faster convergence
        classifier = LinearSVC(
            C=C,
            loss=loss,
            max_iter=rng.choice([200, 300]),  # Further reduced
            class_weight='balanced',
            dual=False,  # Much faster with many features
            tol=1e-3  # Relaxed from default 1e-4 for faster convergence
            # No random_state for diversity
        )
    
    elif classifier_type == 'sgd_classifier':
        loss = rng.choice(['hinge', 'log_loss', 'modified_huber', 'perceptron'])
        penalty = rng.choice(['l2', 'l1', 'elasticnet'])
        alpha = 10 ** rng.uniform(-5, -1)  # 0.00001 to 0.1
        learning_rate = rng.choice(['optimal', 'adaptive', 'constant'])
        eta0 = 10 ** rng.uniform(-4, -1)  # 0.0001 to 0.1 (required for adaptive/constant)
        classifier = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            learning_rate=learning_rate,
            eta0=eta0,
            max_iter=rng.choice([300, 500, 800]),  # Varied iterations for diversity
            early_stopping=True,
            class_weight='balanced'
            # No random_state for diversity
        )
    
    elif classifier_type == 'mlp':
        n_layers = rng.randint(1, 4)  # 1 to 3 hidden layers
        layer_sizes = [int(10 ** rng.uniform(1.3, 2.3)) for _ in range(n_layers)]  # ~20 to 200 neurons (reduced)
        hidden_layer_sizes = tuple(layer_sizes)
        alpha = 10 ** rng.uniform(-5, -1)  # 0.00001 to 0.1
        learning_rate_init = 10 ** rng.uniform(-4, -2)  # 0.0001 to 0.01
        activation = rng.choice(['relu', 'tanh', 'logistic'])
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            activation=activation,
            max_iter=rng.choice([100, 150, 200]),  # Reduced from 200-300 for speed
            early_stopping=True
            # No random_state for diversity
        )
    
    elif classifier_type == 'knn':
        n_neighbors = int(10 ** rng.uniform(0.5, 1.5))  # 3 to 30
        weights = rng.choice(['uniform', 'distance'])
        p = rng.choice([1, 2])  # Manhattan or Euclidean
        leaf_size = int(10 ** rng.uniform(1, 2))  # 10 to 100
        # Use allocated cores for distance calculations
        classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
            leaf_size=leaf_size,
            n_jobs=n_jobs if n_jobs > 1 else 1
        )
    
    elif classifier_type == 'extra_trees':
        n_estimators = int(10 ** rng.uniform(1.0, 2.0))  # ~10 to 100
        max_depth = rng.choice([3, 5, 7, 10, 15, 20, None])  # Reduced deep trees for speed
        min_samples_split = int(10 ** rng.uniform(0.3, 1.3))  # 2 to 20
        min_samples_leaf = int(10 ** rng.uniform(0, 1))  # 1 to 10
        max_features = rng.choice(['sqrt', 'log2', None])
        # Use allocated cores for parallel tree building
        classifier = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            n_jobs=n_jobs if n_jobs > 1 else 1
            # No random_state for diversity
        )
    
    elif classifier_type == 'adaboost':
        n_estimators = int(10 ** rng.uniform(1.0, 2.0))  # ~10 to 100
        learning_rate = 10 ** rng.uniform(-1.0, 0.5)  # 0.1 to 3.0 (wider, faster convergence)
        classifier = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm='SAMME'  # Use SAMME to avoid deprecated SAMME.R
            # No random_state for diversity
        )
    
    elif classifier_type == 'naive_bayes':
        # Use the pre-selected nb_type from above
        if nb_type == 'gaussian':
            # Gaussian Naive Bayes (works well with continuous features)
            var_smoothing = 10 ** rng.uniform(-12, -6)  # 1e-12 to 1e-6
            classifier = GaussianNB(
                var_smoothing=var_smoothing
            )
        elif nb_type == 'multinomial':
            # Multinomial Naive Bayes (good for count/frequency features)
            # Requires non-negative features (handled by MinMaxScaler above)
            alpha = 10 ** rng.uniform(-2, 1)  # 0.01 to 10
            fit_prior = rng.choice([True, False])
            classifier = MultinomialNB(
                alpha=alpha,
                fit_prior=fit_prior
            )
        else:  # bernoulli
            # Bernoulli Naive Bayes (good for binary/boolean features)
            alpha = 10 ** rng.uniform(-2, 1)  # 0.01 to 10
            binarize = rng.choice([None, 0.0, rng.uniform(0.3, 0.7)])
            fit_prior = rng.choice([True, False])
            classifier = BernoulliNB(
                alpha=alpha,
                binarize=binarize,
                fit_prior=fit_prior
            )
    
    elif classifier_type == 'gaussian_process':
        # Gaussian Process Classifier (probabilistic, kernel-based)
        # Very slow O(n³) - use small row samples and simpler kernels
        # Randomly select kernel type
        kernel_type = rng.choice(['rbf', 'matern', 'rational_quadratic', 'dot_product'])
        
        if kernel_type == 'rbf':
            # RBF (Radial Basis Function) kernel - smooth, infinite differentiable
            length_scale = 10 ** rng.uniform(-1, 1)  # 0.1 to 10
            kernel = RBF(length_scale=length_scale)
        elif kernel_type == 'matern':
            # Matern kernel - generalizes RBF, less smooth
            length_scale = 10 ** rng.uniform(-1, 1)  # 0.1 to 10
            nu = rng.choice([0.5, 1.5, 2.5])  # Controls smoothness
            kernel = Matern(length_scale=length_scale, nu=nu)
        elif kernel_type == 'rational_quadratic':
            # Rational Quadratic - mixture of RBF kernels with different length scales
            length_scale = 10 ** rng.uniform(-1, 1)  # 0.1 to 10
            alpha = 10 ** rng.uniform(-1, 1)  # 0.1 to 10
            kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha)
        else:  # dot_product
            # Dot Product kernel - equivalent to linear model in feature space
            sigma_0 = 10 ** rng.uniform(-1, 1)  # 0.1 to 10
            kernel = DotProduct(sigma_0=sigma_0)
        
        # Add white noise kernel for numerical stability
        noise_level = 10 ** rng.uniform(-5, -2)  # 1e-5 to 0.01
        kernel = kernel + WhiteKernel(noise_level=noise_level)
        
        # Limit max iterations and use warm start for faster convergence
        max_iter = rng.randint(15, 30)  # Very low due to O(n³) complexity, focus on speed
        
        # Adaptive n_jobs: GP is very slow O(n³), give it 3-5 cores if possible
        n_jobs = rng.choice([3, 4, 5])
        classifier = GaussianProcessClassifier(
            kernel=kernel,
            max_iter_predict=max_iter,
            warm_start=True,
            n_jobs=n_jobs,
            random_state=None  # No random state for diversity
        )
    
    elif classifier_type == 'lda':
        # Linear Discriminant Analysis - assumes Gaussian distributions with shared covariance
        # Fast O(n) and provides dimensionality reduction as side effect
        solver = rng.choice(['svd', 'lsqr', 'eigen'])
        shrinkage = None
        
        if solver == 'lsqr':
            # lsqr and eigen support shrinkage for regularization
            shrinkage = rng.choice([None, 'auto', rng.uniform(0.0, 1.0)])
        elif solver == 'eigen':
            shrinkage = rng.choice([None, 'auto', rng.uniform(0.0, 1.0)])
        
        classifier = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage
            # No random_state parameter
        )
    
    elif classifier_type == 'qda':
        # Quadratic Discriminant Analysis - assumes Gaussian with separate covariances
        # More flexible than LDA but needs more data, can model non-linear boundaries
        reg_param = 10 ** rng.uniform(-4, 0)  # 0.0001 to 1.0 regularization
        
        classifier = QuadraticDiscriminantAnalysis(
            reg_param=reg_param
            # No random_state parameter
        )
    
    # Build complete pipeline
    pipeline = Pipeline([
        ('preprocessor', clone(base_preprocessor)),  # Clone to avoid sharing fitted state
        ('feature_engineering', Pipeline(feature_steps)),
        ('classifier', classifier)
    ])
    
    # Create metadata
    metadata = {
        'row_sample_pct': row_sample_pct,
        'col_sample_pct': col_sample_pct,
        'transformers_used': transformer_names,
        'dim_reduction': dim_reduction_name,  # Which dimensionality reduction technique (or None)
        'classifier_type': classifier_type,
        'iteration': iteration
    }
    
    return pipeline, metadata


def calculate_ensemble_diversity(predictions: np.ndarray) -> float:
    """Calculate diversity score as mean pairwise prediction correlation.
    
    Parameters
    ----------
    predictions : np.ndarray
        Array of shape (n_samples, n_models) containing predictions.
    
    Returns
    -------
    diversity_score : float
        Mean pairwise correlation (lower is more diverse).
    """
    n_models = predictions.shape[1]
    
    if n_models < 2:
        return 0.0
    
    # Calculate pairwise correlations
    correlations = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            corr = np.corrcoef(predictions[:, i], predictions[:, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    return np.mean(correlations) if correlations else 0.0


def quick_optimize_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 10,
    cv: int = 3,
    n_jobs: int = 8,
    random_state: int = 42
) -> Tuple[Pipeline, float]:
    """Fit pipeline with pre-configured hyperparameters (no optimization, no CV).
    
    This function simply fits the pipeline with its pre-configured hyperparameters.
    No hyperparameter optimization or cross-validation is performed to maximize
    diversity and minimize training time.
    
    Parameters
    ----------
    pipeline : Pipeline
        sklearn pipeline to fit.
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training labels.
    n_iter : int, default=10
        DEPRECATED - Not used. Kept for backwards compatibility.
    cv : int, default=3
        DEPRECATED - Not used. Kept for backwards compatibility.
    n_jobs : int, default=8
        DEPRECATED - Not used. Kept for backwards compatibility.
    random_state : int, default=42
        DEPRECATED - Not used. Kept for backwards compatibility.
    
    Returns
    -------
    fitted_pipeline : Pipeline
        Fitted pipeline.
    dummy_score : float
        Always returns 0.0 (no CV performed).
    """
    try:
        # Just fit the pipeline - no cross-validation
        pipeline.fit(X, y)
        return pipeline, 0.0
        
    except Exception as e:
        print(f"Pipeline fitting failed: {e}")
        raise e


def adaptive_simulated_annealing_acceptance(
    current_score: float,
    candidate_score: float,
    temperature: float,
    random_state: Optional[int] = None
) -> Tuple[bool, str]:
    """Determine acceptance using simulated annealing with adaptive temperature.
    
    Parameters
    ----------
    current_score : float
        Current best ensemble ROC-AUC score.
    candidate_score : float
        Candidate model's contribution to ensemble ROC-AUC.
    temperature : float
        Current temperature parameter.
    random_state : int or None, default=None
        Random state for reproducibility.
    
    Returns
    -------
    accept : bool
        Whether to accept the candidate.
    reason : str
        Reason for acceptance/rejection.
    """
    rng = np.random.RandomState(random_state)
    
    delta = candidate_score - current_score
    
    # Always accept improvements
    if delta > 0:
        return True, f"improvement: Δ={delta:.6f}"
    
    # Accept worse solutions with probability based on temperature
    acceptance_probability = np.exp(delta / temperature)
    
    if rng.random() < acceptance_probability:
        return True, f"simulated_annealing: Δ={delta:.6f}, P={acceptance_probability:.6f}"
    else:
        return False, f"rejected: Δ={delta:.6f}, P={acceptance_probability:.6f}"


def update_temperature(
    iteration: int,
    acceptance_history: List[bool],
    current_temperature: float,
    base_temperature: float = 0.01,
    decay_rate: float = 0.995
) -> float:
    """Update temperature using adaptive strategy.
    
    Parameters
    ----------
    iteration : int
        Current iteration number.
    acceptance_history : list of bool
        Recent acceptance decisions (last 20).
    current_temperature : float
        Current temperature.
    base_temperature : float, default=0.01
        Initial temperature.
    decay_rate : float, default=0.995
        Exponential decay rate.
    
    Returns
    -------
    new_temperature : float
        Updated temperature.
    """
    # Check recent acceptance rate
    if len(acceptance_history) >= 20:
        recent_acceptance_rate = sum(acceptance_history[-20:]) / 20
        
        # If acceptance rate is too low, increase temperature
        if recent_acceptance_rate < 0.1:
            return current_temperature * 1.2
    
    # Otherwise, exponential decay
    return base_temperature * (decay_rate ** iteration)


def compute_pipeline_hash(pipeline: Pipeline, metadata: Dict[str, Any]) -> str:
    """Compute a hash of the pipeline configuration for tracking.
    
    Parameters
    ----------
    pipeline : Pipeline
        sklearn pipeline.
    metadata : dict
        Pipeline metadata.
    
    Returns
    -------
    hash_str : str
        SHA256 hash of pipeline configuration.
    """
    # Create string representation of pipeline config
    config_str = json.dumps({
        'transformers': metadata.get('transformers_used', []),
        'classifier': metadata.get('classifier_type', ''),
        'dim_reduction': metadata.get('dim_reduction'),
        'row_sample_pct': round(metadata.get('row_sample_pct', 0), 4),
        'col_sample_pct': round(metadata.get('col_sample_pct', 0), 4)
    }, sort_keys=True)
    
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def log_iteration(
    iteration: int,
    accepted: bool,
    rejection_reason: str,
    pipeline_hash: str,
    stage1_val_auc: float,
    stage2_val_auc: float,
    ensemble_size: int,
    diversity_score: float,
    temperature: float,
    metadata: Dict[str, Any],
    ensemble_id: str,
    training_memory_mb: Optional[float] = None,
    stage2_memory_mb: Optional[float] = None,
    training_time_sec: Optional[float] = None,
    stage2_time_sec: Optional[float] = None,
    timeout: bool = False
) -> None:
    """Log iteration details to SQLite database for dashboard monitoring.
    
    Parameters
    ----------
    iteration : int
        Iteration number.
    accepted : bool
        Whether candidate was accepted.
    rejection_reason : str
        Reason for acceptance/rejection.
    pipeline_hash : str
        Hash of pipeline configuration.
    stage1_val_auc : float
        Stage 1 validation ROC-AUC score.
    stage2_val_auc : float
        Stage 2 ensemble validation ROC-AUC score.
    ensemble_size : int
        Current ensemble size.
    diversity_score : float
        Ensemble diversity score.
    temperature : float
        Current temperature.
    metadata : dict
        Pipeline metadata.
    ensemble_id : str
        Unique identifier for this ensemble.
    training_memory_mb : float, optional
        Memory used during pipeline training (MB).
    stage2_memory_mb : float, optional
        Memory used during stage 2 DNN training (MB).
    training_time_sec : float, optional
        Time spent training pipeline (seconds).
    stage2_time_sec : float, optional
        Time spent training stage 2 DNN (seconds).
    timeout : bool, optional
        Whether training timed out (default: False).
    """
    try:
        # Serialize transformers list to comma-separated string
        transformers_used = ','.join(metadata.get('transformers_used', []))
        
        iteration_data = {
            'timestamp': datetime.now().isoformat(),
            'iteration_num': iteration,
            'ensemble_id': ensemble_id,
            'stage1_val_auc': stage1_val_auc,
            'stage2_val_auc': stage2_val_auc,
            'diversity_score': diversity_score,
            'temperature': temperature,
            'accepted': 1 if accepted else 0,
            'rejection_reason': rejection_reason,
            'num_models': ensemble_size,
            'classifier_type': metadata.get('classifier_type', ''),
            'transformers_used': transformers_used,
            'use_pca': 1 if metadata.get('dim_reduction') is not None else 0,
            'pca_components': metadata.get('dim_reduction'),
            'pipeline_hash': pipeline_hash,
            'training_memory_mb': training_memory_mb,
            'stage2_memory_mb': stage2_memory_mb,
            'training_time_sec': training_time_sec,
            'stage2_time_sec': stage2_time_sec,
            'timeout': 1 if timeout else 0
        }
        
        ensemble_database.insert_ensemble_iteration(iteration_data)
        
    except Exception as e:
        print(f"Warning: Failed to log iteration {iteration}: {e}")
