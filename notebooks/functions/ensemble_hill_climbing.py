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
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ensemble_transformers import (
    RandomFeatureSelector, RatioTransformer, ProductTransformer,
    DifferenceTransformer, SumTransformer, ReciprocalTransformer,
    SquareTransformer, SquareRootTransformer, LogTransformer,
    BinningTransformer, KDESmoothingTransformer
)


def generate_random_pipeline(
    iteration: int,
    random_state: int,
    base_preprocessor: Any
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Generate a random stage 1 pipeline with diverse feature engineering.
    
    Parameters
    ----------
    iteration : int
        Current iteration number (affects row sampling strategy).
    random_state : int
        Random state for reproducibility.
    base_preprocessor : sklearn transformer
        Base preprocessing pipeline (column transformer for encoding).
    
    Returns
    -------
    pipeline : Pipeline
        sklearn pipeline with random configuration.
    metadata : dict
        Dictionary containing pipeline configuration details.
    """
    rng = np.random.RandomState(random_state)
    
    # Adaptive row sampling: larger samples early, smaller later
    if iteration < 100:
        row_sample_pct = rng.uniform(0.30, 0.50)
    else:
        row_sample_pct = rng.uniform(0.10, 0.30)
    
    # Random column sampling
    col_sample_pct = rng.uniform(0.50, 0.95)
    
    # Select 1-3 feature engineering transformers
    n_transformers = rng.randint(1, 4)
    
    # Available transformers
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
        ('kde', KDESmoothingTransformer)
    ]
    
    # Randomly select transformers
    selected_transformer_indices = rng.choice(
        len(transformer_options),
        size=n_transformers,
        replace=False
    )
    selected_transformers = [transformer_options[i] for i in selected_transformer_indices]
    
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
                random_state=rng.randint(0, 100000)
            )
        elif name == 'binning':
            n_bins = rng.randint(3, 11)
            strategy = rng.choice(['quantile', 'uniform'])
            transformer = TransformerClass(
                n_bins=n_bins,
                strategy=strategy,
                encode='ordinal',
                random_state=rng.randint(0, 100000)
            )
        elif name == 'kde':
            bandwidth = rng.choice(['scott', 'silverman'])
            transformer = TransformerClass(
                bandwidth=bandwidth,
                random_state=rng.randint(0, 100000)
            )
        else:
            # Simple transformers
            transformer = TransformerClass()
        
        feature_steps.append((name, transformer))
    
    # Add column selector
    feature_steps.insert(0, ('column_selector', RandomFeatureSelector(
        feature_fraction=col_sample_pct,
        random_state=rng.randint(0, 100000)
    )))
    
    # Optionally add PCA
    use_pca = rng.random() < 0.5
    if use_pca:
        n_components = rng.randint(10, 201)
        feature_steps.append(('pca', PCA(
            n_components=n_components,
            random_state=rng.randint(0, 100000)
        )))
    
    # Add standard scaler before classifier
    feature_steps.append(('scaler', StandardScaler()))
    
    # Select classifier
    classifier_options = [
        'logistic_regression',
        'random_forest',
        'gradient_boosting',
        'svc',
        'mlp',
        'knn',
        'extra_trees',
        'adaboost'
    ]
    
    classifier_type = rng.choice(classifier_options)
    
    # Create classifier with random hyperparameters
    if classifier_type == 'logistic_regression':
        C = 10 ** rng.uniform(-3, 2)
        penalty = rng.choice(['l2', None])
        classifier = LogisticRegression(
            C=C,
            penalty=penalty,
            max_iter=1000,
            class_weight='balanced',
            random_state=rng.randint(0, 100000)
        )
    
    elif classifier_type == 'random_forest':
        n_estimators = rng.choice([50, 100, 200])
        max_depth = rng.choice([5, 10, 15, 20, None])
        min_samples_split = rng.randint(2, 11)
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=rng.randint(0, 100000),
            n_jobs=1
        )
    
    elif classifier_type == 'gradient_boosting':
        n_estimators = rng.choice([50, 100, 200])
        learning_rate = 10 ** rng.uniform(-2, 0)
        max_depth = rng.randint(3, 8)
        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=rng.randint(0, 100000)
        )
    
    elif classifier_type == 'svc':
        C = 10 ** rng.uniform(-2, 2)
        kernel = rng.choice(['rbf', 'linear'])
        classifier = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            class_weight='balanced',
            random_state=rng.randint(0, 100000)
        )
    
    elif classifier_type == 'mlp':
        hidden_layer_sizes = tuple(rng.choice([32, 64, 128], size=rng.randint(1, 3)))
        alpha = 10 ** rng.uniform(-4, -1)
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=500,
            random_state=rng.randint(0, 100000)
        )
    
    elif classifier_type == 'knn':
        n_neighbors = rng.randint(3, 21)
        weights = rng.choice(['uniform', 'distance'])
        classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            n_jobs=1
        )
    
    elif classifier_type == 'extra_trees':
        n_estimators = rng.choice([50, 100, 200])
        max_depth = rng.choice([5, 10, 15, 20, None])
        classifier = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=rng.randint(0, 100000),
            n_jobs=1
        )
    
    elif classifier_type == 'adaboost':
        n_estimators = rng.choice([50, 100, 200])
        learning_rate = 10 ** rng.uniform(-1, 0)
        classifier = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=rng.randint(0, 100000)
        )
    
    # Build complete pipeline
    pipeline = Pipeline([
        ('preprocessor', base_preprocessor),
        ('feature_engineering', Pipeline(feature_steps)),
        ('classifier', classifier)
    ])
    
    # Create metadata
    metadata = {
        'row_sample_pct': row_sample_pct,
        'col_sample_pct': col_sample_pct,
        'transformers_used': transformer_names,
        'use_pca': use_pca,
        'n_pca_components': n_components if use_pca else None,
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
    """Perform quick hyperparameter optimization on a pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        sklearn pipeline to optimize.
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training labels.
    n_iter : int, default=10
        Number of RandomizedSearchCV iterations.
    cv : int, default=3
        Number of cross-validation folds.
    n_jobs : int, default=8
        Number of parallel jobs.
    random_state : int, default=42
        Random state for reproducibility.
    
    Returns
    -------
    best_pipeline : Pipeline
        Optimized pipeline.
    best_score : float
        Best cross-validation ROC-AUC score.
    """
    # Define simple parameter distributions for quick optimization
    param_distributions = {}
    
    classifier_type = type(pipeline.named_steps['classifier']).__name__
    
    if classifier_type == 'LogisticRegression':
        param_distributions = {
            'classifier__C': loguniform(0.01, 10)
        }
    elif 'Forest' in classifier_type or 'Trees' in classifier_type:
        param_distributions = {
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': randint(2, 11)
        }
    elif classifier_type == 'GradientBoostingClassifier':
        param_distributions = {
            'classifier__learning_rate': loguniform(0.01, 0.3),
            'classifier__max_depth': randint(3, 8)
        }
    elif classifier_type == 'SVC':
        param_distributions = {
            'classifier__C': loguniform(0.1, 10)
        }
    elif classifier_type == 'MLPClassifier':
        param_distributions = {
            'classifier__alpha': loguniform(1e-4, 1e-2)
        }
    
    if param_distributions:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            n_jobs=n_jobs,
            random_state=random_state,
            error_score='raise'
        )
        
        try:
            search.fit(X, y)
            return search.best_estimator_, search.best_score_
        except Exception as e:
            # If optimization fails, return original pipeline with score 0
            print(f"Optimization failed: {e}")
            return pipeline, 0.0
    else:
        # No hyperparameters to optimize, just fit and score
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                pipeline, X, y,
                cv=cv,
                scoring='roc_auc',
                n_jobs=n_jobs
            )
            pipeline.fit(X, y)
            return pipeline, scores.mean()
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            return pipeline, 0.0


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
        'use_pca': metadata.get('use_pca', False),
        'n_pca_components': metadata.get('n_pca_components'),
        'row_sample_pct': round(metadata.get('row_sample_pct', 0), 4),
        'col_sample_pct': round(metadata.get('col_sample_pct', 0), 4)
    }, sort_keys=True)
    
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def log_iteration(
    iteration: int,
    fold: int,
    accepted: bool,
    rejection_reason: str,
    pipeline_hash: str,
    stage1_cv_score: float,
    stage1_val_auc: float,
    stage2_val_auc: float,
    ensemble_size: int,
    diversity_score: float,
    temperature: float,
    metadata: Dict[str, Any],
    ensemble_id: str
) -> None:
    """Log iteration details to SQLite database for dashboard monitoring.
    
    Parameters
    ----------
    iteration : int
        Iteration number.
    fold : int
        Current validation fold.
    accepted : bool
        Whether candidate was accepted.
    rejection_reason : str
        Reason for acceptance/rejection.
    pipeline_hash : str
        Hash of pipeline configuration.
    stage1_cv_score : float
        Stage 1 cross-validation ROC-AUC score.
    stage1_val_auc : float
        Stage 1 validation fold ROC-AUC score.
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
    """
    try:
        # Serialize transformers list to comma-separated string
        transformers_used = ','.join(metadata.get('transformers_used', []))
        
        # Calculate combined score (for dashboard sorting)
        combined_score = stage2_val_auc
        
        iteration_data = {
            'timestamp': datetime.now().isoformat(),
            'iteration_num': iteration,
            'ensemble_id': ensemble_id,
            'cv_score': stage1_cv_score,
            'diversity_score': diversity_score,
            'combined_score': combined_score,
            'temperature': temperature,
            'accepted': 1 if accepted else 0,
            'acceptance_reason': rejection_reason,
            'num_models': ensemble_size,
            'transformers_used': transformers_used,
            'pipeline_hash': pipeline_hash
        }
        
        ensemble_database.insert_ensemble_iteration(iteration_data)
        
    except Exception as e:
        print(f"Warning: Failed to log iteration {iteration}: {e}")
