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
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .ensemble_transformers import (
    RandomFeatureSelector, RatioTransformer, ProductTransformer,
    DifferenceTransformer, SumTransformer, ReciprocalTransformer,
    SquareTransformer, SquareRootTransformer, LogTransformer,
    BinningTransformer, KDESmoothingTransformer
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
    base_preprocessor: Any
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Generate a random stage 1 pipeline with diverse feature engineering.
    
    Parameters
    ----------
    iteration : int
        Current iteration number (affects row sampling strategy).
    random_state : int
        DEPRECATED - Not used. Kept for backwards compatibility.
    base_preprocessor : sklearn transformer
        Base preprocessing pipeline (column transformer for encoding).
    
    Returns
    -------
    pipeline : Pipeline
        sklearn pipeline with random configuration.
    metadata : dict
        Dictionary containing pipeline configuration details.
    """
    rng = np.random
    
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
                random_state=None  # No random state for diversity
            )
        elif name == 'binning':
            n_bins = rng.randint(3, 11)
            strategy = rng.choice(['quantile', 'uniform'])
            transformer = TransformerClass(
                n_bins=n_bins,
                strategy=strategy,
                encode='ordinal',
                random_state=None  # No random state for diversity
            )
        elif name == 'kde':
            bandwidth = rng.choice(['scott', 'silverman'])
            transformer = TransformerClass(
                bandwidth=bandwidth,
                random_state=None  # No random state for diversity
            )
        else:
            # Simple transformers
            transformer = TransformerClass()
        
        feature_steps.append((name, transformer))
    
    # Add constant feature remover (always first to clean up after preprocessing)
    feature_steps.insert(0, ('constant_remover', ConstantFeatureRemover()))
    
    # Add column selector
    feature_steps.insert(0, ('column_selector', RandomFeatureSelector(
        feature_fraction=col_sample_pct,
        random_state=None  # No random state for diversity
    )))
    
    # Optionally add PCA
    use_pca = rng.random() < 0.5
    if use_pca:
        # Use variance-based selection to avoid dimensionality issues
        pca_options = [0.90, 0.95, 0.99, 'mle']
        n_components = rng.choice(pca_options)
        # Convert numpy scalar to Python type if needed
        if n_components != 'mle':
            n_components = float(n_components)
        feature_steps.append(('pca', PCA(
            n_components=n_components,
            random_state=None  # No random state for diversity
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
    
    # Create classifier with random hyperparameters (wide distributions for diversity)
    if classifier_type == 'logistic_regression':
        C = 10 ** rng.uniform(-4, 3)  # 0.0001 to 1000
        penalty = rng.choice(['l2', None])
        solver = 'saga' if penalty is None else 'lbfgs'
        classifier = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=2000,
            class_weight='balanced'
            # No random_state for diversity
        )
    
    elif classifier_type == 'random_forest':
        n_estimators = int(10 ** rng.uniform(1.5, 2.5))  # ~30 to 300
        max_depth = rng.choice([3, 5, 7, 10, 15, 20, 30, None])
        min_samples_split = int(10 ** rng.uniform(0.3, 1.3))  # 2 to 20
        min_samples_leaf = int(10 ** rng.uniform(0, 1))  # 1 to 10
        max_features = rng.choice(['sqrt', 'log2', None])
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            n_jobs=1
            # No random_state for diversity
        )
    
    elif classifier_type == 'gradient_boosting':
        max_iter = int(10 ** rng.uniform(1.5, 2.5))  # ~30 to 300
        learning_rate = 10 ** rng.uniform(-2.5, 0)  # 0.003 to 1.0
        max_depth = rng.choice([None, 3, 5, 7, 10, 15, 20, 30])
        l2_regularization = 10 ** rng.uniform(-4, 1)  # 0.0001 to 10
        min_samples_leaf = int(10 ** rng.uniform(1, 2))  # 10 to 100
        max_bins = rng.choice([32, 64, 128, 255])
        classifier = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            l2_regularization=l2_regularization,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins
            # No random_state for diversity
        )
    
    elif classifier_type == 'svc':
        C = 10 ** rng.uniform(-3, 3)  # 0.001 to 1000
        kernel = rng.choice(['rbf', 'linear', 'poly'])
        gamma = rng.choice(['scale', 'auto']) if kernel in ['rbf', 'poly'] else 'scale'
        degree = rng.randint(2, 5) if kernel == 'poly' else 3
        classifier = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            probability=True,
            class_weight='balanced'
            # No random_state for diversity
        )
    
    elif classifier_type == 'mlp':
        n_layers = rng.randint(1, 4)  # 1 to 3 hidden layers
        layer_sizes = [int(10 ** rng.uniform(1.5, 2.5)) for _ in range(n_layers)]  # ~30 to 300 neurons
        hidden_layer_sizes = tuple(layer_sizes)
        alpha = 10 ** rng.uniform(-5, -1)  # 0.00001 to 0.1
        learning_rate_init = 10 ** rng.uniform(-4, -2)  # 0.0001 to 0.01
        activation = rng.choice(['relu', 'tanh', 'logistic'])
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            activation=activation,
            max_iter=1000,
            early_stopping=True
            # No random_state for diversity
        )
    
    elif classifier_type == 'knn':
        n_neighbors = int(10 ** rng.uniform(0.5, 1.5))  # 3 to 30
        weights = rng.choice(['uniform', 'distance'])
        p = rng.choice([1, 2])  # Manhattan or Euclidean
        leaf_size = int(10 ** rng.uniform(1, 2))  # 10 to 100
        classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
            leaf_size=leaf_size,
            n_jobs=1
        )
    
    elif classifier_type == 'extra_trees':
        n_estimators = int(10 ** rng.uniform(1.5, 2.5))  # ~30 to 300
        max_depth = rng.choice([3, 5, 7, 10, 15, 20, 30, None])
        min_samples_split = int(10 ** rng.uniform(0.3, 1.3))  # 2 to 20
        min_samples_leaf = int(10 ** rng.uniform(0, 1))  # 1 to 10
        max_features = rng.choice(['sqrt', 'log2', None])
        classifier = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            n_jobs=1
            # No random_state for diversity
        )
    
    elif classifier_type == 'adaboost':
        n_estimators = int(10 ** rng.uniform(1.5, 2.5))  # ~30 to 300
        learning_rate = 10 ** rng.uniform(-1.5, 0.5)  # 0.03 to 3.0
        classifier = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
            # No random_state for diversity
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
        'use_pca': metadata.get('use_pca', False),
        'n_pca_components': metadata.get('n_pca_components'),
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
    stage2_time_sec: Optional[float] = None
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
            'use_pca': 1 if metadata.get('use_pca', False) else 0,
            'pca_components': metadata.get('n_pca_components'),
            'pipeline_hash': pipeline_hash,
            'training_memory_mb': training_memory_mb,
            'stage2_memory_mb': stage2_memory_mb,
            'training_time_sec': training_time_sec,
            'stage2_time_sec': stage2_time_sec
        }
        
        ensemble_database.insert_ensemble_iteration(iteration_data)
        
    except Exception as e:
        print(f"Warning: Failed to log iteration {iteration}: {e}")
