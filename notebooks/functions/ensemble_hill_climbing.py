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
from . import ensemble_config
import pandas as pd
from scipy.stats import uniform, loguniform, randint
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
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

# Import from models directory for constant feature removal and IQR clipping
import sys
from pathlib import Path
models_path = Path(__file__).resolve().parent.parent.parent / 'models'
sys.path.insert(0, str(models_path))
from logistic_regression_transformers import ConstantFeatureRemover, IQRClipper


def _generate_hyperparameters(rng: np.random.RandomState, hyperparam_config: Dict, **context) -> Dict:
    """Generate hyperparameters from config lambda functions.
    
    Parameters
    ----------
    rng : np.random.RandomState
        Random number generator.
    hyperparam_config : dict
        Dictionary of hyperparameter name to lambda function or constant value.
    **context : dict
        Context variables that may be needed by dependent hyperparameters
        (e.g., n_jobs, n_layers, solver, kernel, etc.)
    
    Returns
    -------
    hyperparams : dict
        Dictionary of hyperparameter name to generated value.
    """
    hyperparams = {}
    
    for param_name, param_value in hyperparam_config.items():
        if callable(param_value):
            # Get the function signature to determine what arguments it needs
            import inspect
            sig = inspect.signature(param_value)
            param_names = list(sig.parameters.keys())
            
            # Build kwargs for the lambda based on what it needs
            kwargs = {}
            if 'rng' in param_names:
                kwargs['rng'] = rng
            # Add any other context variables the lambda might need
            for ctx_key, ctx_value in context.items():
                if ctx_key in param_names:
                    kwargs[ctx_key] = ctx_value
            # Also add any previously generated hyperparameters (for dependent params)
            for prev_key, prev_value in hyperparams.items():
                if prev_key in param_names:
                    kwargs[prev_key] = prev_value
            
            hyperparams[param_name] = param_value(**kwargs)
        else:
            # Constant value
            hyperparams[param_name] = param_value
    
    return hyperparams


def generate_random_pipeline(
    iteration: int,
    random_state: int,
    base_preprocessor: Any,
    n_jobs: int = 1,
    n_input_features: int = 22
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
    n_input_features : int, optional
        Number of input features before preprocessing (default: 22).
        Used to calculate feature counts after column sampling.
    
    Returns
    -------
    pipeline : Pipeline
        sklearn pipeline with random configuration.
    metadata : dict
        Dictionary containing pipeline configuration details.
    """
    rng = np.random.RandomState(random_state)
    
    # Get sampling configuration from config
    row_sample_pct = rng.uniform(
        ensemble_config.SAMPLING_CONFIG['row_sample_pct']['min'],
        ensemble_config.SAMPLING_CONFIG['row_sample_pct']['max']
    )
    col_sample_pct = rng.uniform(
        ensemble_config.SAMPLING_CONFIG['col_sample_pct']['min'],
        ensemble_config.SAMPLING_CONFIG['col_sample_pct']['max']
    )
    
    # Feature engineering configuration from config
    skip_feature_engineering = rng.random() < ensemble_config.FEATURE_ENGINEERING_CONFIG['skip_probability']
    n_transformers = 0 if skip_feature_engineering else rng.randint(
        ensemble_config.FEATURE_ENGINEERING_CONFIG['n_transformers']['min'],
        ensemble_config.FEATURE_ENGINEERING_CONFIG['n_transformers']['max'] + 1
    )
    
    # Transformer class mapping
    transformer_class_map = {
        'ratio': RatioTransformer,
        'product': ProductTransformer,
        'difference': DifferenceTransformer,
        'sum': SumTransformer,
        'reciprocal': ReciprocalTransformer,
        'square': SquareTransformer,
        'sqrt': SquareRootTransformer,
        'log': LogTransformer,
        'binning': BinningTransformer,
        'iqr_clipper': IQRClipper,
        'kde': KDESmoothingTransformer,
        'kmeans': KMeansClusterTransformer,
        'nystroem': Nystroem,
        'rbf_sampler': RBFSampler,
        'power_transform': PowerTransformer,
        'quantile_transform': QuantileTransformer,
        'noise_injector': NoiseInjector
    }
    
    # Dimensionality reduction class mapping
    dim_reduction_class_map = {
        'pca': PCA,
        'truncated_svd': TruncatedSVD,
        'fast_ica': FastICA,
        'factor_analysis': FactorAnalysis
    }
    
    # Randomly select transformers from config
    available_transformers = ensemble_config.FEATURE_ENGINEERING_CONFIG['available_transformers']
    if n_transformers > 0:
        selected_transformer_names = rng.choice(
            available_transformers,
            size=min(n_transformers, len(available_transformers)),
            replace=False
        )
    else:
        selected_transformer_names = []
    
    # Build feature engineering pipeline steps
    feature_steps = []
    transformer_names = list(selected_transformer_names)
    
    # Calculate n_features after column sampling for transformers that need it
    n_features_after_sampling = int(n_input_features * col_sample_pct)
    
    for name in selected_transformer_names:
        TransformerClass = transformer_class_map[name]
        hyperparam_config = ensemble_config.TRANSFORMER_HYPERPARAMS.get(name, {})
        hyperparams = _generate_hyperparameters(rng, hyperparam_config, n_features=n_features_after_sampling)
        
        # Create transformer with generated hyperparameters
        transformer = TransformerClass(**hyperparams)
        feature_steps.append((name, transformer))
    
    # Add initial scaling BEFORE feature engineering
    scaler_choice = rng.choice(ensemble_config.INITIAL_SCALER_OPTIONS)
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
    
    # Optionally add dimensionality reduction from config
    use_dim_reduction = rng.random() < ensemble_config.DIM_REDUCTION_CONFIG['use_probability']
    dim_reduction_name = None
    if use_dim_reduction:
        # Randomly select one dimensionality reduction technique from config
        available_methods = ensemble_config.DIM_REDUCTION_CONFIG['available_methods']
        dim_reduction_name = rng.choice(available_methods)
        DimReductionClass = dim_reduction_class_map[dim_reduction_name]
        
        # Get hyperparameters from config
        # Calculate n_features after column sampling
        n_features_after_sampling = int(n_input_features * col_sample_pct)
        hyperparam_config = ensemble_config.DIM_REDUCTION_HYPERPARAMS[dim_reduction_name]
        hyperparams = _generate_hyperparameters(rng, hyperparam_config, n_features=n_features_after_sampling)
        
        # Create dimensionality reduction transformer
        dim_reducer = DimReductionClass(**hyperparams)
        feature_steps.append((dim_reduction_name, dim_reducer))
    
    # Select classifier from config
    classifier_type = rng.choice(ensemble_config.ACTIVE_CLASSIFIERS)
    
    # Add final scaler before classifier
    feature_steps.append(('scaler', StandardScaler()))
    
    # Get classifier class and hyperparameters from config
    classifier_config = ensemble_config.CLASSIFIER_CONFIGS[classifier_type]
    ClassifierClass = classifier_config['class']
    hyperparam_config = classifier_config['hyperparameters']
    
    # Generate hyperparameters with n_jobs context for parallel classifiers
    hyperparams = _generate_hyperparameters(rng, hyperparam_config, n_jobs=n_jobs)
    
    # Handle special MLP case where layer_sizes needs to be derived from n_layers
    if classifier_type == 'mlp' and 'hidden_layer_sizes' not in hyperparams:
        # If config uses n_layers pattern, construct hidden_layer_sizes
        if 'layer_sizes' in hyperparams:
            hyperparams['hidden_layer_sizes'] = hyperparams.pop('layer_sizes')
        # Remove n_layers as it's not an MLP parameter
        hyperparams.pop('n_layers', None)
    
    # Create classifier instance
    classifier = ClassifierClass(**hyperparams)
    
    # Wrap classifiers without predict_proba in CalibratedClassifierCV
    # This ensures Stage 2 DNN gets proper [0,1] probabilities as input
    if classifier_type in ['linear_svc', 'ridge']:
        classifier = CalibratedClassifierCV(classifier, cv=3, method='sigmoid')
    
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
    y: pd.Series
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
    random_state: Optional[int] = None,
    diversity_score: float = 0.0,
    diversity_bonus_weight: float = 0.0
) -> Tuple[bool, str]:
    """Determine acceptance using standard simulated annealing.
    
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
    diversity_score : float, default=0.0
        Diversity measure (for logging only, not used in acceptance)
    diversity_bonus_weight : float, default=0.0
        Unused parameter (kept for backward compatibility)
    
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
    
    # Accept worse solutions probabilistically based on temperature
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
