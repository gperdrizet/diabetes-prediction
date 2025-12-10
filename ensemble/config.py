"""Consolidated configuration for the entire ensemble system.

This module provides a type-safe, validated configuration structure using
dataclasses. All configuration parameters are consolidated here with:
- Clear documentation
- Type hints
- Validation logic
- Default values matching the original system

The configuration is organized hierarchically:
    EnsembleConfig (root)
    ├── HillClimbingConfig
    ├── ParallelConfig
    ├── SamplingConfig
    ├── Stage1Config
    │   ├── ClassifierConfig (per classifier)
    │   └── FeatureEngineeringConfig
    ├── Stage2Config
    │   ├── DNNArchitectureConfig
    │   ├── DNNTrainingConfig
    │   ├── OptimizationConfig
    │   └── PseudoLabelingConfig
    └── TrackingConfig

Usage:
    >>> from ensemble.config import EnsembleConfig
    >>> config = EnsembleConfig()  # Use defaults
    >>> config.validate()  # Check configuration validity
    
    >>> # Or customize
    >>> config = EnsembleConfig(
    ...     hill_climbing=HillClimbingConfig(max_iterations=2000),
    ...     stage2=Stage2Config(
    ...         pseudo_labeling=PseudoLabelingConfig(enabled=False)
    ...     )
    ... )
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Callable, Any
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


# ==============================================================================
# HILL CLIMBING CONFIGURATION
# ==============================================================================

@dataclass
class HillClimbingConfig:
    """Simulated annealing hill climbing parameters.
    
    The algorithm uses simulated annealing to explore the model space:
    - Accepts better models with probability 1
    - Accepts worse models with probability exp((new_score - old_score) / temperature)
    - Temperature decreases over time to reduce random exploration
    
    Attributes:
        max_iterations: Maximum number of training iterations
        plateau_iterations: Stop if no improvement for this many iterations
        base_temperature: Initial acceptance temperature (higher = more random)
        temperature_decay: Multiplicative decay per iteration (0 < decay < 1)
        adaptive_temp_increase: Multiplier when plateau detected (encourages exploration)
        adaptive_temp_threshold: Iterations without improvement to trigger increase
    """
    max_iterations: int = 1000
    plateau_iterations: int = 100
    base_temperature: float = 0.0005
    temperature_decay: float = 0.998
    adaptive_temp_increase: float = 1.2
    adaptive_temp_threshold: int = 25
    
    def validate(self):
        """Validate hill climbing configuration."""
        assert self.max_iterations > 0, "max_iterations must be positive"
        assert self.plateau_iterations > 0, "plateau_iterations must be positive"
        assert 0 < self.base_temperature < 1, "base_temperature must be in (0, 1)"
        assert 0 < self.temperature_decay < 1, "temperature_decay must be in (0, 1)"
        assert self.adaptive_temp_increase >= 1.0, "adaptive_temp_increase must be >= 1.0"
        assert self.adaptive_temp_threshold > 0, "adaptive_temp_threshold must be positive"


# ==============================================================================
# PARALLEL EXECUTION CONFIGURATION
# ==============================================================================

@dataclass
class ParallelConfig:
    """Batch parallel training configuration.
    
    Attributes:
        batch_size: Number of candidate models to train in parallel
        n_workers: Number of worker processes (usually equals batch_size)
        timeout_minutes: Maximum training time per model before forced termination
        pre_sample_data: Whether to sample data once per batch (vs. per model)
    """
    batch_size: int = 20
    n_workers: int = 20
    timeout_minutes: int = 60
    pre_sample_data: bool = True
    
    def validate(self):
        """Validate parallel configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.n_workers > 0, "n_workers must be positive"
        assert self.timeout_minutes > 0, "timeout_minutes must be positive"


# ==============================================================================
# DATA SAMPLING CONFIGURATION
# ==============================================================================

@dataclass
class SamplingConfig:
    """Row and column sampling configuration.
    
    Each model is trained on a random subsample of rows (stratified) and
    columns to increase diversity and reduce overfitting.
    
    Attributes:
        row_sample_range: (min, max) fraction of rows to sample (stratified)
        column_sample_range: (min, max) fraction of columns to sample
    """
    row_sample_range: Tuple[float, float] = (0.05, 0.15)
    column_sample_range: Tuple[float, float] = (0.30, 0.70)
    
    def validate(self):
        """Validate sampling configuration."""
        row_min, row_max = self.row_sample_range
        col_min, col_max = self.column_sample_range
        
        assert 0 < row_min <= row_max <= 1, "row_sample_range must be in (0, 1] with min <= max"
        assert 0 < col_min <= col_max <= 1, "column_sample_range must be in (0, 1] with min <= max"


# ==============================================================================
# STAGE 1 CONFIGURATION
# ==============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for a single classifier type.
    
    Attributes:
        classifier_class: The sklearn classifier class
        hyperparameters: Dict mapping parameter names to values or generator functions
        enabled: Whether this classifier is active in the pool
    """
    classifier_class: type
    hyperparameters: Dict[str, Any]
    enabled: bool = True


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering pipeline configuration.
    
    Controls what transformers are available and how they're selected.
    
    Attributes:
        skip_probability: Probability of skipping feature engineering entirely
        n_transformers_range: (min, max) number of transformers to chain
        available_transformers: List of transformer names to choose from
        transformer_hyperparams: Dict mapping transformer names to their hyperparameter generators
    """
    skip_probability: float = 0.30
    n_transformers_range: Tuple[int, int] = (1, 3)
    available_transformers: List[str] = field(default_factory=lambda: [
        'ratio', 'product', 'difference', 'sum', 'reciprocal',
        'square', 'sqrt', 'log', 'binning', 'kde', 'kmeans',
        'nystroem', 'rbf_sampler', 'power_transform', 'quantile_transform',
        'noise_injector'
    ])
    transformer_hyperparams: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default transformer hyperparameters if not provided."""
        if not self.transformer_hyperparams:
            self.transformer_hyperparams = get_default_transformer_hyperparams()
    
    def validate(self):
        """Validate feature engineering configuration."""
        assert 0 <= self.skip_probability <= 1, "skip_probability must be in [0, 1]"
        min_t, max_t = self.n_transformers_range
        assert 0 < min_t <= max_t, "n_transformers_range must have 0 < min <= max"
        assert len(self.available_transformers) > 0, "must have at least one available transformer"


@dataclass
class DimensionalityReductionConfig:
    """Dimensionality reduction configuration.
    
    Attributes:
        use_probability: Probability of applying dimensionality reduction
        available_methods: List of methods to choose from
        method_hyperparams: Dict mapping method names to their hyperparameter generators
    """
    use_probability: float = 0.5
    available_methods: List[str] = field(default_factory=lambda: [
        'pca', 'truncated_svd', 'fast_ica', 'factor_analysis'
    ])
    method_hyperparams: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default method hyperparameters if not provided."""
        if not self.method_hyperparams:
            self.method_hyperparams = get_default_dim_reduction_hyperparams()
    
    def validate(self):
        """Validate dimensionality reduction configuration."""
        assert 0 <= self.use_probability <= 1, "use_probability must be in [0, 1]"
        assert len(self.available_methods) > 0, "must have at least one available method"


@dataclass
class Stage1Config:
    """Complete Stage 1 (sklearn models) configuration.
    
    Attributes:
        classifiers: Dict mapping classifier names to their configs
        active_classifiers: List of classifier names to use
        sampling: Row and column sampling config
        feature_engineering: Feature engineering pipeline config
        dim_reduction: Dimensionality reduction config
        initial_scaler_options: Available scalers for initial preprocessing
    """
    classifiers: Dict[str, ClassifierConfig] = field(default_factory=dict)
    active_classifiers: List[str] = field(default_factory=lambda: [
        'logistic', 'qda', 'adaboost', 'lasso', 
        'random_forest', 'linear_svc', 'sgd_classifier',
        'extra_trees', 'naive_bayes', 'lda', 'ridge',
        'gradient_boosting', 'mlp', 'knn'
    ])
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    dim_reduction: DimensionalityReductionConfig = field(default_factory=DimensionalityReductionConfig)
    initial_scaler_options: List[str] = field(default_factory=lambda: ['standard', 'minmax', 'robust'])
    
    def __post_init__(self):
        """Initialize default classifier configs if not provided."""
        if not self.classifiers:
            self.classifiers = get_default_classifier_configs()
    
    def validate(self):
        """Validate Stage 1 configuration."""
        # Check active classifiers exist
        for name in self.active_classifiers:
            assert name in self.classifiers, f"Active classifier '{name}' not in classifier configs"
            assert self.classifiers[name].enabled, f"Classifier '{name}' is not enabled"
        
        # Validate sub-configs
        self.sampling.validate()
        self.feature_engineering.validate()
        self.dim_reduction.validate()
        
        assert len(self.initial_scaler_options) > 0, "must have at least one scaler option"


# ==============================================================================
# STAGE 2 CONFIGURATION
# ==============================================================================

@dataclass
class DNNArchitectureConfig:
    """Stage 2 DNN architecture specification.
    
    Attributes:
        hidden_layers: List of dicts specifying each hidden layer
            Each dict has: units (int), activation (str), dropout (float)
        output_units: Number of output units (1 for binary classification)
        output_activation: Output activation function ('sigmoid' for binary)
    """
    hidden_layers: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'units': 64, 'activation': 'relu', 'dropout': 0.3},
        {'units': 32, 'activation': 'relu', 'dropout': 0.2},
        {'units': 16, 'activation': 'relu', 'dropout': 0.1}
    ])
    output_units: int = 1
    output_activation: str = 'sigmoid'
    
    def validate(self):
        """Validate DNN architecture configuration."""
        assert len(self.hidden_layers) > 0, "must have at least one hidden layer"
        for i, layer in enumerate(self.hidden_layers):
            assert 'units' in layer, f"layer {i} missing 'units'"
            assert 'activation' in layer, f"layer {i} missing 'activation'"
            assert layer['units'] > 0, f"layer {i} units must be positive"
        
        assert self.output_units > 0, "output_units must be positive"


@dataclass
class DNNTrainingConfig:
    """Stage 2 DNN training parameters.
    
    Attributes:
        optimizer: Optimizer name (e.g., 'Adam', 'SGD')
        learning_rate: Initial learning rate
        loss: Loss function ('binary_crossentropy' for binary classification)
        metrics: List of metrics to track
        epochs: Maximum training epochs
        batch_size: Batch size for training
        patience: Early stopping patience (epochs without improvement)
        monitor: Metric to monitor for early stopping
        mode: 'max' or 'min' for early stopping
        restore_best_weights: Whether to restore best weights after early stopping
        retrain_frequency: Retrain DNN every N accepted Stage 1 models
    """
    optimizer: str = 'Adam'
    learning_rate: float = 0.001
    loss: str = 'binary_crossentropy'
    metrics: List[str] = field(default_factory=lambda: ['AUC', 'accuracy'])
    epochs: int = 50
    batch_size: int = 128
    patience: int = 10
    monitor: str = 'val_auc'
    mode: str = 'max'
    restore_best_weights: bool = True
    retrain_frequency: int = 20
    
    def validate(self):
        """Validate DNN training configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.patience > 0, "patience must be positive"
        assert self.mode in ['max', 'min'], "mode must be 'max' or 'min'"
        assert self.retrain_frequency > 0, "retrain_frequency must be positive"


@dataclass
class OptimizationConfig:
    """Stage 2 DNN hyperparameter optimization configuration.
    
    Uses Keras Tuner for periodic hyperparameter optimization.
    
    Attributes:
        enabled: Whether optimization is enabled
        trials_per_optimization: Number of trials for each optimization run
        optimize_every_n_batches: Optimize every N batches (None = adaptive)
        max_epochs: Maximum epochs per trial
        tuner_directory: Base directory for Keras Tuner files
    """
    enabled: bool = True
    trials_per_optimization: int = 10
    optimize_every_n_batches: Optional[int] = 4
    max_epochs: int = 100
    tuner_directory: str = 'keras_tuner'
    
    def validate(self):
        """Validate optimization configuration."""
        if self.enabled:
            assert self.trials_per_optimization > 0, "trials_per_optimization must be positive"
            if self.optimize_every_n_batches is not None:
                assert self.optimize_every_n_batches > 0, "optimize_every_n_batches must be positive"
            assert self.max_epochs > 0, "max_epochs must be positive"


@dataclass
class PseudoLabelingConfig:
    """Pseudo-labeling configuration.
    
    Uses Stage 2 DNN predictions on unlabeled test data to augment training.
    
    Attributes:
        enabled: Whether pseudo-labeling is enabled
        confidence_threshold: Minimum prediction confidence to use (0-1)
        max_fraction: Maximum fraction of training pool that can be pseudo-labeled
        target_class_ratio: Target positive class ratio (None = use training ratio)
        execute_after_optimization: Only pseudo-label after DNN optimization
    """
    enabled: bool = True
    confidence_threshold: float = 0.75
    max_fraction: float = 0.20
    target_class_ratio: Optional[float] = None
    execute_after_optimization: bool = True
    
    def validate(self):
        """Validate pseudo-labeling configuration."""
        if self.enabled:
            assert 0 < self.confidence_threshold <= 1, "confidence_threshold must be in (0, 1]"
            assert 0 < self.max_fraction <= 1, "max_fraction must be in (0, 1]"
            if self.target_class_ratio is not None:
                assert 0 < self.target_class_ratio < 1, "target_class_ratio must be in (0, 1)"


@dataclass
class Stage2Config:
    """Complete Stage 2 (DNN meta-learner) configuration.
    
    Attributes:
        architecture: DNN architecture specification
        training: Training parameters
        optimization: Hyperparameter optimization config
        pseudo_labeling: Pseudo-labeling config
    """
    architecture: DNNArchitectureConfig = field(default_factory=DNNArchitectureConfig)
    training: DNNTrainingConfig = field(default_factory=DNNTrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    pseudo_labeling: PseudoLabelingConfig = field(default_factory=PseudoLabelingConfig)
    
    def validate(self):
        """Validate Stage 2 configuration."""
        self.architecture.validate()
        self.training.validate()
        self.optimization.validate()
        self.pseudo_labeling.validate()


# ==============================================================================
# TRACKING CONFIGURATION
# ==============================================================================

@dataclass
class TrackingConfig:
    """Database and logging configuration.
    
    Attributes:
        db_path: Path to SQLite database file
        enable_wal: Whether to use WAL mode (better concurrency)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_to_file: Whether to log to file in addition to stdout
        log_directory: Directory for log files
    """
    db_path: str = 'ensemble_tracking.db'
    enable_wal: bool = True
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_directory: str = 'logs'
    
    def validate(self):
        """Validate tracking configuration."""
        assert self.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'], \
            "log_level must be DEBUG, INFO, WARNING, or ERROR"


# ==============================================================================
# PATHS CONFIGURATION
# ==============================================================================

@dataclass
class PathsConfig:
    """File paths configuration.
    
    Attributes:
        data_dir: Directory containing training/test data
        models_dir: Base directory for model files
        checkpoint_path: Path to save/load ensemble checkpoint
    """
    data_dir: Path = Path('../data')
    models_dir: Path = Path('../models')
    checkpoint_path: Optional[Path] = None
    
    def __post_init__(self):
        """Convert strings to Path objects."""
        self.data_dir = Path(self.data_dir)
        self.models_dir = Path(self.models_dir)
        if self.checkpoint_path:
            self.checkpoint_path = Path(self.checkpoint_path)
    
    def validate(self):
        """Validate paths configuration."""
        # No validation needed - directories will be created as needed
        pass


# ==============================================================================
# ROOT CONFIGURATION
# ==============================================================================

@dataclass
class EnsembleConfig:
    """Complete ensemble system configuration.
    
    This is the root configuration object that consolidates all configuration
    for the entire ensemble system. Create an instance and call validate()
    before use.
    
    Attributes:
        random_state: Random seed for reproducibility
        label: Target variable name
        hill_climbing: Hill climbing algorithm configuration
        parallel: Parallel execution configuration
        stage1: Stage 1 (sklearn models) configuration
        stage2: Stage 2 (DNN meta-learner) configuration
        tracking: Database and logging configuration
        paths: File paths configuration
    
    Example:
        >>> config = EnsembleConfig()
        >>> config.validate()
        >>> print(f"Using {len(config.stage1.active_classifiers)} classifiers")
    """
    random_state: int = 315
    label: str = 'diagnosed_diabetes'
    hill_climbing: HillClimbingConfig = field(default_factory=HillClimbingConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    def validate(self):
        """Validate entire configuration hierarchy.
        
        Raises:
            AssertionError: If any configuration parameter is invalid
        """
        self.hill_climbing.validate()
        self.parallel.validate()
        self.stage1.validate()
        self.stage2.validate()
        self.tracking.validate()
        self.paths.validate()
    
    def summary(self) -> str:
        """Generate a human-readable configuration summary.
        
        Returns:
            Multi-line string describing key configuration parameters
        """
        lines = [
            "Ensemble Configuration Summary",
            "=" * 50,
            f"Random State: {self.random_state}",
            "",
            "Hill Climbing:",
            f"  Max iterations: {self.hill_climbing.max_iterations}",
            f"  Base temperature: {self.hill_climbing.base_temperature}",
            f"  Temperature decay: {self.hill_climbing.temperature_decay}",
            "",
            "Parallel Execution:",
            f"  Batch size: {self.parallel.batch_size}",
            f"  Workers: {self.parallel.n_workers}",
            f"  Timeout: {self.parallel.timeout_minutes} minutes",
            "",
            "Stage 1:",
            f"  Active classifiers: {len(self.stage1.active_classifiers)}",
            f"  Row sampling: {self.stage1.sampling.row_sample_range}",
            f"  Column sampling: {self.stage1.sampling.column_sample_range}",
            "",
            "Stage 2:",
            f"  Retrain frequency: {self.stage2.training.retrain_frequency} models",
            f"  Optimization enabled: {self.stage2.optimization.enabled}",
            f"  Pseudo-labeling enabled: {self.stage2.pseudo_labeling.enabled}",
            ""
        ]
        return "\n".join(lines)


# ==============================================================================
# DEFAULT HYPERPARAMETER GENERATORS
# ==============================================================================

def get_default_classifier_configs() -> Dict[str, ClassifierConfig]:
    """Get default classifier configurations matching original system.
    
    Returns:
        Dict mapping classifier names to ClassifierConfig objects
    """
    return {
        'logistic': ClassifierConfig(
            classifier_class=LogisticRegression,
            hyperparameters={
                'solver': lambda rng: rng.choice(['lbfgs', 'newton-cg', 'sag']),
                'max_iter': 1000,
                'class_weight': 'balanced',
                'tol': 1e-3
            }
        ),
        'lasso': ClassifierConfig(
            classifier_class=LogisticRegression,
            hyperparameters={
                'C': lambda rng: 10 ** rng.uniform(-1, 1),
                'penalty': 'l1',
                'solver': lambda rng: rng.choice(['liblinear', 'saga']),
                'max_iter': lambda rng: rng.choice([100, 200, 300]),
                'class_weight': 'balanced',
                'tol': 1e-3
            }
        ),
        'random_forest': ClassifierConfig(
            classifier_class=RandomForestClassifier,
            hyperparameters={
                'n_estimators': lambda rng: int(10 ** rng.uniform(1.0, 2.0)),
                'max_depth': lambda rng: rng.randint(2, 20),
                'min_samples_split': lambda rng: int(10 ** rng.uniform(0.3, 1.3)),
                'min_samples_leaf': lambda rng: int(10 ** rng.uniform(0, 1)),
                'max_features': lambda rng: rng.choice(['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1
            }
        ),
        'linear_svc': ClassifierConfig(
            classifier_class=LinearSVC,
            hyperparameters={
                'C': lambda rng: 10 ** rng.uniform(-1, 1),
                'loss': 'squared_hinge',
                'max_iter': 50000,
                'class_weight': 'balanced',
                'dual': True,
                'tol': 1e-3
            }
        ),
        'sgd_classifier': ClassifierConfig(
            classifier_class=SGDClassifier,
            hyperparameters={
                'loss': lambda rng: rng.choice(['log_loss', 'modified_huber']),
                'penalty': lambda rng: rng.choice(['l2', 'l1', 'elasticnet']),
                'alpha': lambda rng: 10 ** rng.uniform(-5, -1),
                'learning_rate': lambda rng: rng.choice(['optimal', 'adaptive', 'constant']),
                'eta0': lambda rng: 10 ** rng.uniform(-4, -1),
                'max_iter': lambda rng: rng.choice([300, 500, 800]),
                'early_stopping': True,
                'class_weight': 'balanced'
            }
        ),
        'extra_trees': ClassifierConfig(
            classifier_class=ExtraTreesClassifier,
            hyperparameters={
                'n_estimators': lambda rng: rng.randint(2, 20),
                'max_depth': lambda rng: rng.randint(2, 10),
                'min_samples_split': lambda rng: int(10 ** rng.uniform(0.7, 1.5)),
                'min_samples_leaf': lambda rng: int(10 ** rng.uniform(0.5, 1.3)),
                'max_features': lambda rng: rng.choice(['sqrt', 'log2']),
                'class_weight': 'balanced',
                'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1,
                'min_impurity_decrease': 1e-7
            }
        ),
        'adaboost': ClassifierConfig(
            classifier_class=AdaBoostClassifier,
            hyperparameters={
                'n_estimators': lambda rng: rng.randint(2, 50),
                'learning_rate': lambda rng: 10 ** rng.uniform(-1.0, 0.5),
                'algorithm': 'SAMME'
            }
        ),
        'naive_bayes': ClassifierConfig(
            classifier_class=GaussianNB,
            hyperparameters={}
        ),
        'lda': ClassifierConfig(
            classifier_class=LinearDiscriminantAnalysis,
            hyperparameters={'solver': 'svd'}
        ),
        'qda': ClassifierConfig(
            classifier_class=QuadraticDiscriminantAnalysis,
            hyperparameters={'reg_param': lambda rng: 10 ** rng.uniform(-4, 0)}
        ),
        'ridge': ClassifierConfig(
            classifier_class=RidgeClassifier,
            hyperparameters={
                'alpha': lambda rng: 10 ** rng.uniform(-2, 2),
                'solver': lambda rng: rng.choice(['auto', 'cholesky', 'lsqr']),
                'class_weight': 'balanced',
                'tol': 1e-3
            }
        ),
        'gradient_boosting': ClassifierConfig(
            classifier_class=HistGradientBoostingClassifier,
            hyperparameters={
                'max_iter': lambda rng: rng.randint(2, 10),
                'learning_rate': lambda rng: 10 ** rng.uniform(-2.0, 0),
                'max_depth': lambda rng: rng.randint(2, 10),
                'l2_regularization': lambda rng: 10 ** rng.uniform(-4, 1),
                'min_samples_leaf': lambda rng: int(10 ** rng.uniform(1, 2)),
                'max_bins': lambda rng: rng.choice([32, 64, 128, 255])
            }
        ),
        'mlp': ClassifierConfig(
            classifier_class=MLPClassifier,
            hyperparameters={
                'n_layers': lambda rng: rng.randint(1, 4),
                'layer_sizes': lambda rng, n_layers: tuple([
                    int(10 ** rng.uniform(1.3, 2.3)) for _ in range(n_layers)
                ]),
                'alpha': lambda rng: 10 ** rng.uniform(-5, -1),
                'learning_rate_init': lambda rng: 10 ** rng.uniform(-4, -2),
                'activation': lambda rng: rng.choice(['relu', 'tanh', 'logistic']),
                'max_iter': lambda rng: rng.choice([100, 150, 200]),
                'early_stopping': True
            }
        ),
        'knn': ClassifierConfig(
            classifier_class=KNeighborsClassifier,
            hyperparameters={
                'n_neighbors': lambda rng: rng.randint(3, 10),
                'weights': lambda rng: rng.choice(['uniform', 'distance']),
                'p': lambda rng: rng.choice([1, 2]),
                'leaf_size': lambda rng: int(10 ** rng.uniform(1, 2)),
                'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1
            }
        )
    }


def get_default_transformer_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get default transformer hyperparameter generators.
    
    Returns:
        Dict mapping transformer names to their hyperparameter dicts
    """
    return {
        'ratio': {
            'n_features': lambda rng: rng.randint(5, 31)
        },
        'product': {
            'n_features': lambda rng: rng.randint(5, 31)
        },
        'difference': {
            'n_features': lambda rng: rng.randint(5, 31)
        },
        'sum': {
            'n_features': lambda rng: rng.randint(5, 31)
        },
        'reciprocal': {},
        'square': {},
        'sqrt': {},
        'log': {},
        'binning': {
            'n_bins': lambda rng: rng.randint(3, 8),
            'strategy': lambda rng: rng.choice(['quantile', 'uniform']),
            'encode': 'ordinal'
        },
        'kde': {
            'bandwidth': lambda rng: rng.choice(['scott', 'silverman'])
        },
        'kmeans': {
            'n_clusters': lambda rng: rng.randint(3, 11),
            'add_distances': lambda rng: rng.choice([True, False])
        },
        'nystroem': {
            'kernel': lambda rng: rng.choice(['rbf', 'poly', 'sigmoid', 'cosine']),
            'n_components': lambda rng, n_features: rng.randint(1, n_features),
            'gamma': lambda rng, kernel: 10 ** rng.uniform(-3, 0) if kernel in ['rbf', 'poly', 'sigmoid'] else None,
            'degree': lambda rng, kernel: rng.randint(2, 5) if kernel == 'poly' else 3
        },
        'rbf_sampler': {
            'n_components': lambda rng, n_features: rng.randint(1, n_features),
            'gamma': lambda rng: 10 ** rng.uniform(-3, 0)
        },
        'power_transform': {
            'method': 'yeo-johnson',
            'standardize': lambda rng: rng.choice([True, False])
        },
        'quantile_transform': {
            'n_quantiles': lambda rng: rng.choice([100, 500, 1000]),
            'output_distribution': lambda rng: rng.choice(['uniform', 'normal'])
        },
        'noise_injector': {
            'feature_fraction': lambda rng: rng.uniform(0.0, 1.0),
            'noise_scale_range': lambda rng: (rng.uniform(0.001, 0.05), rng.uniform(0.05, 0.3))
        }
    }


def get_default_dim_reduction_hyperparams() -> Dict[str, Dict[str, Any]]:
    """Get default dimensionality reduction hyperparameter generators.
    
    Returns:
        Dict mapping method names to their hyperparameter dicts
    """
    return {
        'pca': {
            'n_components': lambda rng: rng.choice([0.80, 0.85, 0.90, 0.95, 0.99]),
            'svd_solver': 'full',
            'whiten': False
        },
        'truncated_svd': {
            'n_components': lambda rng, n_features: rng.randint(1, max(2, min(n_features, 15))),
            'algorithm': 'randomized',
            'n_iter': 5
        },
        'fast_ica': {
            'n_components': lambda rng, n_features: rng.randint(1, max(2, min(n_features, 15))),
            'whiten': lambda rng: (False if rng.random() < 0.33 else rng.choice(['unit-variance', 'arbitrary-variance'])),
            'max_iter': lambda rng: rng.randint(200, 501),
            'algorithm': lambda rng: rng.choice(['parallel', 'deflation'])
        },
        'factor_analysis': {
            'n_components': lambda rng, n_features: rng.randint(1, max(2, min(n_features, 15))),
            'max_iter': 1000,
            'tol': 0.01
        }
    }
