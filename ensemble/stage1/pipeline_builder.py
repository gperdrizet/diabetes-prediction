"""Pipeline builder for Stage 1 ensemble models.

This module provides the PipelineBuilder class that generates random sklearn
pipelines with diverse feature engineering and classifier configurations.
Replaces the legacy generate_random_pipeline() function.
"""

import inspect
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, QuantileTransformer, RobustScaler

from ensemble.config import EnsembleConfig
# Import preprocessing transformers from transformers package
from ensemble.stage1.transformers import (
    CleanNumericTransformer,
    RandomFeatureSelector,
    ConstantFeatureRemover,
    IQRClipper
)
# Import feature engineering transformers from feature_transformers module
from ensemble.stage1.feature_transformers import (
    RatioTransformer,
    ProductTransformer,
    DifferenceTransformer,
    SumTransformer,
    ReciprocalTransformer,
    SquareTransformer,
    SquareRootTransformer,
    LogTransformer,
    BinningTransformer,
    KDESmoothingTransformer,
    KMeansClusterTransformer,
    NoiseInjector
)


class PipelineBuilder:
    """Builds random sklearn pipelines for Stage 1 ensemble training.
    
    This class generates diverse pipelines by:
    - Randomly selecting and configuring classifiers
    - Randomly applying feature engineering transformations
    - Randomly applying dimensionality reduction
    - Sampling hyperparameters from configured distributions
    
    Parameters
    ----------
    config : EnsembleConfig
        Configuration object containing all hyperparameter distributions
        and pipeline generation settings.
    
    Examples
    --------
    >>> from ensemble.config import EnsembleConfig
    >>> from ensemble.stage1 import PipelineBuilder
    >>> config = EnsembleConfig()
    >>> builder = PipelineBuilder(config)
    >>> pipeline, metadata = builder.generate(
    ...     iteration=0,
    ...     random_state=42,
    ...     base_preprocessor=preprocessor,
    ...     n_jobs=4
    ... )
    """
    
    def __init__(self, config: EnsembleConfig):
        """Initialize the pipeline builder with configuration.
        
        Parameters
        ----------
        config : EnsembleConfig
            Configuration containing all settings for pipeline generation.
        """
        self.config = config
        
        # Create transformer class mapping
        self.transformer_class_map = {
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
        
        # Create dimensionality reduction class mapping
        self.dim_reduction_class_map = {
            'pca': PCA,
            'truncated_svd': TruncatedSVD,
            'fast_ica': FastICA,
            'factor_analysis': FactorAnalysis
        }
    
    @staticmethod
    def _generate_hyperparameters(rng: np.random.RandomState, hyperparam_config: Dict, **context) -> Dict:
        """Generate hyperparameters from config lambda functions.
        
        This static method handles the complexity of calling lambda functions
        that may require different context parameters (e.g., rng, n_jobs, n_features).
        
        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator for sampling.
        hyperparam_config : dict
            Dictionary mapping parameter names to values or lambda functions.
        **context : dict
            Context variables that lambda functions may need (e.g., n_jobs, n_layers).
        
        Returns
        -------
        hyperparams : dict
            Dictionary of hyperparameter names to generated values.
        """
        hyperparams = {}
        
        for param_name, param_value in hyperparam_config.items():
            if callable(param_value):
                # Get the function signature to determine what arguments it needs
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
    
    def generate(
        self,
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
            Current iteration number (included in metadata).
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
            
        Raises
        ------
        ValueError
            If a selected transformer or classifier is not available.
        RuntimeError
            If pipeline generation fails for any reason.
        """
        try:
            rng = np.random.RandomState(random_state)
            
            # Get sampling configuration
            row_sample_pct = rng.uniform(
                *self.config.stage1.sampling.row_sample_range
            )
            col_sample_pct = rng.uniform(
                *self.config.stage1.sampling.column_sample_range
            )
            
            # Feature engineering configuration
            skip_feature_engineering = rng.random() < self.config.stage1.feature_engineering.skip_probability
            n_transformers = 0 if skip_feature_engineering else rng.randint(
                self.config.stage1.feature_engineering.n_transformers_range[0],
                self.config.stage1.feature_engineering.n_transformers_range[1] + 1
            )
            
            # Randomly select transformers
            available_transformers = self.config.stage1.feature_engineering.available_transformers
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
                if name not in self.transformer_class_map:
                    raise ValueError(
                        f"Transformer '{name}' not found in transformer_class_map. "
                        f"Available: {list(self.transformer_class_map.keys())}"
                    )
                
                TransformerClass = self.transformer_class_map[name]
                hyperparam_config = self.config.stage1.feature_engineering.transformer_hyperparams.get(name, {})
                hyperparams = self._generate_hyperparameters(
                    rng, hyperparam_config, n_features=n_features_after_sampling
                )
                
                # Create transformer with generated hyperparameters
                transformer = TransformerClass(**hyperparams)
                feature_steps.append((name, transformer))
            
            # Add initial scaling BEFORE feature engineering
            scaler_choice = rng.choice(self.config.stage1.initial_scaler_options)
            if scaler_choice == 'standard':
                initial_scaler = StandardScaler()
            elif scaler_choice == 'minmax':
                initial_scaler = MinMaxScaler()
            else:  # robust
                initial_scaler = RobustScaler()
            
            # Add NaN/Inf cleaner AFTER feature engineering
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
            
            # Optionally add dimensionality reduction
            use_dim_reduction = rng.random() < self.config.stage1.dim_reduction.use_probability
            dim_reduction_name = None
            if use_dim_reduction:
                available_methods = self.config.stage1.dim_reduction.available_methods
                dim_reduction_name = rng.choice(available_methods)
                
                if dim_reduction_name not in self.dim_reduction_class_map:
                    raise ValueError(
                        f"Dimensionality reduction method '{dim_reduction_name}' not found. "
                        f"Available: {list(self.dim_reduction_class_map.keys())}"
                    )
                
                DimReductionClass = self.dim_reduction_class_map[dim_reduction_name]
                hyperparam_config = self.config.stage1.dim_reduction.method_hyperparams[dim_reduction_name]
                hyperparams = self._generate_hyperparameters(
                    rng, hyperparam_config, n_features=n_features_after_sampling
                )
                
                dim_reducer = DimReductionClass(**hyperparams)
                feature_steps.append((dim_reduction_name, dim_reducer))
            
            # Select classifier
            classifier_type = rng.choice(self.config.stage1.active_classifiers)
            
            if classifier_type not in self.config.stage1.classifiers:
                raise ValueError(
                    f"Classifier '{classifier_type}' not found in config. "
                    f"Available: {list(self.config.stage1.classifiers.keys())}"
                )
            
            # Add final scaler before classifier
            feature_steps.append(('scaler', StandardScaler()))
            
            # Get classifier configuration
            classifier_config = self.config.stage1.classifiers[classifier_type]
            ClassifierClass = classifier_config.classifier_class
            hyperparam_config = classifier_config.hyperparameters
            
            # Generate hyperparameters with n_jobs context for parallel classifiers
            hyperparams = self._generate_hyperparameters(rng, hyperparam_config, n_jobs=n_jobs)
            
            # Handle special MLP case where layer_sizes needs to be derived from n_layers
            if classifier_type == 'mlp' and 'hidden_layer_sizes' not in hyperparams:
                if 'layer_sizes' in hyperparams:
                    hyperparams['hidden_layer_sizes'] = hyperparams.pop('layer_sizes')
                hyperparams.pop('n_layers', None)
            
            # Create classifier instance
            classifier = ClassifierClass(**hyperparams)
            
            # Wrap classifiers without predict_proba in CalibratedClassifierCV
            if classifier_type in ['linear_svc', 'ridge']:
                classifier = CalibratedClassifierCV(classifier, cv=3, method='sigmoid')
            
            # Build complete pipeline
            pipeline = Pipeline([
                ('preprocessor', clone(base_preprocessor)),
                ('feature_engineering', Pipeline(feature_steps)),
                ('classifier', classifier)
            ])
            
            # Create metadata (matches legacy format for database compatibility)
            metadata = {
                'row_sample_pct': row_sample_pct,
                'col_sample_pct': col_sample_pct,
                'transformers_used': transformer_names,
                'dim_reduction': dim_reduction_name,
                'classifier_type': classifier_type,
                'iteration': iteration
            }
            
            return pipeline, metadata
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate pipeline for iteration {iteration} "
                f"with random_state {random_state}: {str(e)}"
            ) from e
