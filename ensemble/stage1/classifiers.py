"""Classifier pool with hyperparameter sampling for Stage 1.

Provides a clean interface for accessing the 14 active classifiers and
sampling their hyperparameters randomly to create diverse models.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator

from ensemble.config import Stage1Config, ClassifierConfig


class ClassifierPool:
    """Manages the pool of available classifiers with hyperparameter sampling.
    
    The pool provides access to 14 different classifier types with random
    hyperparameter sampling to ensure model diversity.
    
    Attributes:
        config: Stage1Config with classifier configurations
        random_state: Random seed for reproducibility
        rng: NumPy random number generator
    
    Example:
        >>> from ensemble.config import EnsembleConfig
        >>> config = EnsembleConfig()
        >>> pool = ClassifierPool(config.stage1, random_state=42)
        >>> 
        >>> # Get a random classifier
        >>> classifier, name, params = pool.sample_classifier()
        >>> print(f"Sampled {name} with {len(params)} hyperparameters")
        >>> 
        >>> # Get specific classifier
        >>> clf_config = pool.get_config('random_forest')
        >>> classifier = pool.build_classifier('random_forest')
    """
    
    def __init__(self, config: Stage1Config, random_state: int):
        """Initialize the classifier pool.
        
        Args:
            config: Stage1Config with classifier configurations
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def get_active_classifiers(self) -> list:
        """Get list of active classifier names.
        
        Returns:
            List of classifier names that are enabled
        """
        return [
            name for name in self.config.active_classifiers
            if self.config.classifiers[name].enabled
        ]
    
    def get_config(self, classifier_name: str) -> ClassifierConfig:
        """Get configuration for a specific classifier.
        
        Args:
            classifier_name: Name of the classifier
            
        Returns:
            ClassifierConfig for the specified classifier
            
        Raises:
            KeyError: If classifier name not found
        """
        if classifier_name not in self.config.classifiers:
            raise KeyError(f"Classifier '{classifier_name}' not found in config")
        
        return self.config.classifiers[classifier_name]
    
    def sample_hyperparameters(
        self, 
        classifier_name: str,
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Sample hyperparameters for a classifier.
        
        Executes any callable hyperparameter generators (lambdas) with the
        random number generator to produce random values.
        
        Args:
            classifier_name: Name of the classifier
            n_jobs: Number of jobs for parallel classifiers (optional)
            
        Returns:
            Dict of hyperparameter names to sampled values
            
        Example:
            >>> params = pool.sample_hyperparameters('random_forest', n_jobs=4)
            >>> print(params['n_estimators'])  # Random value like 45
        """
        clf_config = self.get_config(classifier_name)
        sampled_params = {}
        
        # First pass: sample parameters that don't depend on others
        for param_name, param_value in clf_config.hyperparameters.items():
            if not callable(param_value):
                # Static value
                sampled_params[param_name] = param_value
            else:
                # Try to sample, handling different signatures
                import inspect
                sig = inspect.signature(param_value)
                param_names = list(sig.parameters.keys())
                
                try:
                    if len(param_names) == 1:
                        # Standard case: just rng
                        sampled_params[param_name] = param_value(self.rng)
                    elif len(param_names) == 2:
                        # Two arguments - determine what second arg should be
                        second_arg_name = param_names[1]
                        
                        if second_arg_name == 'n_jobs':
                            # Needs n_jobs
                            sampled_params[param_name] = param_value(self.rng, n_jobs or 1)
                        elif second_arg_name == 'n_layers':
                            # MLP layer_sizes needs n_layers
                            n_layers = sampled_params.get('n_layers', 2)
                            sampled_params[param_name] = param_value(self.rng, n_layers)
                        elif second_arg_name == 'n_features':
                            # Skip for now - will handle in second pass
                            continue
                        elif second_arg_name == 'kernel':
                            # Skip for now - will handle in second pass
                            continue
                        elif second_arg_name == 'solver':
                            # Skip for now - will handle in second pass
                            continue
                        else:
                            # Try with None as default
                            sampled_params[param_name] = param_value(self.rng, None)
                    else:
                        # More than 2 params - just try with rng
                        sampled_params[param_name] = param_value(self.rng)
                except Exception:
                    # If sampling fails, skip this parameter for now
                    continue
        
        # Second pass: handle parameters that depend on others
        for param_name, param_value in clf_config.hyperparameters.items():
            if param_name in sampled_params:
                continue  # Already sampled
            
            if callable(param_value):
                import inspect
                sig = inspect.signature(param_value)
                param_names = list(sig.parameters.keys())
                
                try:
                    if len(param_names) == 2:
                        second_arg_name = param_names[1]
                        
                        if second_arg_name == 'kernel':
                            # Nystroem gamma depends on kernel
                            kernel = sampled_params.get('kernel', 'rbf')
                            result = param_value(self.rng, kernel)
                            if result is not None:  # gamma is None for some kernels
                                sampled_params[param_name] = result
                        elif second_arg_name == 'solver':
                            # LDA shrinkage depends on solver
                            solver = sampled_params.get('solver', 'svd')
                            result = param_value(self.rng, solver)
                            if result is not None:
                                sampled_params[param_name] = result
                        else:
                            # Try anyway
                            sampled_params[param_name] = param_value(self.rng, None)
                except Exception:
                    # Skip if still fails
                    pass
        
        # Handle special cases for specific classifiers
        if classifier_name == 'mlp':
            # MLP has n_layers which determines layer_sizes
            if 'n_layers' in sampled_params and 'layer_sizes' in sampled_params:
                n_layers = sampled_params.pop('n_layers')
                # layer_sizes was already computed using n_layers
                sampled_params['hidden_layer_sizes'] = sampled_params.pop('layer_sizes')
        
        return sampled_params
    
    def build_classifier(
        self, 
        classifier_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None
    ) -> BaseEstimator:
        """Build a classifier instance with given or sampled hyperparameters.
        
        Args:
            classifier_name: Name of the classifier
            hyperparameters: Dict of hyperparameters (if None, will sample)
            n_jobs: Number of jobs for parallel classifiers
            
        Returns:
            Instantiated sklearn classifier
            
        Example:
            >>> # Build with sampled hyperparameters
            >>> clf = pool.build_classifier('random_forest', n_jobs=4)
            >>> 
            >>> # Build with specific hyperparameters
            >>> clf = pool.build_classifier('logistic', {'C': 1.0, 'max_iter': 1000})
        """
        clf_config = self.get_config(classifier_name)
        
        # Sample hyperparameters if not provided
        if hyperparameters is None:
            hyperparameters = self.sample_hyperparameters(classifier_name, n_jobs)
        
        # Instantiate classifier
        classifier = clf_config.classifier_class(**hyperparameters)
        
        return classifier
    
    def sample_classifier(
        self, 
        n_jobs: Optional[int] = None
    ) -> tuple:
        """Sample a random classifier from the active pool.
        
        Args:
            n_jobs: Number of jobs for parallel classifiers
            
        Returns:
            Tuple of (classifier, name, hyperparameters)
            
        Example:
            >>> clf, name, params = pool.sample_classifier(n_jobs=4)
            >>> print(f"Sampled: {name}")
            >>> clf.fit(X_train, y_train)
        """
        # Choose random classifier from active pool
        active = self.get_active_classifiers()
        classifier_name = self.rng.choice(active)
        
        # Sample hyperparameters
        hyperparameters = self.sample_hyperparameters(classifier_name, n_jobs)
        
        # Build classifier
        classifier = self.build_classifier(classifier_name, hyperparameters, n_jobs)
        
        return classifier, classifier_name, hyperparameters
    
    def get_pool_summary(self) -> str:
        """Generate human-readable summary of the classifier pool.
        
        Returns:
            Multi-line string describing the pool
        """
        active = self.get_active_classifiers()
        
        lines = [
            "Classifier Pool Summary",
            "=" * 50,
            f"Total classifiers: {len(self.config.classifiers)}",
            f"Active classifiers: {len(active)}",
            "",
            "Active classifiers:"
        ]
        
        for name in active:
            clf_config = self.config.classifiers[name]
            n_hyperparams = len(clf_config.hyperparameters)
            lines.append(f"  - {name}: {clf_config.classifier_class.__name__} ({n_hyperparams} hyperparams)")
        
        return "\n".join(lines)
    
    def validate_classifier(self, classifier_name: str) -> tuple:
        """Validate that a classifier can be built and has valid config.
        
        Args:
            classifier_name: Name of the classifier to validate
            
        Returns:
            Tuple of (is_valid: bool, message: str)
            
        Example:
            >>> valid, msg = pool.validate_classifier('random_forest')
            >>> if not valid:
            ...     print(f"Error: {msg}")
        """
        try:
            # Check if classifier exists
            if classifier_name not in self.config.classifiers:
                return False, f"Classifier '{classifier_name}' not found"
            
            # Check if enabled
            clf_config = self.config.classifiers[classifier_name]
            if not clf_config.enabled:
                return False, f"Classifier '{classifier_name}' is disabled"
            
            # Try to build with sampled hyperparameters
            classifier = self.build_classifier(classifier_name)
            
            # Check it's a valid sklearn estimator
            if not hasattr(classifier, 'fit'):
                return False, f"Classifier missing 'fit' method"
            
            if not hasattr(classifier, 'predict'):
                return False, f"Classifier missing 'predict' method"
            
            return True, f"Classifier '{classifier_name}' is valid"
            
        except Exception as e:
            return False, f"Error building classifier: {str(e)}"
    
    def validate_all(self) -> Dict[str, tuple]:
        """Validate all classifiers in the pool.
        
        Returns:
            Dict mapping classifier names to (is_valid, message) tuples
        """
        results = {}
        for name in self.config.classifiers.keys():
            results[name] = self.validate_classifier(name)
        return results
