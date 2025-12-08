"""Configuration file for ensemble model hyperparameters.

This file centralizes all classifier definitions and their hyperparameter ranges,
making it easy to view, modify, and experiment with different configurations.

Status Legend:
- ACTIVE: Currently used in ensemble training
- DISABLED: Temporarily disabled due to performance issues
"""

from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


# ==============================================================================
# ACTIVE CLASSIFIERS (13 total)
# ==============================================================================

ACTIVE_CLASSIFIERS = [
    'logistic',
    'random_forest',
    'linear_svc',
    'sgd_classifier',
    'extra_trees',
    'adaboost',
    'naive_bayes',
    'lda',
    'qda',
    'ridge',
    'gradient_boosting',
    'mlp',
    'knn'
]


# ==============================================================================
# DISABLED CLASSIFIERS (0 total)
# ==============================================================================

DISABLED_CLASSIFIERS = []


# ==============================================================================
# CLASSIFIER CONFIGURATIONS
# ==============================================================================
# Each configuration specifies:
# - class: The sklearn classifier class
# - status: 'active' or 'disabled'
# - hyperparameters: Dict of parameter names to generation functions
# - notes: Performance notes and optimization history

CLASSIFIER_CONFIGS = {
    'logistic': {
        'class': LogisticRegression,
        'status': 'active',
        'notes': 'Fast L2-regularized linear model. Optimized for speed: reduced max_iter, narrowed C range, removed L1 penalty.',
        'hyperparameters': {
            'C': lambda rng: 10 ** rng.uniform(-1, 1),  # 0.1 to 10 (narrowed from -2 to 2)
            'penalty': lambda rng: rng.choice(['l2', None]),  # Removed 'l1' (slower)
            'solver': lambda rng, penalty: (
                rng.choice(['lbfgs', 'newton-cg']) if penalty == 'l2' 
                else rng.choice(['lbfgs', 'newton-cg'])
            ),
            'max_iter': lambda rng: rng.choice([100, 200, 300]),  # Reduced from 500-1000
            'class_weight': lambda rng: 'balanced',
            'tol': lambda rng: 1e-3  # Relaxed from 1e-4 for faster convergence
        }
    },
    
    'random_forest': {
        'class': RandomForestClassifier,
        'status': 'active',
        'notes': 'Ensemble of decision trees. Parallelizable with n_jobs. Good balance of speed and performance.',
        'hyperparameters': {
            'n_estimators': lambda rng: int(10 ** rng.uniform(1.0, 2.0)),  # ~10 to 100
            'max_depth': lambda rng: rng.choice([3, 5, 7, 10, 15, 20, None]),
            'min_samples_split': lambda rng: int(10 ** rng.uniform(0.3, 1.3)),  # 2 to 20
            'min_samples_leaf': lambda rng: int(10 ** rng.uniform(0, 1)),  # 1 to 10
            'max_features': lambda rng: rng.choice(['sqrt', 'log2', None]),
            'class_weight': lambda rng: 'balanced',
            'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1
        }
    },
    
    'linear_svc': {
        'class': LinearSVC,
        'status': 'active',
        'notes': 'Fast linear SVM. Optimized: reduced C range, max_iter, dual=False for many features.',
        'hyperparameters': {
            'C': lambda rng: 10 ** rng.uniform(-1, 1),  # 0.1 to 10 (narrowed)
            'loss': lambda rng: 'squared_hinge',  # Only squared_hinge (faster)
            'max_iter': lambda rng: rng.choice([200, 300]),  # Reduced from 500-1000
            'class_weight': lambda rng: 'balanced',
            'dual': lambda rng: False,  # Much faster with many features
            'tol': lambda rng: 1e-3  # Relaxed from 1e-4
        }
    },
    
    'sgd_classifier': {
        'class': SGDClassifier,
        'status': 'active',
        'notes': 'Stochastic gradient descent. Fast online learning, good for large datasets.',
        'hyperparameters': {
            'loss': lambda rng: rng.choice(['hinge', 'log_loss', 'modified_huber', 'perceptron']),
            'penalty': lambda rng: rng.choice(['l2', 'l1', 'elasticnet']),
            'alpha': lambda rng: 10 ** rng.uniform(-5, -1),  # 0.00001 to 0.1
            'learning_rate': lambda rng: rng.choice(['optimal', 'adaptive', 'constant']),
            'eta0': lambda rng: 10 ** rng.uniform(-4, -1),  # 0.0001 to 0.1
            'max_iter': lambda rng: rng.choice([300, 500, 800]),
            'early_stopping': lambda rng: True,
            'class_weight': lambda rng: 'balanced'
        }
    },
    
    'extra_trees': {
        'class': ExtraTreesClassifier,
        'status': 'active',
        'notes': 'Extremely randomized trees. Optimized: reduced n_estimators (5-30), max_depth ≤10, forced feature subsampling.',
        'hyperparameters': {
            'n_estimators': lambda rng: int(10 ** rng.uniform(0.7, 1.5)),  # ~5 to 30 (was 10-100)
            'max_depth': lambda rng: rng.choice([3, 5, 7, 10]),  # Max 10, removed None
            'min_samples_split': lambda rng: int(10 ** rng.uniform(0.7, 1.5)),  # 5 to 30
            'min_samples_leaf': lambda rng: int(10 ** rng.uniform(0.5, 1.3)),  # 3 to 20
            'max_features': lambda rng: rng.choice(['sqrt', 'log2']),  # Removed None
            'class_weight': lambda rng: 'balanced',
            'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1,
            'min_impurity_decrease': lambda rng: 1e-7  # Early stopping
        }
    },
    
    'adaboost': {
        'class': AdaBoostClassifier,
        'status': 'active',
        'notes': 'Adaptive boosting. Sequential but relatively fast with reasonable n_estimators.',
        'hyperparameters': {
            'n_estimators': lambda rng: int(10 ** rng.uniform(1.0, 2.0)),  # ~10 to 100
            'learning_rate': lambda rng: 10 ** rng.uniform(-1.0, 0.5),  # 0.1 to 3.0
            'algorithm': lambda rng: 'SAMME'  # Avoid deprecated SAMME.R
        }
    },
    
    'naive_bayes': {
        'class': None,  # Dynamic based on type
        'status': 'active',
        'notes': 'Probabilistic classifier. Three variants: Gaussian (continuous), Multinomial (counts), Bernoulli (binary).',
        'hyperparameters': {
            'type': lambda rng: rng.choice(['gaussian', 'multinomial', 'bernoulli']),
            # Gaussian params
            'var_smoothing': lambda rng: 10 ** rng.uniform(-12, -6),  # 1e-12 to 1e-6
            # Multinomial/Bernoulli params
            'alpha': lambda rng: 10 ** rng.uniform(-2, 1),  # 0.01 to 10
            'fit_prior': lambda rng: rng.choice([True, False]),
            # Bernoulli only
            'binarize': lambda rng: rng.choice([None, 0.0, rng.uniform(0.3, 0.7)])
        }
    },
    
    'lda': {
        'class': LinearDiscriminantAnalysis,
        'status': 'active',
        'notes': 'Linear Discriminant Analysis. Fast, assumes Gaussian distributions with shared covariance.',
        'hyperparameters': {
            'solver': lambda rng: rng.choice(['svd', 'lsqr', 'eigen']),
            'shrinkage': lambda rng, solver: (
                rng.choice([None, 'auto', rng.uniform(0.0, 1.0)]) 
                if solver in ['lsqr', 'eigen'] else None
            )
        }
    },
    
    'qda': {
        'class': QuadraticDiscriminantAnalysis,
        'status': 'active',
        'notes': 'Quadratic Discriminant Analysis. More flexible than LDA, models non-linear boundaries.',
        'hyperparameters': {
            'reg_param': lambda rng: 10 ** rng.uniform(-4, 0)  # 0.0001 to 1.0
        }
    },
    
    'ridge': {
        'class': RidgeClassifier,
        'status': 'active',
        'notes': 'Fast L2-regularized linear model. Very fast training, good baseline.',
        'hyperparameters': {
            'alpha': lambda rng: 10 ** rng.uniform(-2, 2),  # 0.01 to 100
            'solver': lambda rng: rng.choice(['auto', 'cholesky', 'lsqr']),
            'class_weight': lambda rng: 'balanced',
            'tol': lambda rng: 1e-3  # Relaxed tolerance
        }
    },
    
    # ====== DISABLED CLASSIFIERS ======
    
    'gradient_boosting': {
        'class': HistGradientBoostingClassifier,
        'status': 'active',
        'notes': 'Gradient boosting with histogram-based optimization. Sequential but efficient with limited iterations.',
        'hyperparameters': {
            'max_iter': lambda rng: int(10 ** rng.uniform(1.0, 1.7)),  # ~10 to 50
            'learning_rate': lambda rng: 10 ** rng.uniform(-2.0, 0),  # 0.01 to 1.0
            'max_depth': lambda rng: rng.choice([None, 3, 5, 7, 10]),
            'l2_regularization': lambda rng: 10 ** rng.uniform(-4, 1),  # 0.0001 to 10
            'min_samples_leaf': lambda rng: int(10 ** rng.uniform(1, 2)),  # 10 to 100
            'max_bins': lambda rng: rng.choice([32, 64, 128, 255])
        }
    },
    
    'mlp': {
        'class': MLPClassifier,
        'status': 'active',
        'notes': 'Multi-layer perceptron neural network. Training time controlled with early stopping and limited iterations.',
        'hyperparameters': {
            'n_layers': lambda rng: rng.randint(1, 4),  # 1 to 3 hidden layers
            'layer_sizes': lambda rng, n_layers: tuple([int(10 ** rng.uniform(1.3, 2.3)) for _ in range(n_layers)]),
            'alpha': lambda rng: 10 ** rng.uniform(-5, -1),  # 0.00001 to 0.1
            'learning_rate_init': lambda rng: 10 ** rng.uniform(-4, -2),  # 0.0001 to 0.01
            'activation': lambda rng: rng.choice(['relu', 'tanh', 'logistic']),
            'max_iter': lambda rng: rng.choice([100, 150, 200]),
            'early_stopping': lambda rng: True
        }
    },
    
    'knn': {
        'class': KNeighborsClassifier,
        'status': 'active',
        'notes': 'K-Nearest Neighbors. Distance calculations optimized with limited sample sizes and efficient leaf size.',
        'hyperparameters': {
            'n_neighbors': lambda rng: int(10 ** rng.uniform(0.5, 1.5)),  # 3 to 30
            'weights': lambda rng: rng.choice(['uniform', 'distance']),
            'p': lambda rng: rng.choice([1, 2]),  # Manhattan or Euclidean
            'leaf_size': lambda rng: int(10 ** rng.uniform(1, 2)),  # 10 to 100
            'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1
        }
    }
}


# ==============================================================================
# SAMPLING CONFIGURATION
# ==============================================================================

SAMPLING_CONFIG = {
    'row_sample_pct': {
        'min': 0.10,  # 10% minimum (was 1.25%)
        'max': 0.40,  # 40% maximum (was 15%)
        'notes': 'Higher sampling = stronger models, lower = more diversity. Range gives 2,400-9,600 rows from ~24,000 pool.'
    },
    'col_sample_pct': {
        'min': 0.30,  # 30% minimum (was 50%)
        'max': 0.70,  # 70% maximum (was 95%)
        'notes': 'Random feature selection for diversity. Reduced range increases diversity.'
    }
}


# ==============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ==============================================================================

FEATURE_ENGINEERING_CONFIG = {
    'skip_probability': 0.30,  # 30% chance to use raw features only
    'n_transformers': {
        'min': 1,
        'max': 3,  # 1-3 transformers if not skipping
        'notes': 'Number of feature engineering transformers to apply'
    },
    'available_transformers': [
        'ratio', 'product', 'difference', 'sum', 'reciprocal',
        'square', 'sqrt', 'log', 'binning', 'kde', 'kmeans',
        'nystroem', 'rbf_sampler', 'power_transform', 'quantile_transform',
        'standard_scaler', 'noise_injector'
    ]
}


# ==============================================================================
# DIMENSIONALITY REDUCTION CONFIGURATION
# ==============================================================================

DIM_REDUCTION_CONFIG = {
    'use_probability': 0.5,  # 50% chance to apply dimensionality reduction
    'available_methods': ['pca', 'truncated_svd', 'fast_ica', 'factor_analysis']
}


# ==============================================================================
# STAGE 2 DNN META-LEARNER CONFIGURATION
# ==============================================================================

STAGE2_DNN_CONFIG = {
    'description': 'Deep neural network meta-learner for Stage 2 ensemble aggregation',
    'notes': 'Learns optimal weights for combining Stage 1 model predictions. '
             'Trained on validation set 1, evaluated on validation set 2.',
    'architecture': {
        'input': 'Stage 1 model predictions (n_models,)',
        'hidden_layers': [
            {'units': 64, 'activation': 'relu', 'dropout': 0.3},
            {'units': 32, 'activation': 'relu', 'dropout': 0.2},
            {'units': 16, 'activation': 'relu', 'dropout': 0.1}
        ],
        'output': {'units': 1, 'activation': 'sigmoid'}
    },
    'training': {
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'loss': 'binary_crossentropy',
        'metrics': ['AUC', 'accuracy'],
        'epochs': 100,
        'batch_size': 128,
        'early_stopping': {
            'monitor': 'val_auc',
            'patience': 10,
            'mode': 'max',
            'restore_best_weights': True
        },
        'retrain_frequency': 'Every 10 accepted models'
    },
    'data_splits': {
        'training_pool': '60% of total data',
        'validation_s1': '20% (Stage 2 training)',
        'validation_s2': '20% (Stage 2 evaluation)'
    }
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_active_classifiers():
    """Return list of currently active classifier names."""
    return ACTIVE_CLASSIFIERS.copy()


def get_disabled_classifiers():
    """Return list of currently disabled classifier names."""
    return DISABLED_CLASSIFIERS.copy()


def get_classifier_config(classifier_name):
    """Get configuration for a specific classifier.
    
    Parameters
    ----------
    classifier_name : str
        Name of the classifier (e.g., 'logistic', 'random_forest')
    
    Returns
    -------
    config : dict
        Configuration dictionary with 'class', 'status', 'hyperparameters', 'notes'
    """
    if classifier_name not in CLASSIFIER_CONFIGS:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return CLASSIFIER_CONFIGS[classifier_name].copy()


def print_classifier_summary():
    """Print a summary of all classifiers and their status."""
    print("=" * 80)
    print("ENSEMBLE CLASSIFIER CONFIGURATION SUMMARY")
    print("=" * 80)
    
    print(f"\nACTIVE CLASSIFIERS ({len(ACTIVE_CLASSIFIERS)}):")
    for name in ACTIVE_CLASSIFIERS:
        config = CLASSIFIER_CONFIGS[name]
        print(f"\n  {name}:")
        print(f"    Class: {config['class'].__name__ if config['class'] else 'Dynamic'}")
        print(f"    Notes: {config['notes']}")
        print(f"    Hyperparameter Space:")
        
        # Display hyperparameter ranges based on the config
        hyperparams = config['hyperparameters']
        
        if name == 'logistic':
            print(f"      • C: 0.1 to 10 (log uniform)")
            print(f"      • penalty: l2, None")
            print(f"      • solver: lbfgs, newton-cg")
            print(f"      • max_iter: 100, 200, 300")
            print(f"      • class_weight: balanced")
            print(f"      • tol: 1e-3")
        elif name == 'random_forest':
            print(f"      • n_estimators: ~10 to 100 (log uniform)")
            print(f"      • max_depth: 3, 5, 7, 10, 15, 20, None")
            print(f"      • min_samples_split: 2 to 20 (log uniform)")
            print(f"      • min_samples_leaf: 1 to 10 (log uniform)")
            print(f"      • max_features: sqrt, log2, None")
            print(f"      • class_weight: balanced")
        elif name == 'linear_svc':
            print(f"      • C: 0.1 to 10 (log uniform)")
            print(f"      • loss: squared_hinge")
            print(f"      • max_iter: 200, 300")
            print(f"      • class_weight: balanced")
            print(f"      • dual: False")
            print(f"      • tol: 1e-3")
        elif name == 'sgd_classifier':
            print(f"      • loss: hinge, log_loss, modified_huber, perceptron")
            print(f"      • penalty: l2, l1, elasticnet")
            print(f"      • alpha: 0.00001 to 0.1 (log uniform)")
            print(f"      • learning_rate: optimal, adaptive, constant")
            print(f"      • eta0: 0.0001 to 0.1 (log uniform)")
            print(f"      • max_iter: 300, 500, 800")
            print(f"      • early_stopping: True")
            print(f"      • class_weight: balanced")
        elif name == 'extra_trees':
            print(f"      • n_estimators: ~5 to 30 (log uniform)")
            print(f"      • max_depth: 3, 5, 7, 10")
            print(f"      • min_samples_split: 5 to 30 (log uniform)")
            print(f"      • min_samples_leaf: 3 to 20 (log uniform)")
            print(f"      • max_features: sqrt, log2")
            print(f"      • class_weight: balanced")
            print(f"      • min_impurity_decrease: 1e-7")
        elif name == 'adaboost':
            print(f"      • n_estimators: ~10 to 100 (log uniform)")
            print(f"      • learning_rate: 0.1 to 3.0 (log uniform)")
            print(f"      • algorithm: SAMME")
        elif name == 'naive_bayes':
            print(f"      • type: gaussian, multinomial, bernoulli")
            print(f"      • var_smoothing: 1e-12 to 1e-6 (Gaussian only)")
            print(f"      • alpha: 0.01 to 10 (Multinomial/Bernoulli)")
            print(f"      • fit_prior: True, False")
            print(f"      • binarize: None, 0.0, 0.3-0.7 (Bernoulli only)")
        elif name == 'lda':
            print(f"      • solver: svd, lsqr, eigen")
            print(f"      • shrinkage: None, auto, 0.0-1.0 (lsqr/eigen only)")
        elif name == 'qda':
            print(f"      • reg_param: 0.0001 to 1.0 (log uniform)")
        elif name == 'ridge':
            print(f"      • alpha: 0.01 to 100 (log uniform)")
            print(f"      • solver: auto, cholesky, lsqr")
            print(f"      • class_weight: balanced")
            print(f"      • tol: 1e-3")
        elif name == 'gradient_boosting':
            print(f"      • max_iter: ~10 to 50 (log uniform)")
            print(f"      • learning_rate: 0.01 to 1.0 (log uniform)")
            print(f"      • max_depth: None, 3, 5, 7, 10")
            print(f"      • l2_regularization: 0.0001 to 10 (log uniform)")
            print(f"      • min_samples_leaf: 10 to 100 (log uniform)")
            print(f"      • max_bins: 32, 64, 128, 255")
        elif name == 'mlp':
            print(f"      • n_layers: 1 to 3 hidden layers")
            print(f"      • layer_sizes: 20-200 neurons per layer (log uniform)")
            print(f"      • alpha: 0.00001 to 0.1 (log uniform)")
            print(f"      • learning_rate_init: 0.0001 to 0.01 (log uniform)")
            print(f"      • activation: relu, tanh, logistic")
            print(f"      • max_iter: 100, 150, 200")
            print(f"      • early_stopping: True")
        elif name == 'knn':
            print(f"      • n_neighbors: 3 to 30 (log uniform)")
            print(f"      • weights: uniform, distance")
            print(f"      • p: 1 (Manhattan), 2 (Euclidean)")
            print(f"      • leaf_size: 10 to 100 (log uniform)")
    
    print(f"\n\nDISABLED CLASSIFIERS ({len(DISABLED_CLASSIFIERS)}):")
    if len(DISABLED_CLASSIFIERS) > 0:
        for name in DISABLED_CLASSIFIERS:
            config = CLASSIFIER_CONFIGS[name]
            print(f"\n  {name}:")
            print(f"    Class: {config['class'].__name__}")
            print(f"    Reason: {config['notes']}")
            print(f"    Hyperparameter Space:")
            
            if name == 'gradient_boosting':
                print(f"      • max_iter: ~10 to 50 (log uniform)")
                print(f"      • learning_rate: 0.01 to 1.0 (log uniform)")
                print(f"      • max_depth: None, 3, 5, 7, 10")
                print(f"      • l2_regularization: 0.0001 to 10 (log uniform)")
                print(f"      • min_samples_leaf: 10 to 100 (log uniform)")
                print(f"      • max_bins: 32, 64, 128, 255")
            elif name == 'mlp':
                print(f"      • n_layers: 1 to 3 hidden layers")
                print(f"      • layer_sizes: 20-200 neurons per layer (log uniform)")
                print(f"      • alpha: 0.00001 to 0.1 (log uniform)")
                print(f"      • learning_rate_init: 0.0001 to 0.01 (log uniform)")
                print(f"      • activation: relu, tanh, logistic")
                print(f"      • max_iter: 100, 150, 200")
                print(f"      • early_stopping: True")
            elif name == 'knn':
                print(f"      • n_neighbors: 3 to 30 (log uniform)")
                print(f"      • weights: uniform, distance")
                print(f"      • p: 1 (Manhattan), 2 (Euclidean)")
                print(f"      • leaf_size: 10 to 100 (log uniform)")
    else:
        print("    None")
    
    # Stage 2 DNN Meta-Learner
    print("\n" + "=" * 80)
    print("STAGE 2 DNN META-LEARNER")
    print("=" * 80)
    print(f"\n{STAGE2_DNN_CONFIG['description']}")
    print(f"\n{STAGE2_DNN_CONFIG['notes']}")
    
    print("\n  Architecture:")
    print(f"    Input: {STAGE2_DNN_CONFIG['architecture']['input']}")
    for i, layer in enumerate(STAGE2_DNN_CONFIG['architecture']['hidden_layers'], 1):
        print(f"    Hidden Layer {i}: {layer['units']} units, {layer['activation']} activation, "
              f"{layer['dropout']:.0%} dropout")
    output = STAGE2_DNN_CONFIG['architecture']['output']
    print(f"    Output: {output['units']} unit, {output['activation']} activation")
    
    print("\n  Training Configuration:")
    training = STAGE2_DNN_CONFIG['training']
    print(f"    • Optimizer: {training['optimizer']}")
    print(f"    • Learning Rate: {training['learning_rate']}")
    print(f"    • Loss: {training['loss']}")
    print(f"    • Metrics: {', '.join(training['metrics'])}")
    print(f"    • Epochs: {training['epochs']}")
    print(f"    • Batch Size: {training['batch_size']}")
    print(f"    • Early Stopping: monitor={training['early_stopping']['monitor']}, "
          f"patience={training['early_stopping']['patience']}")
    print(f"    • Retrain Frequency: {training['retrain_frequency']}")
    
    print("\n  Data Splits:")
    splits = STAGE2_DNN_CONFIG['data_splits']
    print(f"    • Training Pool: {splits['training_pool']}")
    print(f"    • Validation S1: {splits['validation_s1']}")
    print(f"    • Validation S2: {splits['validation_s2']}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Print summary when run directly
    print_classifier_summary()
