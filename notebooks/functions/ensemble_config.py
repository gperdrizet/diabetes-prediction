"""Configuration file for ensemble model hyperparameters.

This file centralizes all classifier definitions and their hyperparameter ranges.
All parameters are defined as importable constants for use during training.
"""

from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


# ==============================================================================
# CLASSIFIER POOL
# ==============================================================================

ACTIVE_CLASSIFIERS = [
    'logistic', 'lasso', 'random_forest', 'linear_svc', 'sgd_classifier',
    'extra_trees', 'naive_bayes', 'lda', 'ridge',
    'gradient_boosting', 'mlp', 'knn'
]

DISABLED_CLASSIFIERS = ['qda', 'adaboost']  # Disabled: too slow with large samples


# ==============================================================================
# CLASSIFIER HYPERPARAMETER GENERATORS
# ==============================================================================

CLASSIFIER_CONFIGS = {
    'logistic': {
        'class': LogisticRegression,
        'hyperparameters': {
            'C': lambda rng: 10 ** rng.uniform(-1, 1),
            'penalty': 'l2',
            'solver': lambda rng: rng.choice(['lbfgs', 'newton-cg', 'sag']),
            'max_iter': lambda rng: rng.choice([100, 200, 300]),
            'class_weight': 'balanced',
            'tol': 1e-3
        }
    },
    'lasso': {
        'class': LogisticRegression,
        'hyperparameters': {
            'C': lambda rng: 10 ** rng.uniform(-1, 1),
            'penalty': 'l1',
            'solver': lambda rng: rng.choice(['liblinear', 'saga']),
            'max_iter': lambda rng: rng.choice([100, 200, 300]),
            'class_weight': 'balanced',
            'tol': 1e-3
        }
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'hyperparameters': {
            'n_estimators': lambda rng: int(10 ** rng.uniform(1.0, 2.0)),
            'max_depth': lambda rng: rng.choice([3, 5, 7, 10, 15, 20, None]),
            'min_samples_split': lambda rng: int(10 ** rng.uniform(0.3, 1.3)),
            'min_samples_leaf': lambda rng: int(10 ** rng.uniform(0, 1)),
            'max_features': lambda rng: rng.choice(['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1
        }
    },
    'linear_svc': {
        'class': LinearSVC,
        'hyperparameters': {
            'C': lambda rng: 10 ** rng.uniform(-1, 1),
            'loss': 'squared_hinge',
            'max_iter': lambda rng: rng.choice([200, 300]),
            'class_weight': 'balanced',
            'dual': True,
            'tol': 1e-3
        }
    },
    'sgd_classifier': {
        'class': SGDClassifier,
        'hyperparameters': {
            'loss': lambda rng: rng.choice(['hinge', 'log_loss', 'modified_huber', 'perceptron']),
            'penalty': lambda rng: rng.choice(['l2', 'l1', 'elasticnet']),
            'alpha': lambda rng: 10 ** rng.uniform(-5, -1),
            'learning_rate': lambda rng: rng.choice(['optimal', 'adaptive', 'constant']),
            'eta0': lambda rng: 10 ** rng.uniform(-4, -1),
            'max_iter': lambda rng: rng.choice([300, 500, 800]),
            'early_stopping': True,
            'class_weight': 'balanced'
        }
    },
    'extra_trees': {
        'class': ExtraTreesClassifier,
        'hyperparameters': {
            'n_estimators': lambda rng: int(10 ** rng.uniform(0.7, 1.5)),
            'max_depth': lambda rng: rng.choice([3, 5, 7, 10]),
            'min_samples_split': lambda rng: int(10 ** rng.uniform(0.7, 1.5)),
            'min_samples_leaf': lambda rng: int(10 ** rng.uniform(0.5, 1.3)),
            'max_features': lambda rng: rng.choice(['sqrt', 'log2']),
            'class_weight': 'balanced',
            'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1,
            'min_impurity_decrease': 1e-7
        }
    },
    'adaboost': {
        'class': AdaBoostClassifier,
        'hyperparameters': {
            'n_estimators': lambda rng: int(10 ** rng.uniform(1.0, 2.0)),
            'learning_rate': lambda rng: 10 ** rng.uniform(-1.0, 0.5),
            'algorithm': 'SAMME'
        }
    },
    'naive_bayes': {
        'class': BernoulliNB,
        'hyperparameters': {
            'alpha': lambda rng: 10 ** rng.uniform(-2, 1),
            'fit_prior': lambda rng: rng.choice([True, False]),
            'binarize': lambda rng: rng.choice([None, 0.0, rng.uniform(0.3, 0.7)])
        }
    },
    'lda': {
        'class': LinearDiscriminantAnalysis,
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
        'hyperparameters': {
            'reg_param': lambda rng: 10 ** rng.uniform(-4, 0)
        }
    },
    'ridge': {
        'class': RidgeClassifier,
        'hyperparameters': {
            'alpha': lambda rng: 10 ** rng.uniform(-2, 2),
            'solver': lambda rng: rng.choice(['auto', 'cholesky', 'lsqr']),
            'class_weight': 'balanced',
            'tol': 1e-3
        }
    },
    'gradient_boosting': {
        'class': HistGradientBoostingClassifier,
        'hyperparameters': {
            'max_iter': lambda rng: int(10 ** rng.uniform(1.0, 1.7)),
            'learning_rate': lambda rng: 10 ** rng.uniform(-2.0, 0),
            'max_depth': lambda rng: rng.choice([None, 3, 5, 7, 10]),
            'l2_regularization': lambda rng: 10 ** rng.uniform(-4, 1),
            'min_samples_leaf': lambda rng: int(10 ** rng.uniform(1, 2)),
            'max_bins': lambda rng: rng.choice([32, 64, 128, 255])
        }
    },
    'mlp': {
        'class': MLPClassifier,
        'hyperparameters': {
            'n_layers': lambda rng: rng.randint(1, 4),
            'layer_sizes': lambda rng, n_layers: tuple([int(10 ** rng.uniform(1.3, 2.3)) for _ in range(n_layers)]),
            'alpha': lambda rng: 10 ** rng.uniform(-5, -1),
            'learning_rate_init': lambda rng: 10 ** rng.uniform(-4, -2),
            'activation': lambda rng: rng.choice(['relu', 'tanh', 'logistic']),
            'max_iter': lambda rng: rng.choice([100, 150, 200]),
            'early_stopping': True
        }
    },
    'knn': {
        'class': KNeighborsClassifier,
        'hyperparameters': {
            'n_neighbors': lambda rng: int(10 ** rng.uniform(0.5, 1.5)),
            'weights': lambda rng: rng.choice(['uniform', 'distance']),
            'p': lambda rng: rng.choice([1, 2]),
            'leaf_size': lambda rng: int(10 ** rng.uniform(1, 2)),
            'n_jobs': lambda rng, n_jobs: n_jobs if n_jobs > 1 else 1
        }
    }
}


# ==============================================================================
# DATA SAMPLING CONFIGURATION
# ==============================================================================

SAMPLING_CONFIG = {
    'row_sample_pct': {'min': 0.10, 'max': 0.40},
    'col_sample_pct': {'min': 0.30, 'max': 0.70}
}


# ==============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ==============================================================================

FEATURE_ENGINEERING_CONFIG = {
    'skip_probability': 0.30,
    'n_transformers': {'min': 1, 'max': 3},
    'available_transformers': [
        'ratio', 'product', 'difference', 'sum', 'reciprocal',
        'square', 'sqrt', 'log', 'binning', 'iqr_clipper', 'kde', 'kmeans',
        'nystroem', 'rbf_sampler', 'power_transform', 'quantile_transform',
        'noise_injector'
    ]
}

# Transformer hyperparameter generators
TRANSFORMER_HYPERPARAMS = {
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
    'reciprocal': {},  # No hyperparameters
    'square': {},  # No hyperparameters
    'sqrt': {},  # No hyperparameters
    'log': {},  # No hyperparameters
    'binning': {
        'n_bins': lambda rng: rng.randint(3, 8),
        'strategy': lambda rng: rng.choice(['quantile', 'uniform']),
        'encode': 'ordinal'
    },
    'iqr_clipper': {
        'iqr_multiplier': lambda rng: rng.uniform(1.5, 3.5)
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
        'n_components': lambda rng, n_features=None: min(int(10 ** rng.uniform(1.5, 2.5)), n_features - 1) if n_features else int(10 ** rng.uniform(1.5, 2.5)),
        'gamma': lambda rng, kernel: 10 ** rng.uniform(-3, 0) if kernel in ['rbf', 'poly', 'sigmoid'] else None,
        'degree': lambda rng, kernel: rng.randint(2, 5) if kernel == 'poly' else 3
    },
    'rbf_sampler': {
        'n_components': lambda rng, n_features=None: min(int(10 ** rng.uniform(1.5, 2.5)), n_features - 1) if n_features else int(10 ** rng.uniform(1.5, 2.5)),
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
        'noise_scale_min': lambda rng: rng.uniform(0.001, 0.05),
        'noise_scale_max': lambda rng: rng.uniform(0.05, 0.3)
    }
}


# ==============================================================================
# DIMENSIONALITY REDUCTION CONFIGURATION
# ==============================================================================

DIM_REDUCTION_CONFIG = {
    'use_probability': 0.5,
    'available_methods': ['pca', 'truncated_svd', 'fast_ica', 'factor_analysis']
}

# Dimensionality reduction hyperparameter generators
DIM_REDUCTION_HYPERPARAMS = {
    'pca': {
        'n_components': lambda rng: rng.choice([0.80, 0.85, 0.90, 0.95, 0.99]),
        'svd_solver': 'full',
        'whiten': False
    },
    'truncated_svd': {
        'n_components': lambda rng, n_features: min(int(10 ** rng.uniform(0.7, 1.7)), n_features - 1) if n_features else int(10 ** rng.uniform(0.7, 1.7)),
        'algorithm': 'randomized',
        'n_iter': 5
    },
    'fast_ica': {
        'n_components': lambda rng, n_features: min(int(10 ** rng.uniform(0.7, 1.7)), n_features - 1) if n_features else int(10 ** rng.uniform(0.7, 1.7)),
        'whiten': lambda rng: rng.choice(['unit-variance', 'arbitrary-variance', True, False]),
        'max_iter': lambda rng: rng.randint(200, 501),
        'algorithm': lambda rng: rng.choice(['parallel', 'deflation'])
    },
    'factor_analysis': {
        'n_components': lambda rng, n_features: min(int(10 ** rng.uniform(0.7, 1.7)), n_features - 1) if n_features else int(10 ** rng.uniform(0.7, 1.7)),
        'max_iter': 1000,
        'tol': 0.01
    }
}


# ==============================================================================
# INITIAL SCALER CONFIGURATION
# ==============================================================================

# INITIAL_SCALER_OPTIONS = ['standard', 'minmax', 'robust']


# ==============================================================================
# STAGE 2 DNN META-LEARNER CONFIGURATION
# ==============================================================================

STAGE2_DNN_CONFIG = {
    'architecture': {
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
        }
    },
    'retrain_frequency': 10  # Retrain every N accepted models
}

