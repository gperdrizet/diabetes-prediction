#!/usr/bin/env python3
"""Standalone configuration viewer - displays ensemble config without sklearn dependency."""

# Active classifiers
ACTIVE_CLASSIFIERS = [
    'logistic', 'random_forest', 'linear_svc', 'sgd_classifier',
    'extra_trees', 'adaboost', 'naive_bayes', 'lda', 'qda', 'ridge'
]

# Disabled classifiers
DISABLED_CLASSIFIERS = ['gradient_boosting', 'mlp', 'knn']

# Configuration notes
CLASSIFIER_NOTES = {
    'logistic': {
        'class': 'LogisticRegression',
        'status': 'ACTIVE',
        'notes': 'Fast L2-regularized linear model. Optimized for speed: reduced max_iter (100-300), narrowed C range (0.1-10), removed L1 penalty.',
        'key_params': 'C=0.1-10, penalty=[l2, None], max_iter=100-300, tol=1e-3'
    },
    'random_forest': {
        'class': 'RandomForestClassifier',
        'status': 'ACTIVE',
        'notes': 'Ensemble of decision trees. Parallelizable with n_jobs. Good balance of speed and performance.',
        'key_params': 'n_estimators=10-100, max_depth=[3,5,7,10,15,20,None], min_samples_split=2-20'
    },
    'linear_svc': {
        'class': 'LinearSVC',
        'status': 'ACTIVE',
        'notes': 'Fast linear SVM. Optimized: reduced C range (0.1-10), max_iter=200-300, dual=False for many features.',
        'key_params': 'C=0.1-10, loss=squared_hinge, max_iter=200-300, dual=False, tol=1e-3'
    },
    'sgd_classifier': {
        'class': 'SGDClassifier',
        'status': 'ACTIVE',
        'notes': 'Stochastic gradient descent. Fast online learning, good for large datasets.',
        'key_params': 'loss=[hinge,log_loss,modified_huber,perceptron], penalty=[l2,l1,elasticnet], max_iter=300-800'
    },
    'extra_trees': {
        'class': 'ExtraTreesClassifier',
        'status': 'ACTIVE',
        'notes': 'Extremely randomized trees. Optimized: reduced n_estimators (5-30), max_depth ≤10, forced feature subsampling.',
        'key_params': 'n_estimators=5-30, max_depth=[3,5,7,10], min_samples_split=5-30, max_features=[sqrt,log2]'
    },
    'adaboost': {
        'class': 'AdaBoostClassifier',
        'status': 'ACTIVE',
        'notes': 'Adaptive boosting. Sequential but relatively fast with reasonable n_estimators.',
        'key_params': 'n_estimators=10-100, learning_rate=0.1-3.0, algorithm=SAMME'
    },
    'naive_bayes': {
        'class': 'GaussianNB/MultinomialNB/BernoulliNB',
        'status': 'ACTIVE',
        'notes': 'Probabilistic classifier. Three variants: Gaussian (continuous), Multinomial (counts), Bernoulli (binary).',
        'key_params': 'type=[gaussian,multinomial,bernoulli], alpha=0.01-10, var_smoothing=1e-12 to 1e-6'
    },
    'lda': {
        'class': 'LinearDiscriminantAnalysis',
        'status': 'ACTIVE',
        'notes': 'Linear Discriminant Analysis. Fast, assumes Gaussian distributions with shared covariance.',
        'key_params': 'solver=[svd,lsqr,eigen], shrinkage=[None,auto,0.0-1.0]'
    },
    'qda': {
        'class': 'QuadraticDiscriminantAnalysis',
        'status': 'ACTIVE',
        'notes': 'Quadratic Discriminant Analysis. More flexible than LDA, models non-linear boundaries.',
        'key_params': 'reg_param=0.0001-1.0'
    },
    'ridge': {
        'class': 'RidgeClassifier',
        'status': 'ACTIVE',
        'notes': 'Fast L2-regularized linear model. Very fast training, good baseline.',
        'key_params': 'alpha=0.01-100, solver=[auto,cholesky,lsqr], tol=1e-3'
    },
    'gradient_boosting': {
        'class': 'HistGradientBoostingClassifier',
        'status': 'DISABLED',
        'notes': 'Sequential tree building causes 30+ min timeouts. Was optimized but still too slow.',
        'key_params': 'max_iter=10-50, learning_rate=0.01-1.0, max_depth=[None,3,5,7,10]'
    },
    'mlp': {
        'class': 'MLPClassifier',
        'status': 'DISABLED',
        'notes': 'Multi-layer perceptron causes timeouts. Neural network training is too slow.',
        'key_params': 'hidden_layers=1-3, neurons=20-200, max_iter=100-200, early_stopping=True'
    },
    'knn': {
        'class': 'KNeighborsClassifier',
        'status': 'DISABLED',
        'notes': 'Distance calculations on large datasets cause timeouts.',
        'key_params': 'n_neighbors=3-30, weights=[uniform,distance], p=[1,2]'
    }
}


def print_config():
    """Print formatted configuration summary."""
    print("=" * 100)
    print("ENSEMBLE CLASSIFIER CONFIGURATION")
    print("=" * 100)
    
    print(f"\n{'ACTIVE CLASSIFIERS':<50} ({len(ACTIVE_CLASSIFIERS)} total)")
    print("-" * 100)
    for name in ACTIVE_CLASSIFIERS:
        config = CLASSIFIER_NOTES[name]
        print(f"\n{name.upper()}")
        print(f"  Class:  {config['class']}")
        print(f"  Notes:  {config['notes']}")
        print(f"  Params: {config['key_params']}")
    
    print("\n\n" + "=" * 100)
    print(f"{'DISABLED CLASSIFIERS':<50} ({len(DISABLED_CLASSIFIERS)} total)")
    print("-" * 100)
    for name in DISABLED_CLASSIFIERS:
        config = CLASSIFIER_NOTES[name]
        print(f"\n{name.upper()}")
        print(f"  Class:  {config['class']}")
        print(f"  Reason: {config['notes']}")
        print(f"  Params: {config['key_params']}")
    
    print("\n\n" + "=" * 100)
    print("SAMPLING CONFIGURATION")
    print("=" * 100)
    print("\nRow Sampling: 10.0% to 40.0%")
    print("  Higher sampling = stronger models, lower = more diversity.")
    print("  Range gives 2,400-9,600 rows from ~24,000 pool.")
    print("\nColumn Sampling: 30.0% to 70.0%")
    print("  Random feature selection for diversity. Reduced range increases diversity.")
    
    print("\n" + "=" * 100)
    print("FEATURE ENGINEERING CONFIGURATION")
    print("=" * 100)
    print("\nSkip Probability: 30% (use raw features only)")
    print("Transformers Applied: 1-3 (when not skipping)")
    print("Available: 16 transformer types")
    print("  [ratio, product, difference, sum, reciprocal, square, sqrt, log,")
    print("   binning, kde, kmeans, nystroem, rbf_sampler, power_transform,")
    print("   quantile_transform, standard_scaler, noise_injector]")
    
    print("\n" + "=" * 100)


if __name__ == '__main__':
    print_config()
