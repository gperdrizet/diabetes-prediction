"""Test script for Phase 3: Stage 1 Isolation.

Validates that Stage 1 modules work correctly:
- ClassifierPool
- Transformer factory
- Integration with config
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ensemble.config import EnsembleConfig
from ensemble.stage1 import ClassifierPool, get_transformer
from ensemble.stage1.transformers import get_available_transformers


def test_classifier_pool():
    """Test ClassifierPool functionality."""
    print("Testing ClassifierPool...")
    
    # Create config and pool
    config = EnsembleConfig()
    pool = ClassifierPool(config.stage1, random_state=42)
    
    # Test getting active classifiers
    active = pool.get_active_classifiers()
    assert len(active) == 14, f"Expected 14 active classifiers, got {len(active)}"
    print(f"  ✓ Active classifiers: {len(active)}")
    
    # Test getting specific config
    rf_config = pool.get_config('random_forest')
    assert rf_config.enabled == True
    assert rf_config.classifier_class.__name__ == 'RandomForestClassifier'
    print(f"  ✓ Got config for random_forest")
    
    # Test hyperparameter sampling
    params = pool.sample_hyperparameters('logistic')
    assert 'solver' in params
    assert 'max_iter' in params
    print(f"  ✓ Sampled logistic hyperparameters: {len(params)} params")
    
    # Test hyperparameter sampling for classifier with n_jobs
    rf_params = pool.sample_hyperparameters('random_forest', n_jobs=4)
    assert 'n_estimators' in rf_params
    assert 'max_depth' in rf_params
    print(f"  ✓ Sampled random_forest hyperparameters: {len(rf_params)} params")
    
    # Test building specific classifier
    logistic = pool.build_classifier('logistic')
    assert hasattr(logistic, 'fit')
    assert hasattr(logistic, 'predict')
    print(f"  ✓ Built logistic classifier: {type(logistic).__name__}")
    
    # Test sampling random classifier
    clf, name, params = pool.sample_classifier(n_jobs=2)
    assert name in active
    assert hasattr(clf, 'fit')
    assert isinstance(params, dict)
    print(f"  ✓ Sampled random classifier: {name} with {len(params)} params")
    
    # Test validation
    valid, msg = pool.validate_classifier('random_forest')
    assert valid == True, f"Validation failed: {msg}"
    print(f"  ✓ Validation: {msg}")
    
    # Test validation of all classifiers
    results = pool.validate_all()
    valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
    print(f"  ✓ Validated all classifiers: {valid_count}/{len(results)} valid")
    
    # Test pool summary
    summary = pool.get_pool_summary()
    assert "Classifier Pool Summary" in summary
    assert "Active classifiers: 14" in summary
    print(f"  ✓ Generated pool summary ({len(summary)} chars)")


def test_transformers():
    """Test transformer factory and transformers."""
    print("\nTesting Transformers...")
    
    # Get available transformers
    available = get_available_transformers()
    assert len(available) > 0
    print(f"  ✓ Available transformers: {len(available)}")
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    # Test ratio transformer
    ratio = get_transformer('ratio', n_features=5, random_state=42)
    ratio.fit(X.values)
    X_ratio = ratio.transform(X.values)
    assert X_ratio.shape[0] == X.shape[0], "Should preserve number of samples"
    print(f"  ✓ Ratio transformer: {X.shape[1]} → {X_ratio.shape[1]} features")
    
    # Test product transformer
    product = get_transformer('product', n_features=5, random_state=42)
    product.fit(X.values)
    X_product = product.transform(X.values)
    assert X_product.shape[0] == X.shape[0]
    print(f"  ✓ Product transformer: {X.shape[1]} → {X_product.shape[1]} features")
    
    # Test reciprocal transformer
    reciprocal = get_transformer('reciprocal')
    reciprocal.fit(X.values)
    X_reciprocal = reciprocal.transform(X.values)
    assert X_reciprocal.shape == X.values.shape, "Reciprocal should preserve shape"
    print(f"  ✓ Reciprocal transformer: shape preserved {X_reciprocal.shape}")
    
    # Test log transformer
    log_tf = get_transformer('log')
    log_tf.fit(X.values)
    X_log = log_tf.transform(X.values)
    assert X_log.shape == X.values.shape
    print(f"  ✓ Log transformer: shape preserved {X_log.shape}")
    
    # Test sqrt transformer
    sqrt_tf = get_transformer('sqrt')
    sqrt_tf.fit(X.values)
    X_sqrt = sqrt_tf.transform(X.values)
    assert X_sqrt.shape == X.values.shape
    print(f"  ✓ Sqrt transformer: shape preserved {X_sqrt.shape}")
    
    # Test square transformer
    square_tf = get_transformer('square')
    square_tf.fit(X.values)
    X_square = square_tf.transform(X.values)
    assert X_square.shape == X.values.shape
    print(f"  ✓ Square transformer: shape preserved {X_square.shape}")
    
    # Test binning transformer
    binning = get_transformer('binning', n_bins=5, strategy='quantile', encode='ordinal')
    binning.fit(X.values)
    X_binned = binning.transform(X.values)
    assert X_binned.shape == X.values.shape
    print(f"  ✓ Binning transformer: shape preserved {X_binned.shape}")
    
    # Test KDE transformer
    kde = get_transformer('kde', bandwidth='scott')
    kde.fit(X.values)
    X_kde = kde.transform(X.values)
    assert X_kde.shape[0] == X.shape[0]
    print(f"  ✓ KDE transformer: {X.shape[1]} → {X_kde.shape[1]} features")
    
    # Test kmeans transformer
    kmeans = get_transformer('kmeans', n_clusters=3, add_distances=True, random_state=42)
    kmeans.fit(X.values)
    X_kmeans = kmeans.transform(X.values)
    assert X_kmeans.shape[0] == X.shape[0]
    print(f"  ✓ KMeans transformer: {X.shape[1]} → {X_kmeans.shape[1]} features")
    
    # Test noise injector
    noise = get_transformer('noise_injector', feature_fraction=0.5, 
                           noise_scale_range=(0.01, 0.1), random_state=42)
    noise.fit(X.values)
    X_noise = noise.transform(X.values)
    assert X_noise.shape == X.values.shape
    assert not np.array_equal(X.values, X_noise), "Noise should modify data"
    print(f"  ✓ Noise injector: shape preserved, data modified")


def test_integration():
    """Test integration between components."""
    print("\nTesting integration...")
    
    # Create config
    config = EnsembleConfig(random_state=42)
    
    # Create pool
    pool = ClassifierPool(config.stage1, random_state=42)
    
    # Sample a classifier
    clf, name, params = pool.sample_classifier(n_jobs=2)
    print(f"  ✓ Sampled classifier: {name}")
    
    # Create data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    y = pd.Series(np.random.choice([0, 1], size=100))
    
    # Apply transformer
    transformer = get_transformer('ratio', n_features=3, random_state=42)
    transformer.fit(X.values)
    X_transformed = transformer.transform(X.values)
    print(f"  ✓ Applied transformer: {X.shape} → {X_transformed.shape}")
    
    # Train classifier
    clf.fit(X_transformed, y)
    predictions = clf.predict(X_transformed)
    assert len(predictions) == len(y)
    print(f"  ✓ Trained and predicted: {len(predictions)} predictions")
    
    # Test with multiple transformers
    transformers = [
        get_transformer('product', n_features=3, random_state=42),
        get_transformer('log'),
        get_transformer('sqrt')
    ]
    
    X_multi = X.values.copy()
    for tf in transformers:
        tf.fit(X_multi)
        X_multi = tf.transform(X_multi)
    
    print(f"  ✓ Chained transformers: {X.shape} → {X_multi.shape}")
    
    # Train on multi-transformed data
    clf2, name2, _ = pool.sample_classifier()
    clf2.fit(X_multi, y)
    predictions2 = clf2.predict(X_multi)
    print(f"  ✓ Trained {name2} on transformed data: {len(predictions2)} predictions")


def test_sklearn_compatibility():
    """Test that all classifiers work with sklearn."""
    print("\nTesting sklearn compatibility...")
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, roc_auc_score
    
    config = EnsembleConfig(random_state=42)
    pool = ClassifierPool(config.stage1, random_state=42)
    
    # Create data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 20))
    y = pd.Series(np.random.choice([0, 1], size=200, p=[0.7, 0.3]))
    
    # Test a few classifiers with cross-validation
    test_classifiers = ['logistic', 'random_forest', 'gradient_boosting']
    
    for clf_name in test_classifiers:
        clf = pool.build_classifier(clf_name, n_jobs=1)
        
        # Quick cross-validation
        scores = cross_val_score(
            clf, X, y, 
            cv=3, 
            scoring=make_scorer(roc_auc_score),
            error_score='raise'
        )
        
        mean_score = np.mean(scores)
        print(f"  ✓ {clf_name}: CV AUC = {mean_score:.3f}")


def main():
    """Run all tests."""
    print("="*70)
    print("PHASE 3: STAGE 1 ISOLATION VALIDATION")
    print("="*70)
    
    try:
        test_classifier_pool()
        test_transformers()
        test_integration()
        test_sklearn_compatibility()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nStage 1 modules are working correctly.")
        print("Ready to proceed with Phase 4 (Stage 2 Refactoring).")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
