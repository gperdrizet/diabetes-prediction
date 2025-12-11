"""Test script to validate new configuration system.

This script verifies that the new EnsembleConfig produces the same
configuration as the original system, ensuring backward compatibility.
"""

import sys
from pathlib import Path

# Add parent directory to path to import ensemble package
sys.path.insert(0, str(Path(__file__).parent))

from ensemble.config import EnsembleConfig


def test_basic_instantiation():
    """Test that config can be instantiated with defaults."""
    print("Testing basic instantiation...")
    config = EnsembleConfig()
    config.validate()
    print("✓ Configuration instantiated and validated successfully")
    return config


def test_configuration_values(config):
    """Verify key configuration values match original system."""
    print("\nVerifying configuration values...")
    
    # Hill climbing
    assert config.hill_climbing.max_iterations == 1000
    assert config.hill_climbing.plateau_iterations == 100
    assert config.hill_climbing.base_temperature == 0.0005
    assert config.hill_climbing.temperature_decay == 0.998
    print("✓ Hill climbing config matches")
    
    # Parallel
    assert config.parallel.batch_size == 20
    assert config.parallel.n_workers == 20
    assert config.parallel.timeout_minutes == 60
    print("✓ Parallel config matches")
    
    # Stage 1
    assert len(config.stage1.active_classifiers) == 14
    assert 'logistic' in config.stage1.active_classifiers
    assert 'gradient_boosting' in config.stage1.active_classifiers
    assert config.stage1.sampling.row_sample_range == (0.05, 0.15)
    assert config.stage1.sampling.column_sample_range == (0.30, 0.70)
    print("✓ Stage 1 config matches")
    
    # Stage 2
    assert config.stage2.training.retrain_frequency == 20
    assert config.stage2.training.epochs == 50
    assert config.stage2.training.batch_size == 128
    assert config.stage2.training.patience == 10
    assert config.stage2.optimization.enabled == True
    assert config.stage2.optimization.trials_per_optimization == 10
    assert config.stage2.optimization.optimize_every_n_batches == 4
    assert config.stage2.pseudo_labeling.enabled == True
    assert config.stage2.pseudo_labeling.confidence_threshold == 0.75
    assert config.stage2.pseudo_labeling.max_fraction == 0.20
    print("✓ Stage 2 config matches")
    
    # Tracking
    assert config.tracking.enable_wal == True
    print("✓ Tracking config matches")


def test_classifier_configs(config):
    """Verify classifier configurations."""
    print("\nVerifying classifier configurations...")
    
    # Check all expected classifiers exist
    expected = [
        'logistic', 'lasso', 'random_forest', 'linear_svc', 'sgd_classifier',
        'extra_trees', 'adaboost', 'naive_bayes', 'lda', 'qda', 'ridge',
        'gradient_boosting', 'mlp', 'knn'
    ]
    
    for name in expected:
        assert name in config.stage1.classifiers, f"Missing classifier: {name}"
        clf_config = config.stage1.classifiers[name]
        assert clf_config.enabled == True
        assert clf_config.classifier_class is not None
        assert isinstance(clf_config.hyperparameters, dict)
    
    print(f"✓ All {len(expected)} classifiers configured correctly")


def test_transformer_configs(config):
    """Verify transformer configurations."""
    print("\nVerifying transformer configurations...")
    
    expected = [
        'ratio', 'product', 'difference', 'sum', 'reciprocal',
        'square', 'sqrt', 'log', 'binning', 'kde', 'kmeans',
        'nystroem', 'rbf_sampler', 'power_transform', 'quantile_transform',
        'noise_injector'
    ]
    
    for name in expected:
        assert name in config.stage1.feature_engineering.available_transformers, \
            f"Transformer '{name}' not in available_transformers"
        assert name in config.stage1.feature_engineering.transformer_hyperparams, \
            f"Transformer '{name}' not in transformer_hyperparams"
    
    print(f"✓ All {len(expected)} transformers configured correctly")


def test_customization():
    """Test that configuration can be customized."""
    print("\nTesting configuration customization...")
    
    from ensemble.config import (
        EnsembleConfig, HillClimbingConfig, Stage2Config, 
        PseudoLabelingConfig
    )
    
    custom_config = EnsembleConfig(
        random_state=42,
        hill_climbing=HillClimbingConfig(max_iterations=2000),
        stage2=Stage2Config(
            pseudo_labeling=PseudoLabelingConfig(enabled=False)
        )
    )
    
    custom_config.validate()
    
    assert custom_config.random_state == 42
    assert custom_config.hill_climbing.max_iterations == 2000
    assert custom_config.stage2.pseudo_labeling.enabled == False
    
    print("✓ Custom configuration works correctly")


def test_summary():
    """Test configuration summary generation."""
    print("\nTesting configuration summary...")
    
    config = EnsembleConfig()
    summary = config.summary()
    
    assert "Ensemble Configuration Summary" in summary
    assert "Hill Climbing:" in summary
    assert "Stage 1:" in summary
    assert "Stage 2:" in summary
    
    print("✓ Configuration summary generated successfully")
    print("\n" + "="*50)
    print(summary)
    print("="*50)


def main():
    """Run all tests."""
    print("="*70)
    print("ENSEMBLE CONFIGURATION VALIDATION")
    print("="*70)
    
    try:
        # Run tests
        config = test_basic_instantiation()
        test_configuration_values(config)
        test_classifier_configs(config)
        test_transformer_configs(config)
        test_customization()
        test_summary()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nConfiguration system is working correctly and matches original.")
        print("Ready to proceed with Phase 2 (Core Abstractions).")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
