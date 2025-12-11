"""Unit tests for ensemble configuration system.

This test suite validates that the EnsembleConfig produces correct
configuration values and maintains backward compatibility.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.config import (
    EnsembleConfig, HillClimbingConfig, Stage2Config, 
    PseudoLabelingConfig
)


class TestBasicInstantiation(unittest.TestCase):
    """Test basic configuration instantiation."""
    
    def test_default_instantiation(self):
        """Test that config can be instantiated with defaults."""
        config = EnsembleConfig()
        config.validate()
        self.assertIsNotNone(config)
    
    def test_random_state(self):
        """Test default random state."""
        config = EnsembleConfig()
        self.assertEqual(config.random_state, 315)


class TestConfigurationValues(unittest.TestCase):
    """Test that configuration values match original system."""
    
    def setUp(self):
        """Set up test config."""
        self.config = EnsembleConfig()
    
    def test_hill_climbing_config(self):
        """Test hill climbing configuration."""
        self.assertEqual(self.config.hill_climbing.max_iterations, 1000)
        self.assertEqual(self.config.hill_climbing.plateau_iterations, 100)
        self.assertEqual(self.config.hill_climbing.base_temperature, 0.0005)
        self.assertEqual(self.config.hill_climbing.temperature_decay, 0.998)
    
    def test_parallel_config(self):
        """Test parallel execution configuration."""
        self.assertEqual(self.config.parallel.batch_size, 20)
        self.assertEqual(self.config.parallel.n_workers, 20)
        self.assertEqual(self.config.parallel.timeout_minutes, 60)
    
    def test_stage1_config(self):
        """Test Stage 1 configuration."""
        self.assertEqual(len(self.config.stage1.active_classifiers), 14)
        self.assertIn('logistic', self.config.stage1.active_classifiers)
        self.assertIn('gradient_boosting', self.config.stage1.active_classifiers)
        self.assertEqual(self.config.stage1.sampling.row_sample_range, (0.05, 0.15))
        self.assertEqual(self.config.stage1.sampling.column_sample_range, (0.30, 0.70))
    
    def test_stage2_config(self):
        """Test Stage 2 configuration."""
        self.assertEqual(self.config.stage2.training.retrain_frequency, 20)
        self.assertEqual(self.config.stage2.training.epochs, 50)
        self.assertEqual(self.config.stage2.training.batch_size, 128)
        self.assertEqual(self.config.stage2.training.patience, 10)
        self.assertTrue(self.config.stage2.optimization.enabled)
        self.assertEqual(self.config.stage2.optimization.trials_per_optimization, 10)
        self.assertEqual(self.config.stage2.optimization.optimize_every_n_batches, 4)
        self.assertTrue(self.config.stage2.pseudo_labeling.enabled)
        self.assertEqual(self.config.stage2.pseudo_labeling.confidence_threshold, 0.75)
        self.assertEqual(self.config.stage2.pseudo_labeling.max_fraction, 0.20)
    
    def test_tracking_config(self):
        """Test tracking configuration."""
        self.assertTrue(self.config.tracking.enable_wal)


class TestClassifierConfigs(unittest.TestCase):
    """Test classifier configurations."""
    
    def setUp(self):
        """Set up test config."""
        self.config = EnsembleConfig()
        self.expected_classifiers = [
            'logistic', 'lasso', 'random_forest', 'linear_svc', 'sgd_classifier',
            'extra_trees', 'adaboost', 'naive_bayes', 'lda', 'qda', 'ridge',
            'gradient_boosting', 'mlp', 'knn'
        ]
    
    def test_all_classifiers_exist(self):
        """Test that all expected classifiers are configured."""
        for name in self.expected_classifiers:
            self.assertIn(name, self.config.stage1.classifiers,
                         f"Missing classifier: {name}")
    
    def test_classifier_properties(self):
        """Test that classifiers have required properties."""
        for name in self.expected_classifiers:
            clf_config = self.config.stage1.classifiers[name]
            self.assertTrue(clf_config.enabled)
            self.assertIsNotNone(clf_config.classifier_class)
            self.assertIsInstance(clf_config.hyperparameters, dict)


class TestTransformerConfigs(unittest.TestCase):
    """Test transformer configurations."""
    
    def setUp(self):
        """Set up test config."""
        self.config = EnsembleConfig()
        self.expected_transformers = [
            'ratio', 'product', 'difference', 'sum', 'reciprocal',
            'square', 'sqrt', 'log', 'binning', 'kde', 'kmeans',
            'nystroem', 'rbf_sampler', 'power_transform', 'quantile_transform',
            'noise_injector'
        ]
    
    def test_all_transformers_exist(self):
        """Test that all expected transformers are configured."""
        for name in self.expected_transformers:
            self.assertIn(name, 
                         self.config.stage1.feature_engineering.available_transformers,
                         f"Transformer '{name}' not in available_transformers")
            self.assertIn(name,
                         self.config.stage1.feature_engineering.transformer_hyperparams,
                         f"Transformer '{name}' not in transformer_hyperparams")


class TestCustomization(unittest.TestCase):
    """Test configuration customization."""
    
    def test_custom_random_state(self):
        """Test custom random state."""
        custom_config = EnsembleConfig(random_state=42)
        custom_config.validate()
        self.assertEqual(custom_config.random_state, 42)
    
    def test_custom_hill_climbing(self):
        """Test custom hill climbing config."""
        custom_config = EnsembleConfig(
            hill_climbing=HillClimbingConfig(max_iterations=2000)
        )
        custom_config.validate()
        self.assertEqual(custom_config.hill_climbing.max_iterations, 2000)
    
    def test_custom_pseudo_labeling(self):
        """Test disabling pseudo-labeling."""
        custom_config = EnsembleConfig(
            stage2=Stage2Config(
                pseudo_labeling=PseudoLabelingConfig(enabled=False)
            )
        )
        custom_config.validate()
        self.assertFalse(custom_config.stage2.pseudo_labeling.enabled)


class TestSummary(unittest.TestCase):
    """Test configuration summary generation."""
    
    def test_summary_generation(self):
        """Test that summary can be generated."""
        config = EnsembleConfig()
        summary = config.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("Ensemble Configuration Summary", summary)
        self.assertIn("Hill Climbing:", summary)
        self.assertIn("Stage 1:", summary)
        self.assertIn("Stage 2:", summary)
        self.assertGreater(len(summary), 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
