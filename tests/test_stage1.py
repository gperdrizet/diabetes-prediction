"""Unit tests for Stage 1 classifiers and transformers.

This test suite validates ClassifierPool and transformer functionality.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.stage1 import ClassifierPool
from ensemble.stage1.transformers import (
    RatioTransformer, ProductTransformer, ReciprocalTransformer,
    LogTransformer, SquareRootTransformer, SquareTransformer,
    BinningTransformer, KDESmoothingTransformer, KMeansClusterTransformer,
    NoiseInjector
)
from ensemble.config import EnsembleConfig


class TestClassifierPool(unittest.TestCase):
    """Test ClassifierPool functionality."""
    
    def setUp(self):
        """Set up test pool."""
        config = EnsembleConfig()
        self.pool = ClassifierPool(
            config=config.stage1,
            random_state=42
        )
    
    def test_active_classifiers(self):
        """Test getting active classifiers."""
        active = self.pool.get_active_classifiers()
        
        self.assertIsInstance(active, list)
        self.assertEqual(len(active), 14)
        self.assertIn('logistic', active)
        self.assertIn('random_forest', active)
    
    def test_get_config(self):
        """Test getting classifier configuration."""
        config = self.pool.get_config('random_forest')
        
        self.assertIsNotNone(config)
        self.assertTrue(config.enabled)
        self.assertIsNotNone(config.classifier_class)
    
    def test_sample_hyperparameters(self):
        """Test hyperparameter sampling."""
        params = self.pool.sample_hyperparameters('logistic')
        
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)
        
        # Test with n_jobs
        params_with_jobs = self.pool.sample_hyperparameters('random_forest', n_jobs=4)
        self.assertIn('n_jobs', params_with_jobs)
        self.assertEqual(params_with_jobs['n_jobs'], 4)
    
    def test_build_classifier(self):
        """Test building a classifier."""
        clf = self.pool.build_classifier('logistic')
        
        self.assertIsNotNone(clf)
        self.assertTrue(hasattr(clf, 'fit'))
        self.assertTrue(hasattr(clf, 'predict'))
    
    def test_sample_classifier(self):
        """Test sampling random classifier."""
        clf, name, params = self.pool.sample_classifier()
        
        self.assertIsNotNone(clf)
        self.assertIsInstance(name, str)
        self.assertIsInstance(params, dict)
        self.assertIn(name, self.pool.get_active_classifiers())
    
    def test_validate_all_classifiers(self):
        """Test that all classifiers can be built."""
        for name in self.pool.get_active_classifiers():
            with self.subTest(classifier=name):
                clf = self.pool.build_classifier(name)
                self.assertIsNotNone(clf)
                self.assertTrue(hasattr(clf, 'fit'))


class TestTransformers(unittest.TestCase):
    """Test custom transformer functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
    
    def test_ratio_transformer(self):
        """Test RatioTransformer."""
        transformer = RatioTransformer(n_features=5, random_state=42)
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertEqual(X_transformed.shape[1], 5)
    
    def test_product_transformer(self):
        """Test ProductTransformer."""
        transformer = ProductTransformer(n_features=5, random_state=42)
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertEqual(X_transformed.shape[1], 5)
    
    def test_reciprocal_transformer(self):
        """Test ReciprocalTransformer."""
        transformer = ReciprocalTransformer()
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
        # Check no infinities or NaNs
        self.assertFalse(np.any(np.isinf(X_transformed)))
        self.assertFalse(np.any(np.isnan(X_transformed)))
    
    def test_log_transformer(self):
        """Test LogTransformer."""
        transformer = LogTransformer()
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertFalse(np.any(np.isinf(X_transformed)))
        self.assertFalse(np.any(np.isnan(X_transformed)))
    
    def test_sqrt_transformer(self):
        """Test SquareRootTransformer."""
        transformer = SquareRootTransformer()
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertFalse(np.any(np.isnan(X_transformed)))
    
    def test_square_transformer(self):
        """Test SquareTransformer."""
        transformer = SquareTransformer()
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
    
    def test_binning_transformer(self):
        """Test BinningTransformer."""
        transformer = BinningTransformer(n_bins=5, random_state=42)
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
    
    def test_kde_transformer(self):
        """Test KDESmoothingTransformer."""
        transformer = KDESmoothingTransformer()
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
    
    def test_kmeans_transformer(self):
        """Test KMeansClusterTransformer."""
        transformer = KMeansClusterTransformer(n_clusters=4, random_state=42)
        X_transformed = transformer.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        # Output is 1 (cluster label) + 4 (distances to each cluster) = 5
        self.assertEqual(X_transformed.shape[1], 5)
    
    def test_noise_injector(self):
        """Test NoiseInjector."""
        transformer = NoiseInjector(feature_fraction=0.5, random_state=42)
        X_transformed = transformer.fit_transform(self.X.copy())
        
        self.assertEqual(X_transformed.shape, self.X.shape)
        # Data should be different but similar
        self.assertFalse(np.array_equal(X_transformed, self.X))


class TestIntegration(unittest.TestCase):
    """Test integration of classifiers and transformers."""
    
    def setUp(self):
        """Set up integrated test."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        self.X = X  # Keep as numpy array for transformers
        self.y = y
        
        config = EnsembleConfig()
        self.pool = ClassifierPool(config.stage1, random_state=42)
    
    def test_classifier_with_transformer(self):
        """Test using classifier with transformed data."""
        # Build classifier
        clf = self.pool.build_classifier('logistic')
        
        # Transform data
        transformer = RatioTransformer(n_features=3, random_state=42)
        X_transformed = transformer.fit_transform(self.X)
        
        # Train and predict
        clf.fit(X_transformed, self.y)
        predictions = clf.predict(X_transformed)
        
        self.assertEqual(len(predictions), len(self.y))
    
    def test_chained_transformers(self):
        """Test chaining multiple transformers."""
        from sklearn.pipeline import Pipeline
        
        pipeline = Pipeline([
            ('ratio', RatioTransformer(n_features=5, random_state=42)),
            ('log', LogTransformer()),
            ('final', RatioTransformer(n_features=3, random_state=42))
        ])
        
        X_transformed = pipeline.fit_transform(self.X)
        self.assertEqual(X_transformed.shape[1], 3)


class TestSklearnCompatibility(unittest.TestCase):
    """Test sklearn compatibility with cross-validation."""
    
    def setUp(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        self.X = X
        self.y = y
        
        config = EnsembleConfig()
        self.pool = ClassifierPool(config.stage1, random_state=42)
    
    def test_cross_validation(self):
        """Test that classifiers work with sklearn cross-validation."""
        # Test a few classifiers
        for name in ['logistic', 'random_forest', 'naive_bayes']:
            with self.subTest(classifier=name):
                clf = self.pool.build_classifier(name)
                scores = cross_val_score(clf, self.X, self.y, cv=3)
                
                self.assertEqual(len(scores), 3)
                self.assertTrue(all(0 <= score <= 1 for score in scores))


if __name__ == '__main__':
    unittest.main(verbosity=2)
