"""Unit tests for ensemble core abstractions.

This test suite validates DataSampler, AcceptanceCriterion, and DiversityScorer.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.core import DataSampler, AcceptanceCriterion, DiversityScorer
from ensemble.config import EnsembleConfig


class TestDataSampler(unittest.TestCase):
    """Test DataSampler functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic data
        X, y = make_classification(
            n_samples=1000,
            n_features=50,
            n_informative=30,
            n_redundant=10,
            n_classes=2,
            random_state=42,
            flip_y=0.1
        )
        
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y, name='target')
        
        config = EnsembleConfig()
        self.sampler = DataSampler(
            config=config.stage1.sampling,
            random_state=42
        )
    
    def test_row_sampling(self):
        """Test row sampling preserves data types."""
        X_sample, y_sample = self.sampler.sample_rows(self.X, self.y)
        
        self.assertIsInstance(X_sample, pd.DataFrame)
        self.assertIsInstance(y_sample, pd.Series)
        self.assertLess(len(X_sample), len(self.X))
        self.assertEqual(len(X_sample), len(y_sample))
    
    def test_row_sampling_stratification(self):
        """Test that row sampling preserves class distribution."""
        original_ratio = self.y.mean()
        X_sample, y_sample = self.sampler.sample_rows(self.X, self.y)
        sample_ratio = y_sample.mean()
        
        # Should be within 5% of original ratio
        self.assertAlmostEqual(original_ratio, sample_ratio, delta=0.05)
    
    def test_column_sampling(self):
        """Test column sampling."""
        columns = self.sampler.sample_columns(self.X)
        
        self.assertIsInstance(columns, list)
        self.assertLess(len(columns), len(self.X.columns))
        self.assertTrue(all(col in self.X.columns for col in columns))
    
    def test_combined_sampling(self):
        """Test row and column sampling together."""
        X_sample, y_sample = self.sampler.sample_rows(self.X, self.y)
        columns = self.sampler.sample_columns(X_sample)
        X_final = X_sample[columns]
        
        self.assertLess(len(X_sample), len(self.X))
        self.assertLess(len(columns), len(self.X.columns))
        self.assertEqual(X_final.shape[1], len(columns))
        self.assertEqual(len(X_sample), len(y_sample))


class TestAcceptanceCriterion(unittest.TestCase):
    """Test AcceptanceCriterion (simulated annealing)."""
    
    def test_improvement_acceptance(self):
        """Test that improvements are always accepted."""
        criterion = AcceptanceCriterion(temperature=0.001, random_state=42)
        
        accepted, reason = criterion.should_accept(
            current_score=0.65,
            candidate_score=0.655
        )
        
        self.assertTrue(accepted)
        self.assertIn("Improvement", reason)
    
    def test_equal_score_acceptance(self):
        """Test that equal scores are accepted."""
        criterion = AcceptanceCriterion(temperature=0.001, random_state=42)
        
        accepted, reason = criterion.should_accept(
            current_score=0.65,
            candidate_score=0.65
        )
        
        self.assertTrue(accepted)
        self.assertIn("Equal", reason)
    
    def test_probabilistic_acceptance(self):
        """Test probabilistic acceptance for worse scores."""
        criterion = AcceptanceCriterion(temperature=0.005, random_state=42)
        
        # Run many trials to test probabilistic behavior
        acceptances = []
        for _ in range(1000):
            accepted, _ = criterion.should_accept(
                current_score=0.70,
                candidate_score=0.695
            )
            acceptances.append(accepted)
        
        # Should accept some but not all (probabilistic)
        acceptance_rate = sum(acceptances) / len(acceptances)
        self.assertGreater(acceptance_rate, 0)
        self.assertLess(acceptance_rate, 0.5)


class TestDiversityScorer(unittest.TestCase):
    """Test DiversityScorer functionality."""
    
    def setUp(self):
        """Set up test predictions."""
        np.random.seed(42)
        self.scorer = DiversityScorer()
    
    def test_high_diversity(self):
        """Test diversity score for uncorrelated predictions."""
        # Create random predictions (high diversity)
        predictions = np.random.rand(3, 100)
        score = self.scorer.score(predictions)
        
        # Random predictions should have low correlation (high diversity)
        self.assertLess(score, 0.3)
    
    def test_low_diversity(self):
        """Test diversity score for correlated predictions."""
        # Create highly correlated predictions (low diversity)
        base = np.random.rand(100)
        predictions = np.array([
            base + np.random.randn(100) * 0.01,
            base + np.random.randn(100) * 0.01,
            base + np.random.randn(100) * 0.01
        ])
        score = self.scorer.score(predictions)
        
        # Correlated predictions should have high correlation (low diversity)
        self.assertGreater(score, 0.8)
    
    def test_detailed_stats(self):
        """Test detailed statistics generation."""
        predictions = [np.random.rand(100) for _ in range(3)]
        stats = self.scorer.detailed_diversity(predictions)
        
        self.assertIn('mean_correlation', stats)
        self.assertIn('min_correlation', stats)
        self.assertIn('max_correlation', stats)
        self.assertIn('std_correlation', stats)
    
    def test_correlation_matrix(self):
        """Test correlation matrix generation."""
        predictions = np.random.rand(3, 100)
        matrix = self.scorer.correlation_matrix(predictions)
        
        self.assertEqual(matrix.shape, (3, 3))
        # Diagonal should be 1.0 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(matrix), np.ones(3))
    
    def test_diversity_check(self):
        """Test diversity threshold checking."""
        predictions = [np.random.rand(100) for _ in range(5)]
        
        diversity_score = self.scorer.score(predictions)
        is_diverse = DiversityScorer.is_diverse(diversity_score, threshold=0.7)
        self.assertIsInstance(is_diverse, bool)


class TestIntegration(unittest.TestCase):
    """Test integration of core components."""
    
    def setUp(self):
        """Set up integrated test."""
        X, y = make_classification(
            n_samples=500,
            n_features=30,
            n_classes=2,
            random_state=42
        )
        
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        
        config = EnsembleConfig()
        self.sampler = DataSampler(config.stage1.sampling, random_state=42)
        self.acceptance = AcceptanceCriterion(temperature=0.001, random_state=42)
        self.diversity = DiversityScorer()
    
    def test_complete_workflow(self):
        """Test complete workflow with all components."""
        # Sample data
        X_sample, y_sample = self.sampler.sample_rows(self.X, self.y)
        columns = self.sampler.sample_columns(X_sample)
        X_final = X_sample[columns]
        self.assertGreater(len(X_sample), 0)
        
        # Make acceptance decision
        accepted, reason = self.acceptance.should_accept(
            current_score=0.65,
            candidate_score=0.655
        )
        self.assertTrue(accepted)
        
        # Calculate diversity
        predictions = np.random.rand(3, 100)
        diversity_score = self.diversity.score(predictions)
        self.assertGreaterEqual(diversity_score, 0)
        self.assertLessEqual(diversity_score, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
