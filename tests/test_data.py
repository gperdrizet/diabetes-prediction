"""Unit tests for data management modules.

This test suite validates DataSplits and preprocessing utilities.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.data import DataSplits, create_three_way_split, create_base_preprocessor


class TestDataSplits(unittest.TestCase):
    """Test DataSplits functionality."""
    
    def setUp(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=1000,
            n_features=8,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            weights=[0.7, 0.3],
            random_state=42,
            flip_y=0.1
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.df = pd.DataFrame(X, columns=feature_names)
        self.df['target'] = y
    
    def test_initialization(self):
        """Test DataSplits initialization."""
        splits = DataSplits(
            df=self.df,
            target_column='target',
            random_state=42
        )
        
        self.assertIsNotNone(splits)
    
    def test_split_sizes(self):
        """Test that splits have correct proportions."""
        splits = DataSplits(
            df=self.df,
            target_column='target',
            random_state=42
        )
        
        X_train, y_train = splits.get_train_pool()
        X_val_s1, y_val_s1 = splits.get_val_stage1()
        X_val_s2, y_val_s2 = splits.get_val_stage2()
        
        total = len(X_train) + len(X_val_s1) + len(X_val_s2)
        
        # Should be approximately 60/35/5
        self.assertAlmostEqual(len(X_train) / total, 0.60, delta=0.02)
        self.assertAlmostEqual(len(X_val_s1) / total, 0.35, delta=0.02)
        self.assertAlmostEqual(len(X_val_s2) / total, 0.05, delta=0.02)
    
    def test_stratification(self):
        """Test that stratification preserves class distribution."""
        splits = DataSplits(
            df=self.df,
            target_column='target',
            random_state=42
        )
        
        full_positive_rate = self.df['target'].mean()
        
        _, y_train = splits.get_train_pool()
        _, y_val_s1 = splits.get_val_stage1()
        _, y_val_s2 = splits.get_val_stage2()
        
        # All splits should have similar positive rates
        train_rate = y_train.mean()
        val_s1_rate = y_val_s1.mean()
        val_s2_rate = y_val_s2.mean()
        
        self.assertAlmostEqual(full_positive_rate, train_rate, delta=0.05)
        self.assertAlmostEqual(full_positive_rate, val_s1_rate, delta=0.05)
        self.assertAlmostEqual(full_positive_rate, val_s2_rate, delta=0.10)  # Larger delta for small set
    
    def test_summary(self):
        """Test summary generation."""
        splits = DataSplits(
            df=self.df,
            target_column='target',
            random_state=42
        )
        
        summary = splits.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("Data Splits Summary", summary)
        self.assertIn("Training pool", summary)


class TestConvenienceFunction(unittest.TestCase):
    """Test create_three_way_split convenience function."""
    
    def setUp(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=500,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        
        self.df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        self.df['target'] = y
    
    def test_convenience_function(self):
        """Test convenience function creates valid splits."""
        splits = create_three_way_split(
            df=self.df,
            target_column='target',
            random_state=42
        )
        
        self.assertIsInstance(splits, DataSplits)
        
        X_train, y_train = splits.get_train_pool()
        self.assertGreater(len(X_train), 0)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing utilities."""
    
    def setUp(self):
        """Set up test data with different feature types."""
        np.random.seed(42)
        
        # Numerical features
        numerical = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'height': np.random.uniform(150, 200, 100),
            'weight': np.random.uniform(50, 100, 100),
            'bmi': np.random.uniform(18, 35, 100)
        })
        
        # Ordinal features
        ordinal = pd.DataFrame({
            'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], 100),
            'income': np.random.choice(['Low', 'Medium', 'High'], 100)
        })
        
        # Nominal features
        nominal = pd.DataFrame({
            'gender': np.random.choice(['M', 'F'], 100),
            'smoking_status': np.random.choice(['Never', 'Former', 'Current'], 100)
        })
        
        self.X = pd.concat([numerical, ordinal, nominal], axis=1)
    
    def test_create_base_preprocessor(self):
        """Test base preprocessor creation."""
        preprocessor = create_base_preprocessor(self.X)
        
        self.assertIsNotNone(preprocessor)
        self.assertTrue(hasattr(preprocessor, 'fit'))
        self.assertTrue(hasattr(preprocessor, 'transform'))
    
    def test_preprocessor_fit_transform(self):
        """Test preprocessor can fit and transform."""
        preprocessor = create_base_preprocessor(self.X)
        
        X_transformed = preprocessor.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertGreater(X_transformed.shape[1], 0)


class TestIntegration(unittest.TestCase):
    """Test integration of data management components."""
    
    def setUp(self):
        """Set up integrated test."""
        X, y = make_classification(
            n_samples=800,
            n_features=6,
            n_classes=2,
            random_state=42
        )
        
        self.df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        self.df['target'] = y
    
    def test_complete_workflow(self):
        """Test complete data management workflow."""
        # Create splits
        splits = create_three_way_split(
            df=self.df,
            target_column='target',
            random_state=42
        )
        
        # Get training data
        X_train, y_train = splits.get_train_pool()
        self.assertGreater(len(X_train), 0)
        
        # Create preprocessor
        preprocessor = create_base_preprocessor(X_train)
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X_train)
        self.assertEqual(len(X_transformed), len(X_train))
        
        # Transform validation data
        X_val, y_val = splits.get_val_stage1()
        X_val_transformed = preprocessor.transform(X_val)
        self.assertEqual(len(X_val_transformed), len(X_val))


if __name__ == '__main__':
    unittest.main(verbosity=2)
