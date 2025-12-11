"""Unit tests for parallel execution modules.

This test suite validates batch scheduling and worker management.
"""

import unittest
import sys
import tempfile
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.parallel import prepare_training_batch, get_batch_info
from ensemble.tracking import EnsembleDatabase, setup_logger


class TestBatchPreparation(unittest.TestCase):
    """Test batch preparation and scheduling."""
    
    def setUp(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        self.X_train = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        self.y_train = pd.Series(y)
        self.X_val = self.X_train.iloc[:100].copy()
        self.y_val = self.y_train.iloc[:100].copy()
        
        self.preprocessor = ColumnTransformer(
            transformers=[('scaler', StandardScaler(), list(self.X_train.columns))]
        )
    
    def test_batch_structure(self):
        """Test batch job structure."""
        batch_jobs = prepare_training_batch(
            iteration=0,
            batch_size=4,
            max_iterations=100,
            X_train_pool=self.X_train,
            y_train_pool=self.y_train,
            X_val_s1=self.X_val,
            y_val_s1=self.y_val,
            base_preprocessor=self.preprocessor,
            random_state=42,
            total_cpus=8,
            timeout_minutes=5
        )
        
        self.assertEqual(len(batch_jobs), 4)
        
        # Check job structure
        for job in batch_jobs:
            self.assertEqual(len(job), 12)
            
            iteration, X_train, y_train, X_val, y_val, prep, rs, n_jobs, \
                worker_id, batch_num, timeout, clf_type = job
            
            self.assertIsInstance(iteration, (int, np.integer))
            self.assertIsInstance(X_train, pd.DataFrame)
            self.assertIsInstance(y_train, pd.Series)
            self.assertIsInstance(n_jobs, (int, np.integer))
            self.assertIsInstance(clf_type, str)
            self.assertGreaterEqual(n_jobs, 1)
    
    def test_pre_sampling(self):
        """Test pre-sampling optimization."""
        batch_jobs = prepare_training_batch(
            iteration=0,
            batch_size=3,
            max_iterations=100,
            X_train_pool=self.X_train,
            y_train_pool=self.y_train,
            X_val_s1=self.X_val,
            y_val_s1=self.y_val,
            base_preprocessor=self.preprocessor,
            random_state=42,
            row_sample_range=(0.2, 0.3)
        )
        
        for job in batch_jobs:
            X_sample = job[1]
            # Pre-sampled data should be smaller than pool
            self.assertLess(len(X_sample), len(self.X_train))


class TestCPUAllocation(unittest.TestCase):
    """Test intelligent CPU allocation."""
    
    def setUp(self):
        """Set up test data."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.preprocessor = ColumnTransformer(
            transformers=[('scaler', StandardScaler(), list(self.X.columns))]
        )
    
    def test_cpu_allocation_with_many_cpus(self):
        """Test CPU allocation when many CPUs available."""
        batch_jobs = prepare_training_batch(
            iteration=100,
            batch_size=6,
            max_iterations=200,
            X_train_pool=self.X,
            y_train_pool=self.y,
            X_val_s1=self.X.iloc[:50],
            y_val_s1=self.y.iloc[:50],
            base_preprocessor=self.preprocessor,
            random_state=123,
            total_cpus=16
        )
        
        total_cores = sum(job[7] for job in batch_jobs)
        self.assertLessEqual(total_cores, 16)
        self.assertGreater(total_cores, 6)  # Should use more than 1 per job
    
    def test_cpu_allocation_with_limited_cpus(self):
        """Test CPU allocation when CPUs limited."""
        batch_jobs = prepare_training_batch(
            iteration=100,
            batch_size=6,
            max_iterations=200,
            X_train_pool=self.X,
            y_train_pool=self.y,
            X_val_s1=self.X.iloc[:50],
            y_val_s1=self.y.iloc[:50],
            base_preprocessor=self.preprocessor,
            random_state=123,
            total_cpus=4
        )
        
        total_cores = sum(job[7] for job in batch_jobs)
        self.assertLessEqual(total_cores, 6)  # At most 1 per job


class TestBatchInfo(unittest.TestCase):
    """Test batch information summary."""
    
    def setUp(self):
        """Set up test data."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.preprocessor = ColumnTransformer(
            transformers=[('scaler', StandardScaler(), list(self.X.columns))]
        )
    
    def test_batch_info(self):
        """Test batch information generation."""
        batch_jobs = prepare_training_batch(
            iteration=0,
            batch_size=10,
            max_iterations=100,
            X_train_pool=self.X,
            y_train_pool=self.y,
            X_val_s1=self.X.iloc[:50],
            y_val_s1=self.y.iloc[:50],
            base_preprocessor=self.preprocessor,
            random_state=42
        )
        
        info = get_batch_info(batch_jobs)
        
        self.assertIn('n_jobs', info)
        self.assertIn('total_cores', info)
        self.assertIn('avg_cores_per_job', info)
        self.assertIn('classifier_counts', info)
        
        self.assertEqual(info['n_jobs'], len(batch_jobs))
        self.assertGreater(info['total_cores'], 0)
        self.assertIsInstance(info['classifier_counts'], dict)
    
    def test_empty_batch_info(self):
        """Test batch info with empty batch."""
        info = get_batch_info([])
        
        self.assertEqual(info['n_jobs'], 0)
        self.assertEqual(info['total_cores'], 0)


class TestIntegration(unittest.TestCase):
    """Test integration of parallel components."""
    
    def setUp(self):
        """Set up integrated test."""
        X, y = make_classification(n_samples=200, n_features=8, random_state=42)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.preprocessor = ColumnTransformer(
            transformers=[('scaler', StandardScaler(), list(self.X.columns))]
        )
        
        temp_dir = tempfile.mkdtemp()
        self.db_path = Path(temp_dir) / 'test.db'
        self.log_file = Path(temp_dir) / 'test.log'
    
    def test_batch_preparation_with_database(self):
        """Test batch preparation with database tracking."""
        # Setup database and logger
        database = EnsembleDatabase(db_path=self.db_path)
        database.initialize()
        logger = setup_logger(name='test_parallel', level=logging.INFO)
        
        # Prepare batch
        batch_jobs = prepare_training_batch(
            iteration=0,
            batch_size=2,
            max_iterations=100,
            X_train_pool=self.X,
            y_train_pool=self.y,
            X_val_s1=self.X.iloc[:50],
            y_val_s1=self.y.iloc[:50],
            base_preprocessor=self.preprocessor,
            random_state=42
        )
        
        # Get info
        info = get_batch_info(batch_jobs)
        
        logger.info(f"Prepared batch with {info['n_jobs']} jobs")
        
        self.assertEqual(len(batch_jobs), 2)
        self.assertTrue(self.db_path.exists())
        self.assertGreater(len(logger.handlers), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
