"""Unit tests for tracking modules (database and logging).

This test suite validates EnsembleDatabase and logging utilities.
"""

import unittest
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.tracking import EnsembleDatabase, setup_logger


class TestDatabaseInitialization(unittest.TestCase):
    """Test database initialization."""
    
    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
    
    def test_initialization(self):
        """Test database can be initialized."""
        database = EnsembleDatabase(db_path=self.db_path)
        database.initialize()
        
        self.assertTrue(self.db_path.exists())
    
    def test_database_size(self):
        """Test getting database size."""
        database = EnsembleDatabase(db_path=self.db_path)
        database.initialize()
        
        size_mb = database.get_size_mb()
        self.assertGreater(size_mb, 0)


class TestDatabaseOperations(unittest.TestCase):
    """Test database CRUD operations."""
    
    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.database = EnsembleDatabase(db_path=self.db_path)
        self.database.initialize()
    
    def test_insert_iteration(self):
        """Test inserting iteration record."""
        iteration_data = {
            'timestamp': datetime.now().isoformat(),
            'iteration_num': 0,
            'ensemble_id': 'test_ensemble',
            'stage1_val_auc': 0.75,
            'stage2_val_auc': 0.80,
            'diversity_score': 0.65,
            'temperature': 0.001,
            'accepted': 1,
            'rejection_reason': None,
            'num_models': 5,
            'classifier_type': 'random_forest',
            'transformers_used': 'ratio,log',
            'use_pca': 0,
            'pca_components': None,
            'pipeline_hash': 'abc123',
            'training_memory_mb': 100.0,
            'stage2_memory_mb': 50.0,
            'training_time_sec': 10.0,
            'stage2_time_sec': 5.0,
            'timeout': 0,
            'stage2_tp': 100,
            'stage2_fp': 20,
            'stage2_tn': 80,
            'stage2_fn': 30
        }
        
        self.database.insert_iteration(iteration_data)
        
        # Query to verify
        df = self.database.query_iterations(limit=1)
        self.assertEqual(len(df), 1)
    
    def test_query_iterations(self):
        """Test querying iterations."""
        # Insert multiple records
        for i in range(5):
            iteration_data = {
                'timestamp': datetime.now().isoformat(),
                'iteration_num': i,
                'ensemble_id': f'ensemble_{i}',
                'stage1_val_auc': 0.70 + i * 0.01,
                'stage2_val_auc': 0.75 + i * 0.01,
                'diversity_score': 0.60,
                'temperature': 0.001,
                'accepted': 1,
                'rejection_reason': None,
                'num_models': 5,
                'classifier_type': 'logistic',
                'transformers_used': '',
                'use_pca': 0,
                'pca_components': None,
                'pipeline_hash': f'hash_{i}',
                'training_memory_mb': 0.0,
                'stage2_memory_mb': 0.0,
                'training_time_sec': 0.0,
                'stage2_time_sec': 0.0,
                'timeout': 0,
                'stage2_tp': None,
                'stage2_fp': None,
                'stage2_tn': None,
                'stage2_fn': None
            }
            self.database.insert_iteration(iteration_data)
        
        df = self.database.query_iterations()
        self.assertEqual(len(df), 5)
        
        # Test with limit
        df_limited = self.database.query_iterations(limit=2)
        self.assertEqual(len(df_limited), 2)
    
    def test_get_accepted_ensemble_ids(self):
        """Test getting accepted ensemble IDs."""
        # Insert mixed accepted/rejected
        for i in range(5):
            iteration_data = {
                'timestamp': datetime.now().isoformat(),
                'iteration_num': i,
                'ensemble_id': f'ensemble_{i}',
                'stage1_val_auc': 0.70,
                'stage2_val_auc': 0.75,
                'diversity_score': 0.60,
                'temperature': 0.001,
                'accepted': 1 if i % 2 else 0,  # Accept odd iterations: 1, 3
                'rejection_reason': None if i % 2 else 'worse_score',
                'num_models': 5,
                'classifier_type': 'logistic',
                'transformers_used': '',
                'use_pca': 0,
                'pca_components': None,
                'pipeline_hash': f'hash_{i}',
                'training_memory_mb': 0.0,
                'stage2_memory_mb': 0.0,
                'training_time_sec': 0.0,
                'stage2_time_sec': 0.0,
                'timeout': 0,
                'stage2_tp': None,
                'stage2_fp': None,
                'stage2_tn': None,
                'stage2_fn': None
            }
            self.database.insert_iteration(iteration_data)
        
        ids = self.database.get_accepted_ensemble_ids()
        self.assertEqual(len(ids), 2)  # ensemble_1, ensemble_3


class TestStage2Logging(unittest.TestCase):
    """Test Stage 2 epoch logging."""
    
    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.database = EnsembleDatabase(db_path=self.db_path)
        self.database.initialize()
    
    def test_insert_stage2_epoch(self):
        """Test inserting Stage 2 epoch data."""
        for epoch in range(5):
            epoch_data = {
                'timestamp': datetime.now().isoformat(),
                'ensemble_id': 'test_ensemble',
                'epoch': epoch,
                'train_loss': 0.5 - epoch * 0.05,
                'val_loss': 0.6 - epoch * 0.05,
                'train_auc': 0.7 + epoch * 0.02,
                'val_auc': 0.65 + epoch * 0.02
            }
            self.database.insert_stage2_epoch(epoch_data)
        
        df = self.database.query_stage2_epochs(ensemble_id='test_ensemble')
        self.assertEqual(len(df), 5)
    
    def test_query_with_limit(self):
        """Test querying Stage 2 epochs with limit."""
        for epoch in range(10):
            epoch_data = {
                'timestamp': datetime.now().isoformat(),
                'ensemble_id': 'test_ensemble',
                'epoch': epoch,
                'train_loss': 0.5,
                'val_loss': 0.6,
                'train_auc': 0.7,
                'val_auc': 0.65
            }
            self.database.insert_stage2_epoch(epoch_data)
        
        df = self.database.query_stage2_epochs(
            ensemble_id='test_ensemble',
            limit=3
        )
        self.assertEqual(len(df), 3)


class TestBatchStatus(unittest.TestCase):
    """Test batch status tracking."""
    
    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.database = EnsembleDatabase(db_path=self.db_path)
        self.database.initialize()
    
    def test_update_worker_status(self):
        """Test updating worker status."""
        self.database.update_worker_status(
            worker_id=0,
            iteration_num=1,
            batch_num=0,
            status='running',
            start_time=datetime.now().isoformat()
        )
        
        df = self.database.get_batch_status()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['status'], 'running')
    
    def test_clear_batch_status(self):
        """Test clearing batch status."""
        # Add some workers
        for i in range(3):
            self.database.update_worker_status(
                worker_id=i,
                iteration_num=1,
                batch_num=0,
                status='running',
                start_time=datetime.now().isoformat()
            )
        
        self.database.clear_batch_status()
        
        df = self.database.get_batch_status()
        self.assertEqual(len(df), 0)


class TestLogger(unittest.TestCase):
    """Test logger setup and utilities."""
    
    def test_logger_setup(self):
        """Test logger can be set up."""
        logger = setup_logger(name='test_logger', level=logging.INFO)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertGreater(len(logger.handlers), 0)
    
    def test_no_duplicate_handlers(self):
        """Test that multiple setups don't create duplicate handlers."""
        logger1 = setup_logger(name='test_logger2', level=logging.INFO)
        initial_handlers = len(logger1.handlers)
        
        logger2 = setup_logger(name='test_logger2', level=logging.INFO)
        final_handlers = len(logger2.handlers)
        
        self.assertEqual(initial_handlers, final_handlers)


class TestIntegration(unittest.TestCase):
    """Test integration of tracking components."""
    
    def test_database_and_logging_integration(self):
        """Test database and logging work together."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / 'test.db'
        log_file = Path(temp_dir) / 'test.log'
        
        # Setup
        database = EnsembleDatabase(db_path=db_path)
        database.initialize()
        logger = setup_logger(name='integration_test', level=logging.INFO)
        
        # Log and insert data
        logger.info("Starting test")
        
        for i in range(3):
            iteration_data = {
                'timestamp': datetime.now().isoformat(),
                'iteration_num': i,
                'ensemble_id': f'ensemble_{i}',
                'stage1_val_auc': 0.70,
                'stage2_val_auc': 0.75,
                'diversity_score': 0.60,
                'temperature': 0.001,
                'accepted': 1,
                'rejection_reason': None,
                'num_models': 5,
                'classifier_type': 'logistic',
                'transformers_used': '',
                'use_pca': 0,
                'pca_components': None,
                'pipeline_hash': f'hash_{i}',
                'training_memory_mb': 0.0,
                'stage2_memory_mb': 0.0,
                'training_time_sec': 0.0,
                'stage2_time_sec': 0.0,
                'timeout': 0,
                'stage2_tp': None,
                'stage2_fp': None,
                'stage2_tn': None,
                'stage2_fn': None
            }
            database.insert_iteration(iteration_data)
            logger.info(f"Inserted iteration {i}")
        
        # Verify
        df = database.query_iterations()
        self.assertEqual(len(df), 3)
        self.assertGreater(len(logger.handlers), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
