"""Test script for Phase 6: Tracking Simplification.

Tests database operations and logging utilities.
"""

import tempfile
import logging
from pathlib import Path

import pandas as pd

from ensemble.tracking.database import EnsembleDatabase
from ensemble.tracking.logger import (
    setup_logger,
    log_phase_start,
    log_phase_end,
    log_iteration,
    log_training_progress,
    log_performance_metrics,
    log_error,
    log_warning,
    log_success
)


def test_database_initialization():
    """Test database initialization."""
    print("Testing database initialization...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = EnsembleDatabase(db_path)
        
        # Test that file doesn't exist yet
        assert not db.exists(), "Database should not exist initially"
        
        # Initialize
        db.initialize()
        
        # Check that file was created
        assert db.exists(), "Database should exist after initialization"
        print(f"  ✓ Database created at {db_path}")
        
        # Check size
        size_mb = db.get_size_mb()
        print(f"  ✓ Database size: {size_mb:.3f} MB")


def test_database_operations():
    """Test database CRUD operations."""
    print("\nTesting database operations...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = EnsembleDatabase(db_path)
        db.initialize()
        
        # Insert iteration data
        iteration_data = {
            'timestamp': '2025-12-10T10:00:00',
            'iteration_num': 1,
            'ensemble_id': 'test_ensemble_1',
            'stage1_val_auc': 0.75,
            'stage2_val_auc': 0.80,
            'diversity_score': 0.65,
            'temperature': 0.001,
            'accepted': 1,
            'rejection_reason': 'better_score',
            'num_models': 5,
            'classifier_type': 'random_forest',
            'transformers_used': 'ratio,product',
            'pipeline_hash': 'abc123'
        }
        
        db.insert_iteration(iteration_data)
        print("  ✓ Inserted iteration record")
        
        # Query iterations
        df = db.query_iterations()
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        assert df.iloc[0]['ensemble_id'] == 'test_ensemble_1', "Ensemble ID mismatch"
        assert df.iloc[0]['stage1_val_auc'] == 0.75, "Stage 1 AUC mismatch"
        print(f"  ✓ Queried iterations: {len(df)} records")
        
        # Insert multiple iterations
        for i in range(2, 6):
            iteration_data['iteration_num'] = i
            iteration_data['ensemble_id'] = f'test_ensemble_{i}'
            iteration_data['accepted'] = 1 if i % 2 == 0 else 0
            db.insert_iteration(iteration_data)
        
        df = db.query_iterations()
        assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
        print(f"  ✓ Inserted multiple iterations: {len(df)} total")
        
        # Test limit
        df_limited = db.query_iterations(limit=3)
        assert len(df_limited) == 3, f"Expected 3 rows with limit, got {len(df_limited)}"
        print("  ✓ Query with limit works")
        
        # Get accepted ensemble IDs
        ensemble_ids = db.get_accepted_ensemble_ids()
        assert len(ensemble_ids) > 0, "Should have accepted ensembles"
        print(f"  ✓ Got accepted ensemble IDs: {ensemble_ids}")


def test_stage2_logging():
    """Test Stage 2 epoch logging."""
    print("\nTesting Stage 2 epoch logging...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = EnsembleDatabase(db_path)
        db.initialize()
        
        # Insert Stage 2 epochs
        for epoch in range(5):
            epoch_data = {
                'timestamp': f'2025-12-10T10:{epoch:02d}:00',
                'ensemble_id': 'test_ensemble_1',
                'epoch': epoch,
                'train_loss': 0.5 - epoch * 0.05,
                'val_loss': 0.6 - epoch * 0.04,
                'train_auc': 0.70 + epoch * 0.02,
                'val_auc': 0.68 + epoch * 0.02
            }
            db.insert_stage2_epoch(epoch_data)
        
        print("  ✓ Inserted 5 Stage 2 epochs")
        
        # Query Stage 2 data
        df = db.query_stage2_epochs('test_ensemble_1')
        assert len(df) == 5, f"Expected 5 epochs, got {len(df)}"
        print(f"  ✓ Queried Stage 2 epochs: {len(df)} records")
        
        # Test limit
        df_limited = db.query_stage2_epochs('test_ensemble_1', limit=3)
        assert len(df_limited) == 3, f"Expected 3 epochs with limit, got {len(df_limited)}"
        print("  ✓ Query with limit works")


def test_batch_status():
    """Test batch status tracking."""
    print("\nTesting batch status tracking...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = EnsembleDatabase(db_path)
        db.initialize()
        
        # Update worker status
        db.update_worker_status(
            worker_id=0,
            iteration_num=1,
            batch_num=1,
            status='running',
            classifier_type='logistic',
            start_time='2025-12-10T10:00:00'
        )
        print("  ✓ Updated worker 0 status to 'running'")
        
        # Get batch status
        df = db.get_batch_status()
        assert len(df) == 1, f"Expected 1 worker, got {len(df)}"
        assert df.iloc[0]['status'] == 'running', "Status mismatch"
        print(f"  ✓ Got batch status: {len(df)} workers")
        
        # Update same worker to completed
        db.update_worker_status(
            worker_id=0,
            iteration_num=1,
            batch_num=1,
            status='completed',
            end_time='2025-12-10T10:05:00',
            runtime_sec=300.0
        )
        
        df = db.get_batch_status()
        assert df.iloc[0]['status'] == 'completed', "Status not updated"
        assert df.iloc[0]['runtime_sec'] == 300.0, "Runtime mismatch"
        print("  ✓ Updated worker 0 status to 'completed'")
        
        # Clear batch status
        db.clear_batch_status()
        df = db.get_batch_status()
        assert len(df) == 0, f"Expected 0 workers after clear, got {len(df)}"
        print("  ✓ Cleared batch status")


def test_logger_setup():
    """Test logger setup and configuration."""
    print("\nTesting logger setup...")
    
    logger = setup_logger('test_logger', level=logging.INFO)
    assert logger.name == 'test_logger', "Logger name mismatch"
    assert logger.level == logging.INFO, "Logger level mismatch"
    assert len(logger.handlers) > 0, "Logger has no handlers"
    print("  ✓ Logger setup successful")
    
    # Test that multiple calls don't create duplicate handlers
    logger2 = setup_logger('test_logger', level=logging.INFO)
    assert len(logger2.handlers) == len(logger.handlers), "Duplicate handlers created"
    print("  ✓ No duplicate handlers created")


def test_logging_functions():
    """Test logging utility functions."""
    print("\nTesting logging functions...")
    
    logger = setup_logger('test_logger', level=logging.INFO)
    
    # Test phase logging
    print("\n  Testing phase logging:")
    log_phase_start(logger, "Test Phase", "This is a test")
    log_phase_end(logger, "Test Phase", elapsed_time=10.5)
    print("  ✓ Phase logging works")
    
    # Test iteration logging
    print("\n  Testing iteration logging:")
    log_iteration(
        logger,
        iteration=1,
        accepted=True,
        reason="better_score",
        metrics={'stage1_auc': 0.75, 'stage2_auc': 0.80, 'diversity': 0.65}
    )
    print("  ✓ Iteration logging works")
    
    # Test progress logging
    print("\n  Testing progress logging:")
    log_training_progress(logger, current=50, total=100)
    print("  ✓ Progress logging works")
    
    # Test metrics logging
    print("\n  Testing metrics logging:")
    log_performance_metrics(
        logger,
        metrics={'accuracy': 0.85, 'precision': 0.80, 'recall': 0.90},
        prefix="Model Performance"
    )
    print("  ✓ Metrics logging works")
    
    # Test error logging
    print("\n  Testing error logging:")
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_error(logger, e, context="test function")
    print("  ✓ Error logging works")
    
    # Test warning and success
    print("\n  Testing warning and success logging:")
    log_warning(logger, "This is a warning")
    log_success(logger, "Operation completed successfully")
    print("  ✓ Warning and success logging works")


def test_integration():
    """Test integration of database and logging."""
    print("\nTesting integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = EnsembleDatabase(db_path)
        logger = setup_logger('integration_test')
        
        # Initialize database
        log_phase_start(logger, "Database Initialization")
        db.initialize()
        log_success(logger, f"Database created at {db_path}")
        
        # Insert data
        log_phase_start(logger, "Data Insertion")
        for i in range(3):
            iteration_data = {
                'timestamp': f'2025-12-10T10:{i:02d}:00',
                'iteration_num': i,
                'ensemble_id': f'ensemble_{i}',
                'stage1_val_auc': 0.70 + i * 0.05,
                'stage2_val_auc': 0.75 + i * 0.05,
                'diversity_score': 0.60,
                'temperature': 0.001,
                'accepted': 1,
                'num_models': i + 1,
                'pipeline_hash': f'hash_{i}'
            }
            db.insert_iteration(iteration_data)
            log_success(logger, f"Inserted iteration {i}")
        
        # Query data
        df = db.query_iterations()
        log_performance_metrics(
            logger,
            {'total_iterations': len(df), 'accepted': df['accepted'].sum()},
            prefix="Database Statistics"
        )
        
        log_phase_end(logger, "Integration Test", elapsed_time=1.0)
        print("  ✓ Integration successful!")


if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 6: TRACKING SIMPLIFICATION VALIDATION")
    print("=" * 80)
    
    try:
        test_database_initialization()
        test_database_operations()
        test_stage2_logging()
        test_batch_status()
        test_logger_setup()
        test_logging_functions()
        test_integration()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nTracking modules are working correctly.")
        print("Components validated:")
        print("  - Database initialization and operations")
        print("  - Iteration and Stage 2 epoch logging")
        print("  - Batch status tracking")
        print("  - Structured logger setup")
        print("  - Logging utility functions")
        print("  - Integration of database and logging")
        print("\nReady to proceed with Phase 7 (Parallel Execution).")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
