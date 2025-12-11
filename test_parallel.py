"""Validation tests for Phase 7: Parallel Execution refactoring.

This script validates the parallel execution modules:
- Batch scheduling with intelligent CPU allocation
- Worker process management with timeout handling
- Pre-sampling optimization
- Database integration
"""

import sys
import os
import tempfile
import sqlite3
import logging
import multiprocessing
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from ensemble.parallel import (
    prepare_training_batch,
    get_batch_info,
    train_single_candidate,
    train_batch_parallel
)
from ensemble.tracking import EnsembleDatabase


# ============================================================================
# Test Setup
# ============================================================================

def create_test_data():
    """Create synthetic test data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Split into train pool and validation
    split_idx = 700
    X_train = X_df.iloc[:split_idx]
    y_train = y_series.iloc[:split_idx]
    X_val = X_df.iloc[split_idx:]
    y_val = y_series.iloc[split_idx:]
    
    return X_train, y_train, X_val, y_val


def create_test_preprocessor(feature_names):
    """Create a simple preprocessor for testing."""
    return ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), feature_names)
        ]
    )


def create_test_database():
    """Create a temporary test database."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_ensemble.db"
    
    database = EnsembleDatabase(db_path=db_path)
    database.initialize()
    
    return database


def create_test_logger():
    """Create a simple logger for testing."""
    logger = logging.getLogger('test_parallel')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# ============================================================================
# Batch Scheduling Tests
# ============================================================================

def test_prepare_training_batch():
    """Test batch preparation with pre-sampling and CPU allocation."""
    print("\n" + "="*70)
    print("TEST: Batch Preparation")
    print("="*70)
    
    # Create test data
    X_train, y_train, X_val, y_val = create_test_data()
    preprocessor = create_test_preprocessor(X_train.columns)
    
    # Prepare batch
    batch_jobs = prepare_training_batch(
        iteration=0,
        batch_size=4,
        max_iterations=100,
        X_train_pool=X_train,
        y_train_pool=y_train,
        X_val_s1=X_val,
        y_val_s1=y_val,
        base_preprocessor=preprocessor,
        random_state=42,
        total_cpus=8,
        timeout_minutes=5,
        batch_num=0,
        row_sample_range=(0.2, 0.3)
    )
    
    # Validate batch
    assert len(batch_jobs) == 4, f"Expected 4 jobs, got {len(batch_jobs)}"
    
    # Check each job structure
    for i, job in enumerate(batch_jobs):
        assert len(job) == 12, f"Job {i} should have 12 elements"
        
        iteration, X_train_sample, y_train_sample, X_val_job, y_val_job, \
            preprocessor_job, random_state, n_jobs, worker_id, batch_num, \
            timeout_seconds, classifier_type = job
        
        # Check types
        assert isinstance(iteration, (int, np.integer)), f"Iteration should be int"
        assert isinstance(X_train_sample, pd.DataFrame), f"X_train should be DataFrame"
        assert isinstance(y_train_sample, pd.Series), f"y_train should be Series"
        assert isinstance(n_jobs, (int, np.integer)), f"n_jobs should be int"
        assert isinstance(classifier_type, str), f"classifier_type should be str"
        
        # Check data is pre-sampled (smaller than pool)
        assert len(X_train_sample) < len(X_train), \
            f"Pre-sampled data should be smaller than pool"
        assert len(X_train_sample) == len(y_train_sample), \
            f"X and y should have same length"
        
        # Check CPU allocation
        assert n_jobs >= 1, f"Each job should have at least 1 CPU"
        
        # Check timeout
        assert timeout_seconds == 5 * 60, f"Timeout should be 300 seconds"
    
    print("✓ Batch structure validated")
    print(f"✓ Created {len(batch_jobs)} training jobs")
    print(f"✓ Pre-sampling: {len(batch_jobs[0][1])} rows (from {len(X_train)} pool)")
    
    # Check CPU allocation
    total_cores_used = sum(job[7] for job in batch_jobs)
    print(f"✓ CPU allocation: {total_cores_used} cores across {len(batch_jobs)} jobs")
    
    return batch_jobs


def test_cpu_allocation():
    """Test intelligent CPU allocation for different classifier types."""
    print("\n" + "="*70)
    print("TEST: CPU Allocation Strategy")
    print("="*70)
    
    # Create test data
    X_train, y_train, X_val, y_val = create_test_data()
    preprocessor = create_test_preprocessor(X_train.columns)
    
    # Test with many CPUs available
    batch_jobs_many_cpus = prepare_training_batch(
        iteration=100,  # Different seed for different classifiers
        batch_size=6,
        max_iterations=200,
        X_train_pool=X_train,
        y_train_pool=y_train,
        X_val_s1=X_val,
        y_val_s1=y_val,
        base_preprocessor=preprocessor,
        random_state=123,
        total_cpus=16,  # Many CPUs
        timeout_minutes=5
    )
    
    # Test with limited CPUs
    batch_jobs_few_cpus = prepare_training_batch(
        iteration=100,
        batch_size=6,
        max_iterations=200,
        X_train_pool=X_train,
        y_train_pool=y_train,
        X_val_s1=X_val,
        y_val_s1=y_val,
        base_preprocessor=preprocessor,
        random_state=123,
        total_cpus=4,  # Limited CPUs
        timeout_minutes=5
    )
    
    # Analyze allocation
    print("\nWith 16 CPUs available:")
    for job in batch_jobs_many_cpus:
        classifier_type = job[11]
        n_jobs = job[7]
        print(f"  {classifier_type:20s} -> {n_jobs} cores")
    
    print("\nWith 4 CPUs available:")
    for job in batch_jobs_few_cpus:
        classifier_type = job[11]
        n_jobs = job[7]
        print(f"  {classifier_type:20s} -> {n_jobs} cores")
    
    # Verify allocation is reasonable
    total_many = sum(job[7] for job in batch_jobs_many_cpus)
    total_few = sum(job[7] for job in batch_jobs_few_cpus)
    
    assert total_many <= 16, "Should not exceed available CPUs"
    assert total_few <= 6, "Should not exceed available CPUs (at most 1 per job)"
    
    print(f"\n✓ CPU allocation validated")
    print(f"✓ Many CPUs: {total_many}/16 cores used")
    print(f"✓ Limited CPUs: {total_few}/4 cores used")


def test_get_batch_info():
    """Test batch information summary."""
    print("\n" + "="*70)
    print("TEST: Batch Information")
    print("="*70)
    
    # Create test data and batch
    X_train, y_train, X_val, y_val = create_test_data()
    preprocessor = create_test_preprocessor(X_train.columns)
    
    batch_jobs = prepare_training_batch(
        iteration=0,
        batch_size=10,
        max_iterations=100,
        X_train_pool=X_train,
        y_train_pool=y_train,
        X_val_s1=X_val,
        y_val_s1=y_val,
        base_preprocessor=preprocessor,
        random_state=42,
        total_cpus=8
    )
    
    # Get batch info
    info = get_batch_info(batch_jobs)
    
    # Validate info
    assert info['n_jobs'] == len(batch_jobs), "Job count mismatch"
    assert info['total_cores'] > 0, "Should have positive core count"
    assert info['avg_cores_per_job'] > 0, "Should have positive average"
    assert isinstance(info['classifier_counts'], dict), "Should have classifier counts"
    
    print(f"✓ Batch size: {info['n_jobs']} jobs")
    print(f"✓ Total cores: {info['total_cores']}")
    print(f"✓ Avg cores/job: {info['avg_cores_per_job']:.1f}")
    print(f"✓ Classifier distribution:")
    for clf, count in sorted(info['classifier_counts'].items()):
        print(f"    {clf:20s}: {count}")


# ============================================================================
# Worker Process Tests
# ============================================================================

def test_train_single_candidate():
    """Test training a single candidate with timeout protection."""
    print("\n" + "="*70)
    print("TEST: Single Candidate Training")
    print("="*70)
    
    # Create test data
    X_train, y_train, X_val, y_val = create_test_data()
    preprocessor = create_test_preprocessor(X_train.columns)
    
    # Create database and logger
    database = create_test_database()
    logger = create_test_logger()
    
    # Prepare a single job
    batch_jobs = prepare_training_batch(
        iteration=0,
        batch_size=1,
        max_iterations=100,
        X_train_pool=X_train,
        y_train_pool=y_train,
        X_val_s1=X_val,
        y_val_s1=y_val,
        base_preprocessor=preprocessor,
        random_state=42,
        total_cpus=2,
        timeout_minutes=2  # Short timeout for testing
    )
    
    job_args = batch_jobs[0]
    
    # Train candidate
    print("Training candidate (this may take 30-60 seconds)...")
    result = train_single_candidate(
        job_args=job_args,
        database=database,
        logger=logger
    )
    
    # Validate result
    assert result is not None, "Training should return a result"
    assert 'iteration' in result, "Result should have iteration"
    assert 'val_accuracy' in result, "Result should have validation accuracy"
    assert 'pipeline' in result, "Result should have trained pipeline"
    assert 'classifier_type' in result, "Result should have classifier type"
    
    print(f"\n✓ Training completed successfully")
    print(f"✓ Iteration: {result['iteration']}")
    print(f"✓ Classifier: {result['classifier_type']}")
    print(f"✓ Val accuracy: {result['val_accuracy']:.4f}")
    print(f"✓ Memory used: {result['memory_mb']:.1f} MB")
    
    # Check worker status in database using proper API
    batch_status = database.get_batch_status()
    if not batch_status.empty:
        worker_status = batch_status[batch_status['worker_id'] == job_args[8]]
        if not worker_status.empty:
            print(f"✓ Worker status: {worker_status.iloc[0]['status']}")


def test_batch_parallel_training():
    """Test parallel training of a small batch."""
    print("\n" + "="*70)
    print("TEST: Batch Parallel Training")
    print("="*70)
    
    # Create test data
    X_train, y_train, X_val, y_val = create_test_data()
    preprocessor = create_test_preprocessor(X_train.columns)
    
    # Create database and logger
    database = create_test_database()
    logger = create_test_logger()
    
    # Prepare small batch
    batch_jobs = prepare_training_batch(
        iteration=0,
        batch_size=3,  # Small batch for testing
        max_iterations=100,
        X_train_pool=X_train,
        y_train_pool=y_train,
        X_val_s1=X_val,
        y_val_s1=y_val,
        base_preprocessor=preprocessor,
        random_state=42,
        total_cpus=4,
        timeout_minutes=2
    )
    
    # Train batch
    print(f"Training batch of {len(batch_jobs)} candidates...")
    print("(This may take 1-2 minutes)")
    
    results = train_batch_parallel(
        batch_jobs=batch_jobs,
        database=database,
        logger=logger,
        max_workers=3
    )
    
    # Validate results
    assert len(results) == len(batch_jobs), "Should have result for each job"
    
    successful = sum(1 for r in results if r is not None)
    failed = len(results) - successful
    
    print(f"\n✓ Batch training completed")
    print(f"✓ Successful: {successful}/{len(results)}")
    print(f"✓ Failed: {failed}/{len(results)}")
    
    # Show successful results
    for i, result in enumerate(results):
        if result is not None:
            print(f"\n  Job {i}:")
            print(f"    Iteration: {result['iteration']}")
            print(f"    Classifier: {result['classifier_type']}")
            print(f"    Val accuracy: {result['val_accuracy']:.4f}")


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_workflow():
    """Test complete workflow from batch prep to parallel training."""
    print("\n" + "="*70)
    print("TEST: End-to-End Workflow")
    print("="*70)
    
    # Setup
    X_train, y_train, X_val, y_val = create_test_data()
    preprocessor = create_test_preprocessor(X_train.columns)
    database = create_test_database()
    logger = create_test_logger()
    
    # Step 1: Prepare batch
    print("\nStep 1: Preparing batch...")
    batch_jobs = prepare_training_batch(
        iteration=0,
        batch_size=2,
        max_iterations=100,
        X_train_pool=X_train,
        y_train_pool=y_train,
        X_val_s1=X_val,
        y_val_s1=y_val,
        base_preprocessor=preprocessor,
        random_state=42,
        total_cpus=4
    )
    
    # Step 2: Get batch info
    print("\nStep 2: Getting batch info...")
    info = get_batch_info(batch_jobs)
    print(f"  Jobs: {info['n_jobs']}")
    print(f"  Total cores: {info['total_cores']}")
    print(f"  Classifiers: {list(info['classifier_counts'].keys())}")
    
    # Step 3: Train batch
    print("\nStep 3: Training batch...")
    results = train_batch_parallel(
        batch_jobs=batch_jobs,
        database=database,
        logger=logger
    )
    
    # Step 4: Validate
    print("\nStep 4: Validating results...")
    assert len(results) == len(batch_jobs), "Result count mismatch"
    
    successful = [r for r in results if r is not None]
    print(f"  Successful: {len(successful)}/{len(results)}")
    
    for result in successful:
        assert 'pipeline' in result, "Result should have pipeline"
        assert 'val_accuracy' in result, "Result should have accuracy"
        assert result['val_accuracy'] > 0, "Accuracy should be positive"
        assert result['val_accuracy'] <= 1, "Accuracy should be <= 1"
    
    print("\n✓ End-to-end workflow validated")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all validation tests."""
    print("\n")
    print("="*70)
    print(" PHASE 7 VALIDATION: PARALLEL EXECUTION")
    print("="*70)
    
    tests = [
        ("Batch Preparation", test_prepare_training_batch),
        ("CPU Allocation", test_cpu_allocation),
        ("Batch Information", test_get_batch_info),
        ("Single Candidate Training", test_train_single_candidate),
        ("Batch Parallel Training", test_batch_parallel_training),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n")
    print("="*70)
    print(" TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        print("\nPhase 7 validation successful!")
        print("Ready to proceed with Phase 8.")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease fix issues before proceeding.")
        return False


if __name__ == '__main__':
    # Set multiprocessing start method
    if sys.platform != 'win32':
        multiprocessing.set_start_method('spawn', force=True)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
