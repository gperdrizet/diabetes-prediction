"""
Parallel training utilities for ensemble hill climbing.

Handles batch training of candidate models and helper functions.
"""

import time
import psutil
import os
import signal
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .ensemble_hill_climbing import generate_random_pipeline, compute_pipeline_hash


class TimeoutError(Exception):
    """Exception raised when training exceeds timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Training exceeded 5 minute timeout")


def train_single_candidate(args):
    """
    Train a single candidate pipeline in a separate process.
    
    Parameters
    ----------
    args : tuple
        (iteration, X_train_pool, y_train_pool, X_val_s1, y_val_s1, base_preprocessor, random_state)
    
    Returns
    -------
    dict : Dictionary containing:
        - iteration: iteration number
        - fitted_pipeline: trained pipeline
        - metadata: pipeline configuration
        - val_auc_s1: stage 1 validation AUC
        - pipeline_hash: unique pipeline hash
        - training_time: time to train (seconds)
    """
    iteration, X_train_pool, y_train_pool, X_val_s1, y_val_s1, base_preprocessor, random_state = args
    
    # Set 5 minute timeout for training
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minutes = 300 seconds
    
    try:
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 ** 2)  # MB
        
        # Generate random pipeline (includes adaptive row_sample_pct in metadata)
        pipeline, metadata = generate_random_pipeline(
            iteration=iteration,
            random_state=random_state,
            base_preprocessor=base_preprocessor
        )
        
        # Apply adaptive row sampling based on metadata
        row_sample_pct = metadata['row_sample_pct']
        n_total = len(X_train_pool)
        n_sample = max(100, int(n_total * row_sample_pct))  # At least 100 samples
        
        # Sample from training pool
        X_train, _, y_train, _ = train_test_split(
            X_train_pool,
            y_train_pool,
            train_size=n_sample,
            stratify=y_train_pool,
            random_state=random_state
        )
        
        # Train pipeline
        fitted_pipeline = pipeline.fit(X_train, y_train)
        
        # Track peak memory
        peak_memory = process.memory_info().rss / (1024 ** 2)  # MB
        memory_used = peak_memory - start_memory
        
        # Evaluate on stage 1 validation
        if hasattr(fitted_pipeline, 'predict_proba'):
            val_pred_s1 = fitted_pipeline.predict_proba(X_val_s1)[:, 1]
        else:
            val_pred_s1 = fitted_pipeline.decision_function(X_val_s1)
        
        val_auc_s1 = roc_auc_score(y_val_s1, val_pred_s1)
        
        # Compute hash
        pipeline_hash = compute_pipeline_hash(fitted_pipeline, metadata)
        
        training_time = time.time() - start_time
        
        # Cancel the alarm - training completed successfully
        signal.alarm(0)
        
        return {
            'iteration': iteration,
            'fitted_pipeline': fitted_pipeline,
            'metadata': metadata,
            'val_auc_s1': val_auc_s1,
            'pipeline_hash': pipeline_hash,
            'training_time': training_time,
            'memory_mb': memory_used,
            'training_time_sec': training_time
        }
    
    except TimeoutError as e:
        # Cancel the alarm
        signal.alarm(0)
        # Re-raise with classifier info
        raise TimeoutError(f"Training exceeded 5 minute timeout (classifier: {metadata.get('classifier_type', 'unknown')})")
    
    except Exception as e:
        # Cancel the alarm on any other error
        signal.alarm(0)
        raise


def prepare_training_batch(iteration, batch_size, max_iterations, X_train_pool, y_train_pool,
                           X_val_s1, y_val_s1, base_preprocessor, random_state):
    """
    Prepare a batch of training jobs for parallel execution.
    
    Parameters
    ----------
    iteration : int
        Current iteration number
    batch_size : int
        Number of candidates to train in parallel
    max_iterations : int
        Maximum total iterations
    X_train_pool, y_train_pool : arrays
        Training pool data
    X_val_s1, y_val_s1 : arrays
        Stage 1 validation data
    base_preprocessor : ColumnTransformer
        Base preprocessor for features
    random_state : int
        Base random state
    
    Returns
    -------
    list
        List of tuples for parallel training
    """
    batch_jobs = []
    for i in range(batch_size):
        current_iter = iteration + i
        if current_iter >= max_iterations:
            break
        
        # Pass full training pool - adaptive sampling will be done in train_single_candidate
        # based on classifier complexity (2.5-27.5% of data)
        batch_jobs.append((
            current_iter,
            X_train_pool,
            y_train_pool,
            X_val_s1,
            y_val_s1,
            base_preprocessor,
            random_state + current_iter
        ))
    
    return batch_jobs
