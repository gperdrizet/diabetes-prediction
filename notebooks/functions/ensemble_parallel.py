"""
Parallel training utilities for ensemble hill climbing.

Handles batch training of candidate models and helper functions.
"""

import time
import psutil
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .ensemble_hill_climbing import generate_random_pipeline, compute_pipeline_hash


def train_single_candidate(args):
    """
    Train a single candidate pipeline in a separate process.
    
    Parameters
    ----------
    args : tuple
        (iteration, X_train, y_train, X_val_s1, y_val_s1, base_preprocessor, random_state)
    
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
    iteration, X_train, y_train, X_val_s1, y_val_s1, base_preprocessor, random_state = args
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 ** 2)  # MB
    
    # Generate random pipeline
    pipeline, metadata = generate_random_pipeline(
        iteration=iteration,
        random_state=random_state,
        base_preprocessor=base_preprocessor
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
        
        # Random sample size for this iteration
        rng = np.random.RandomState(random_state + current_iter)
        iteration_sample_size = rng.randint(10000, 50001)
        
        # Sample from training pool
        X_train, _, y_train, _ = train_test_split(
            X_train_pool,
            y_train_pool,
            train_size=iteration_sample_size,
            stratify=y_train_pool
        )
        
        batch_jobs.append((
            current_iter,
            X_train,
            y_train,
            X_val_s1,
            y_val_s1,
            base_preprocessor,
            random_state + current_iter
        ))
    
    return batch_jobs
