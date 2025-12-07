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
    
    NOTE: Training data is pre-sampled in main process to minimize serialization overhead.
    
    Parameters
    ----------
    args : tuple
        (iteration, X_train_sample, y_train_sample, X_val_s1, y_val_s1, base_preprocessor, random_state, n_jobs)
    
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
    iteration, X_train_sample, y_train_sample, X_val_s1, y_val_s1, base_preprocessor, random_state, n_jobs = args
    
    # Set 5 minute timeout for training
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minutes = 300 seconds
    
    try:
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 ** 2)  # MB
        
        # Generate random pipeline with allocated CPU cores
        pipeline, metadata = generate_random_pipeline(
            iteration=iteration,
            random_state=random_state,
            base_preprocessor=base_preprocessor,
            n_jobs=n_jobs
        )
        
        # Train pipeline on pre-sampled data
        fitted_pipeline = pipeline.fit(X_train_sample, y_train_sample)
        
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
                           X_val_s1, y_val_s1, base_preprocessor, random_state, total_cpus=None):
    """
    Prepare a batch of training jobs for parallel execution.
    
    OPTIMIZATION: Pre-samples training data in main process to minimize serialization.
    Each worker receives only the subset of data it needs (~2.5-27.5%), not the full pool.
    Data is kept as DataFrames (required by sklearn ColumnTransformer).
    
    CPU ALLOCATION: Intelligently distributes available cores across models in the batch.
    Slower parallelizable models (RandomForest, KNN, ExtraTrees) get more cores.
    
    Parameters
    ----------
    iteration : int
        Current iteration number
    batch_size : int
        Number of candidates to train in parallel
    max_iterations : int
        Maximum total iterations
    X_train_pool, y_train_pool : DataFrame/Series
        Training pool data (kept as DataFrames for ColumnTransformer)
    X_val_s1, y_val_s1 : DataFrame/Series
        Stage 1 validation data (kept as DataFrames for ColumnTransformer)
    base_preprocessor : ColumnTransformer
        Base preprocessor for features
    random_state : int
        Base random state
    total_cpus : int, optional
        Total CPUs available for allocation. If None, uses all available cores.
    
    Returns
    -------
    list
        List of tuples for parallel training, each containing pre-sampled training data
        and allocated CPU cores
    """
    # Determine total CPUs available
    if total_cpus is None:
        import multiprocessing
        total_cpus = multiprocessing.cpu_count()
    
    # NOTE: We keep data as DataFrames (NOT converted to numpy) because sklearn's
    # ColumnTransformer requires column names for feature selection.
    # Pre-sampling (below) is the key optimization that reduces serialization overhead.
    
    # Peek at what classifiers will be generated to allocate cores intelligently
    # Classifiers that benefit from parallelization (higher priority):
    # - random_forest, knn, extra_trees: significant speedup with more cores
    # - gradient_boosting: moderate speedup (but limited parallelization)
    # - logistic, linear_svc, sgd, mlp, adaboost, naive_bayes, lda, qda, ridge: no parallelization benefit
    
    classifier_types = []
    for i in range(batch_size):
        current_iter = iteration + i
        if current_iter >= max_iterations:
            break
        
        # Determine classifier type for this iteration
        rng = np.random.RandomState(random_state + current_iter)
        classifier_pool = [
            'logistic', 'random_forest', 'gradient_boosting', 'linear_svc',
            'sgd_classifier', 'mlp', 'knn', 'extra_trees', 'adaboost',
            'naive_bayes', 'lda', 'qda', 'ridge'
        ]
        classifier_type = rng.choice(classifier_pool)
        classifier_types.append(classifier_type)
    
    # Allocate CPU cores based on classifier types
    # Priority: high-parallelizable > medium > no-parallelization
    high_parallel = ['random_forest', 'extra_trees', 'knn']  # Best speedup
    medium_parallel = ['gradient_boosting']  # Some speedup
    
    n_high = sum(1 for ct in classifier_types if ct in high_parallel)
    n_medium = sum(1 for ct in classifier_types if ct in medium_parallel)
    n_low = len(classifier_types) - n_high - n_medium
    
    # Allocate cores intelligently:
    # - If we have fewer CPUs than jobs, give 1 core to each (sequential training)
    # - Otherwise, give 1 core baseline + distribute extra to parallelizable models
    
    if total_cpus < len(classifier_types):
        # Not enough cores for all jobs to run in parallel - each gets 1 core
        cores_per_job = [1] * len(classifier_types)
    else:
        # Start with 1 core per job as baseline
        cores_per_job = [1] * len(classifier_types)
        
        # We have extra cores to distribute
        extra_cores = total_cpus - len(classifier_types)
        
        if extra_cores > 0 and n_high > 0:
            # Give most extra cores to high-parallelizable models
            cores_for_high = max(1, int(extra_cores * 0.7))
            cores_per_high_model = cores_for_high // n_high
            
            for i, ct in enumerate(classifier_types):
                if ct in high_parallel:
                    cores_per_job[i] += cores_per_high_model
            
            extra_cores -= cores_for_high
        
        if extra_cores > 0 and n_medium > 0:
            # Give remaining cores to medium-parallelizable models
            cores_per_medium_model = extra_cores // n_medium
            
            for i, ct in enumerate(classifier_types):
                if ct in medium_parallel:
                    cores_per_job[i] += cores_per_medium_model
    
    batch_jobs = []
    for i in range(len(classifier_types)):
        current_iter = iteration + i
        if current_iter >= max_iterations:
            break
        
        # Generate pipeline metadata to determine sample size
        # Use same logic as generate_random_pipeline for row_sample_pct
        rng = np.random.RandomState(random_state + current_iter)
        
        # Sample based on typical adaptive sampling range (2.5% - 27.5%)
        row_sample_pct = rng.uniform(0.025, 0.275)
        n_total = len(X_train_pool)
        n_sample = max(100, int(n_total * row_sample_pct))
        
        # Pre-sample training data in main process
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train_pool,
            y_train_pool,
            train_size=n_sample,
            stratify=y_train_pool,
            random_state=random_state + current_iter
        )
        
        # Pass only the sampled data (much smaller!) + allocated CPU cores
        # Data remains as DataFrames for ColumnTransformer compatibility
        batch_jobs.append((
            current_iter,
            X_train_sample,  # Pre-sampled subset (DataFrame)
            y_train_sample,  # Pre-sampled subset (Series)
            X_val_s1,        # Full validation (DataFrame)
            y_val_s1,        # Full validation (Series)
            base_preprocessor,
            random_state + current_iter,
            cores_per_job[i]  # Allocated CPU cores for this model
        ))
    
    return batch_jobs
