"""
Parallel training utilities for ensemble hill climbing.

Handles batch training of candidate models and helper functions.
"""

import time
import psutil
import os
import signal
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from concurrent.futures import TimeoutError
from multiprocessing import Process, Queue

from .ensemble_hill_climbing import generate_random_pipeline, compute_pipeline_hash


def train_single_candidate(args):
    """
    Train a single candidate with robust timeout handling.
    
    Uses multiprocessing.Process with forceful termination to ensure
    that sklearn's internal loky workers are also killed on timeout.
    
    Parameters
    ----------
    args : tuple
        (iteration, X_train_sample, y_train_sample, X_val_s1, y_val_s1, 
         base_preprocessor, random_state, n_jobs, timeout_seconds)
    
    Returns
    -------
    dict : Training result
    
    Raises
    ------
    TimeoutError : If training exceeds timeout
    Exception : If training fails
    """
    # Extract timeout from args (default 15 minutes = 900 seconds)
    if len(args) == 9:
        timeout_seconds = args[8]
        args = args[:8]  # Remove timeout from args for worker
    else:
        timeout_seconds = 900  # Default 15 minutes
    
    result_queue = Queue()
    
    # Start worker process
    process = Process(target=_train_worker, args=(args, result_queue))
    process.start()
    
    # Wait for completion with timeout
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
        # Timeout - forcefully kill the process and all children
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except:
                pass
        parent.kill()
        process.join()  # Clean up zombie
        raise TimeoutError(f"Training exceeded {timeout_seconds/60:.1f} minutes (iteration {args[0]})")
    
    # Check if we got a result
    if result_queue.empty():
        raise Exception(f"Process terminated without result (iteration {args[0]})")
    
    status, result = result_queue.get()
    
    if status == 'error':
        raise Exception(result)
    
    return result


def _train_worker(args, result_queue):
    """
    Worker function that trains a model and puts result in queue.
    Runs in a separate process that can be forcefully terminated.
    
    Parameters
    ----------
    args : tuple
        Training arguments
    result_queue : multiprocessing.Queue
        Queue to put the result in
    """
    iteration, X_train_sample, y_train_sample, X_val_s1, y_val_s1, base_preprocessor, random_state, n_jobs = args
    
    try:
        # Suppress expected warnings from random hyperparameter exploration
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.decomposition._fastica')
        warnings.filterwarnings('ignore', message='.*FastICA did not converge.*')
        
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
        
        # Print model configuration before training starts
        classifier_type = metadata['classifier_type']
        sample_size = len(X_train_sample)
        
        print(f"\n[Iteration {iteration}] Training {classifier_type} + {', '.join(metadata['transformers_used']) if metadata['transformers_used'] else 'None'}, Sample size: {sample_size} rows ({metadata['row_sample_pct']*100:.1f}%)")
        print(f"  Feature sampling: {metadata['col_sample_pct']*100:.1f}%")
     
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
        
        result = {
            'iteration': iteration,
            'fitted_pipeline': fitted_pipeline,
            'metadata': metadata,
            'val_auc_s1': val_auc_s1,
            'pipeline_hash': pipeline_hash,
            'training_time': training_time,
            'memory_mb': memory_used,
            'training_time_sec': training_time
        }
        
        result_queue.put(('success', result))
        
    except Exception as e:
        result_queue.put(('error', str(e)))


def prepare_training_batch(iteration, batch_size, max_iterations, X_train_pool, y_train_pool,
                           X_val_s1, y_val_s1, base_preprocessor, random_state, total_cpus=None, timeout_minutes=15):
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
    timeout_minutes : int, optional
        Timeout in minutes for each model training. Default is 15 minutes.
    
    Returns
    -------
    list
        List of tuples for parallel training, each containing pre-sampled training data,
        allocated CPU cores, and timeout in seconds
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
            'logistic', 'random_forest', 'linear_svc',
            'sgd_classifier', 'extra_trees', 'adaboost',
            'naive_bayes', 'lda', 'qda', 'ridge'
            # TEMPORARILY DISABLED (too slow):
            # 'gradient_boosting', 'mlp', 'knn'
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
        
        # Sample based on actual range used in generate_random_pipeline (10% - 40%)
        row_sample_pct = rng.uniform(0.10, 0.40)
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
        
        # Pass only the sampled data (much smaller!) + allocated CPU cores + timeout
        # Data remains as DataFrames for ColumnTransformer compatibility
        batch_jobs.append((
            current_iter,
            X_train_sample,  # Pre-sampled subset (DataFrame)
            y_train_sample,  # Pre-sampled subset (Series)
            X_val_s1,        # Full validation (DataFrame)
            y_val_s1,        # Full validation (Series)
            base_preprocessor,
            random_state + current_iter,
            cores_per_job[i],  # Allocated CPU cores for this model
            timeout_minutes * 60  # Timeout in seconds
        ))
    
    return batch_jobs
