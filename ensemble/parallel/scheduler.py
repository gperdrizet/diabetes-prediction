"""Batch scheduling and coordination for parallel ensemble training.

This module provides clean functions for preparing and managing batches
of candidate models for parallel training.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
from sklearn.model_selection import train_test_split

from ensemble.config import EnsembleConfig


def prepare_training_batch(
    iteration: int,
    batch_size: int,
    max_iterations: int,
    X_train_pool,
    y_train_pool,
    X_val_s1,
    y_val_s1,
    base_preprocessor,
    random_state: int,
    total_cpus: Optional[int] = None,
    timeout_minutes: int = 60,
    batch_num: int = 0,
    row_sample_range: Tuple[float, float] = (0.10, 0.40)
) -> List[Tuple]:
    """Prepare a batch of training jobs for parallel execution.
    
    OPTIMIZATION: Pre-samples training data in main process to minimize serialization.
    Each worker receives only the subset of data it needs (~10-40%), not the full pool.
    
    CPU ALLOCATION: Intelligently distributes available cores across models in the batch.
    Parallelizable models (RandomForest, KNN, ExtraTrees) get more cores.
    
    Parameters
    ----------
    iteration : int
        Starting iteration number for this batch.
    batch_size : int
        Number of candidates to train in parallel.
    max_iterations : int
        Maximum total iterations.
    X_train_pool, y_train_pool : DataFrame/Series
        Training pool data (kept as DataFrames for ColumnTransformer).
    X_val_s1, y_val_s1 : DataFrame/Series
        Stage 1 validation data.
    base_preprocessor : ColumnTransformer
        Base preprocessor for features.
    random_state : int
        Base random state.
    total_cpus : int or None, default=None
        Total CPUs available. If None, uses all available cores.
    timeout_minutes : int, default=60
        Timeout in minutes for each model training.
    batch_num : int, default=0
        Batch number for tracking.
    row_sample_range : tuple of float, default=(0.10, 0.40)
        Range for random row sampling.
    
    Returns
    -------
    batch_jobs : list of tuple
        List of arguments for each training job. Each tuple contains 12 elements:
        (iteration, X_train_sample, y_train_sample, X_val_s1, y_val_s1,
         base_preprocessor, random_state, n_jobs, worker_id, batch_num,
         timeout_seconds, classifier_type)
    """
    # Determine total CPUs available
    if total_cpus is None:
        import multiprocessing
        total_cpus = multiprocessing.cpu_count()
    
    # Determine classifier types for this batch
    classifier_types = _sample_classifier_types(
        iteration=iteration,
        batch_size=batch_size,
        max_iterations=max_iterations,
        random_state=random_state
    )
    
    # Allocate CPU cores intelligently based on classifier types
    cores_per_job = _allocate_cpu_cores(
        classifier_types=classifier_types,
        total_cpus=total_cpus
    )
    
    # Prepare batch jobs with pre-sampled data
    batch_jobs = []
    for i, classifier_type in enumerate(classifier_types):
        current_iter = iteration + i
        if current_iter >= max_iterations:
            break
        
        # Determine sample size
        rng = np.random.RandomState(random_state + current_iter)
        row_sample_pct = rng.uniform(*row_sample_range)
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
        
        # Create job tuple
        batch_jobs.append((
            current_iter,                  # iteration
            X_train_sample,                # pre-sampled features (DataFrame)
            y_train_sample,                # pre-sampled labels (Series)
            X_val_s1,                      # validation features
            y_val_s1,                      # validation labels
            base_preprocessor,             # preprocessor
            random_state + current_iter,   # random state
            cores_per_job[i],              # allocated CPU cores
            i,                             # worker ID
            batch_num,                     # batch number
            timeout_minutes * 60,          # timeout in seconds
            classifier_type                # classifier type
        ))
    
    return batch_jobs


def _sample_classifier_types(
    iteration: int,
    batch_size: int,
    max_iterations: int,
    random_state: int
) -> List[str]:
    """Sample classifier types for a batch.
    
    Parameters
    ----------
    iteration : int
        Starting iteration number.
    batch_size : int
        Number of classifiers to sample.
    max_iterations : int
        Maximum total iterations.
    random_state : int
        Base random state.
    
    Returns
    -------
    classifier_types : list of str
        List of classifier type names.
    """
    # Active classifier pool
    classifier_pool = [
        'logistic', 'random_forest', 'linear_svc',
        'sgd_classifier', 'extra_trees', 'adaboost',
        'naive_bayes', 'lda', 'qda', 'ridge'
    ]
    
    classifier_types = []
    for i in range(batch_size):
        current_iter = iteration + i
        if current_iter >= max_iterations:
            break
        
        rng = np.random.RandomState(random_state + current_iter)
        classifier_type = rng.choice(classifier_pool)
        classifier_types.append(classifier_type)
    
    return classifier_types


def _allocate_cpu_cores(
    classifier_types: List[str],
    total_cpus: int
) -> List[int]:
    """Allocate CPU cores to classifiers based on parallelizability.
    
    Classifiers that benefit from parallelization get more cores:
    - High parallel: random_forest, extra_trees, knn (significant speedup)
    - Medium parallel: gradient_boosting (moderate speedup)
    - Low parallel: logistic, linear_svc, sgd, mlp, adaboost, naive_bayes, lda, qda, ridge
    
    Parameters
    ----------
    classifier_types : list of str
        List of classifier types for the batch.
    total_cpus : int
        Total CPU cores available.
    
    Returns
    -------
    cores_per_job : list of int
        CPU cores allocated to each job.
    """
    high_parallel = ['random_forest', 'extra_trees', 'knn']
    medium_parallel = ['gradient_boosting']
    
    n_high = sum(1 for ct in classifier_types if ct in high_parallel)
    n_medium = sum(1 for ct in classifier_types if ct in medium_parallel)
    n_jobs = len(classifier_types)
    
    # Start with 1 core per job as baseline
    cores_per_job = [1] * n_jobs
    
    # If not enough cores, each job gets 1 core (sequential)
    if total_cpus < n_jobs:
        return cores_per_job
    
    # Distribute extra cores to parallelizable models
    extra_cores = total_cpus - n_jobs
    
    # Give 70% of extra cores to high-parallelizable models
    if extra_cores > 0 and n_high > 0:
        cores_for_high = max(1, int(extra_cores * 0.7))
        cores_per_high_model = cores_for_high // n_high
        
        for i, ct in enumerate(classifier_types):
            if ct in high_parallel:
                cores_per_job[i] += cores_per_high_model
        
        extra_cores -= cores_for_high
    
    # Give remaining cores to medium-parallelizable models
    if extra_cores > 0 and n_medium > 0:
        cores_per_medium_model = extra_cores // n_medium
        
        for i, ct in enumerate(classifier_types):
            if ct in medium_parallel:
                cores_per_job[i] += cores_per_medium_model
    
    return cores_per_job


def get_batch_info(batch_jobs: List[Tuple]) -> dict:
    """Get summary information about a prepared batch.
    
    Parameters
    ----------
    batch_jobs : list of tuple
        Prepared batch jobs.
    
    Returns
    -------
    info : dict
        Dictionary with batch statistics.
    """
    if not batch_jobs:
        return {
            'n_jobs': 0,
            'total_cores': 0,
            'avg_cores_per_job': 0.0,
            'classifier_counts': {}
        }
    
    total_cores = sum(job[7] for job in batch_jobs)  # cores are at index 7
    classifier_types = [job[11] for job in batch_jobs]  # classifier_type at index 11
    
    # Count classifier types
    classifier_counts = {}
    for ct in classifier_types:
        classifier_counts[ct] = classifier_counts.get(ct, 0) + 1
    
    return {
        'n_jobs': len(batch_jobs),
        'total_cores': total_cores,
        'avg_cores_per_job': total_cores / len(batch_jobs),
        'classifier_counts': classifier_counts
    }
