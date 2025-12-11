"""Worker process management for parallel ensemble training.

This module provides functions for training individual candidates in
isolated worker processes with timeout handling.
"""

import multiprocessing
import os
import signal
import psutil
import warnings
from typing import Any, Optional
from datetime import datetime


def train_single_candidate(
    job_args: tuple,
    database,
    logger
) -> dict:
    """Train a single candidate model with timeout protection.
    
    This function runs the training in a separate process to enable
    forceful termination on timeout. It kills not just the worker process
    but all child processes (e.g., sklearn's loky workers).
    
    Parameters
    ----------
    job_args : tuple
        Arguments for the training job:
        (iteration, X_train, y_train, X_val, y_val, preprocessor,
         random_state, n_jobs, worker_id, batch_num, timeout_seconds,
         classifier_type)
    database : EnsembleDatabase
        Database for status tracking.
    logger : logging.Logger
        Logger for status messages.
    
    Returns
    -------
    result : dict or None
        Training result dictionary if successful, None if timeout/error.
    
    Raises
    ------
    TimeoutError
        If training exceeds timeout.
    """
    iteration = job_args[0]
    worker_id = job_args[8]
    timeout_seconds = job_args[10]
    
    # Create queue for result passing
    result_queue = multiprocessing.Queue()
    
    # Start worker process
    worker = multiprocessing.Process(
        target=_train_worker,
        args=(job_args, database, logger, result_queue)
    )
    
    worker.start()
    worker.join(timeout=timeout_seconds)
    
    if worker.is_alive():
        # Timeout occurred - forcefully terminate
        logger.warning(
            f"Iteration {iteration} (worker {worker_id}) exceeded timeout "
            f"({timeout_seconds}s). Forcefully terminating..."
        )
        
        # Get parent process to kill all children
        try:
            parent = psutil.Process(worker.pid)
            children = parent.children(recursive=True)
            
            # Kill all child processes first (e.g., sklearn loky workers)
            for child in children:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Kill parent worker process
            parent.kill()
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Wait for process cleanup
        worker.join(timeout=5)
        
        # Update database status
        batch_num = job_args[9]
        database.update_worker_status(
            worker_id=worker_id,
            iteration_num=iteration,
            batch_num=batch_num,
            status='timeout'
        )
        
        raise TimeoutError(
            f"Training for iteration {iteration} exceeded timeout "
            f"({timeout_seconds}s)"
        )
    
    # Check if result is available
    if not result_queue.empty():
        status, payload = result_queue.get()
        
        if status == 'success':
            return payload
        elif status == 'error':
            logger.error(f"Iteration {iteration}: Training failed - {payload}")
            return None
    
    # Process ended without result
    logger.error(f"Iteration {iteration}: Worker process ended without result")
    return None


def _train_worker(
    job_args: tuple,
    database,
    logger,
    result_queue: multiprocessing.Queue
):
    """Worker process function for training a candidate model.
    
    This function runs in an isolated process. It trains a model,
    evaluates it, and puts the result in a queue.
    
    Parameters
    ----------
    job_args : tuple
        Training job arguments.
    database : EnsembleDatabase
        Database for status tracking.
    logger : logging.Logger
        Logger for status messages.
    result_queue : Queue
        Queue for passing results back to parent.
    """
    # Unpack arguments
    (iteration, X_train, y_train, X_val, y_val, preprocessor,
     random_state, n_jobs, worker_id, batch_num, timeout_seconds,
     classifier_type) = job_args
    
    # Suppress sklearn convergence warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    
    try:
        # Update status to running
        start_time = datetime.now().isoformat()
        database.update_worker_status(
            worker_id=worker_id,
            iteration_num=iteration,
            batch_num=batch_num,
            status='running',
            classifier_type=classifier_type,
            start_time=start_time
        )
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Import here to avoid serialization issues
        import sys
        from pathlib import Path
        
        # Add notebooks/functions to path for generate_random_pipeline
        notebooks_functions = Path(__file__).parent.parent.parent / 'notebooks' / 'functions'
        if str(notebooks_functions) not in sys.path:
            sys.path.insert(0, str(notebooks_functions))
        
        from ensemble_hill_climbing import generate_random_pipeline
        
        # Generate random pipeline with full feature engineering
        candidate, metadata = generate_random_pipeline(
            iteration=iteration,
            random_state=random_state,
            base_preprocessor=preprocessor,
            n_jobs=n_jobs,
            n_input_features=X_train.shape[1]
        )
        
        # Train pipeline
        candidate.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = candidate.score(X_val, y_val)
        val_probs = candidate.predict_proba(X_val)
        
        # Get memory after training
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before
        
        # Prepare result
        result = {
            'iteration': iteration,
            'classifier_type': classifier_type,
            'val_accuracy': val_score,
            'val_probs': val_probs,
            'pipeline': candidate,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'memory_mb': mem_delta,
            'worker_id': worker_id,
            'batch_num': batch_num
        }
        
        # Update status to completed
        from dateutil import parser as date_parser
        end_time = datetime.now().isoformat()
        start_dt = date_parser.parse(start_time)
        end_dt = date_parser.parse(end_time)
        runtime_sec = (end_dt - start_dt).total_seconds()
        
        database.update_worker_status(
            worker_id=worker_id,
            iteration_num=iteration,
            batch_num=batch_num,
            status='completed',
            classifier_type=classifier_type,
            end_time=end_time,
            runtime_sec=runtime_sec
        )
        
        # Put result in queue
        result_queue.put(('success', result))
        
    except Exception as e:
        # Update status to error
        error_time = datetime.now().isoformat()
        database.update_worker_status(
            worker_id=worker_id,
            iteration_num=iteration,
            batch_num=batch_num,
            status='error',
            end_time=error_time
        )
        
        # Put error in queue
        error_msg = f"{type(e).__name__}: {str(e)}"
        result_queue.put(('error', error_msg))


def train_batch_parallel(
    batch_jobs: list,
    database,
    logger,
    max_workers: Optional[int] = None
) -> list:
    """Train a batch of candidates in parallel using multiprocessing.
    
    Parameters
    ----------
    batch_jobs : list of tuple
        List of job arguments for parallel training.
    database : EnsembleDatabase
        Database for status tracking.
    logger : logging.Logger
        Logger for status messages.
    max_workers : int or None, default=None
        Maximum number of parallel workers. If None, uses number of CPUs.
    
    Returns
    -------
    results : list of dict
        List of successful training results. Failed jobs return None.
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Clear batch status before starting
    database.clear_batch_status()
    
    results = []
    
    # Train each candidate (already prepared for parallel execution)
    for job_args in batch_jobs:
        try:
            result = train_single_candidate(
                job_args=job_args,
                database=database,
                logger=logger
            )
            results.append(result)
            
        except TimeoutError as e:
            logger.warning(str(e))
            results.append(None)
        except Exception as e:
            logger.error(
                f"Iteration {job_args[0]}: Unexpected error - {e}"
            )
            results.append(None)
    
    return results
