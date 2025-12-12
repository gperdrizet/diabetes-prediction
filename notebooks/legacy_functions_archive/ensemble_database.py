"""SQLite database manager for ensemble hill climbing training.

This module provides functions for logging ensemble training iterations and stage 2 DNN
training to a SQLite database. Uses WAL mode for concurrent read/write access, allowing
the dashboard to query data while training is in progress.

Database Schema:
    ensemble_log: Hill climbing iteration data
    stage2_log: Stage 2 DNN training epoch data

Author: Generated for diabetes-prediction project
Date: December 6, 2025
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd

# Database path - use relative path from project root
DB_PATH = Path(__file__).parent.parent.parent / 'data' / 'ensemble_training.db'

def reset_database() -> None:
    """Delete the database file if it exists to start fresh.
    
    This ensures each training run starts with a clean database.
    """
    db_path = Path(DB_PATH)
    if db_path.exists():
        db_path.unlink()
        print(f"Deleted existing database: {DB_PATH}")
    else:
        print(f"No existing database found at: {DB_PATH}")


def init_database() -> None:
    """Initialize the SQLite database with required tables and indexes.
    
    Creates two tables:
        - ensemble_log: Hill climbing iteration data
        - stage2_log: Stage 2 DNN training epoch data
    
    Enables WAL mode for concurrent access and creates indexes for common queries.
    Safe to call multiple times - only creates tables if they don't exist.
    """
    # Ensure data directory exists
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect and enable WAL mode
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute('PRAGMA journal_mode=WAL')
    
    # Create ensemble_log table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ensemble_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            iteration_num INTEGER NOT NULL,
            ensemble_id TEXT NOT NULL,
            stage1_val_auc REAL NOT NULL,
            stage2_val_auc REAL NOT NULL,
            diversity_score REAL NOT NULL,
            temperature REAL NOT NULL,
            accepted INTEGER NOT NULL,
            rejection_reason TEXT,
            num_models INTEGER NOT NULL,
            classifier_type TEXT,
            transformers_used TEXT,
            use_pca INTEGER,
            pca_components REAL,
            pipeline_hash TEXT NOT NULL,
            training_memory_mb REAL,
            stage2_memory_mb REAL,
            training_time_sec REAL,
            stage2_time_sec REAL,
            timeout INTEGER DEFAULT 0,
            stage2_tp INTEGER,
            stage2_fp INTEGER,
            stage2_tn INTEGER,
            stage2_fn INTEGER
        )
    ''')
    
    # Create stage2_log table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS stage2_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ensemble_id TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            train_loss REAL NOT NULL,
            val_loss REAL NOT NULL,
            train_auc REAL NOT NULL,
            val_auc REAL NOT NULL
        )
    ''')
    
    # Create batch_status table for tracking active workers
    conn.execute('''
        CREATE TABLE IF NOT EXISTS batch_status (
            worker_id INTEGER PRIMARY KEY,
            iteration_num INTEGER NOT NULL,
            batch_num INTEGER NOT NULL,
            status TEXT NOT NULL,
            classifier_type TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            runtime_sec REAL,
            pipeline_hash TEXT,
            last_update TEXT NOT NULL
        )
    ''')
    
    # Create indexes for common queries
    conn.execute('CREATE INDEX IF NOT EXISTS idx_iteration_num ON ensemble_log(iteration_num)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ensemble_id ON ensemble_log(ensemble_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_stage2_ensemble ON stage2_log(ensemble_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_stage2_epoch ON stage2_log(epoch)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_status(batch_num)')
    
    # Add confusion matrix columns if they don't exist (for existing databases)
    try:
        conn.execute('ALTER TABLE ensemble_log ADD COLUMN stage2_tp INTEGER')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        conn.execute('ALTER TABLE ensemble_log ADD COLUMN stage2_fp INTEGER')
    except sqlite3.OperationalError:
        pass
    
    try:
        conn.execute('ALTER TABLE ensemble_log ADD COLUMN stage2_tn INTEGER')
    except sqlite3.OperationalError:
        pass
    
    try:
        conn.execute('ALTER TABLE ensemble_log ADD COLUMN stage2_fn INTEGER')
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized at: {DB_PATH}")


def insert_ensemble_iteration(iteration_data: Dict) -> None:
    """Insert a hill climbing iteration record into the database.
    
    Args:
        iteration_data: Dictionary containing iteration data with keys:
            - timestamp (str): ISO format timestamp
            - iteration_num (int): Iteration number
            - ensemble_id (str): Unique ensemble identifier
            - stage1_val_auc (float): Stage 1 validation AUC
            - stage2_val_auc (float): Stage 2 validation AUC (ensemble)
            - diversity_score (float): Ensemble diversity score
            - temperature (float): Current simulated annealing temperature
            - accepted (int): 1 if accepted, 0 if rejected
            - rejection_reason (str): Reason for acceptance/rejection
            - num_models (int): Number of models in ensemble
            - classifier_type (str): Type of classifier
            - transformers_used (str): Comma-separated transformer names
            - use_pca (int): 1 if PCA used, 0 otherwise
            - pca_components (float): Number of PCA components
            - pipeline_hash (str): Hash of pipeline configuration
            - timeout (int): 1 if training timed out, 0 otherwise
    
    Raises:
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        conn.execute('''
            INSERT INTO ensemble_log (
                timestamp, iteration_num, ensemble_id, stage1_val_auc, stage2_val_auc,
                diversity_score, temperature, accepted, rejection_reason,
                num_models, classifier_type, transformers_used, use_pca, pca_components,
                pipeline_hash, training_memory_mb, stage2_memory_mb, training_time_sec, stage2_time_sec,
                timeout, stage2_tp, stage2_fp, stage2_tn, stage2_fn
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            iteration_data['timestamp'],
            iteration_data['iteration_num'],
            iteration_data['ensemble_id'],
            iteration_data['stage1_val_auc'],
            iteration_data['stage2_val_auc'],
            iteration_data['diversity_score'],
            iteration_data['temperature'],
            iteration_data['accepted'],
            iteration_data.get('rejection_reason', ''),
            iteration_data['num_models'],
            iteration_data.get('classifier_type', ''),
            iteration_data.get('transformers_used', ''),
            iteration_data.get('use_pca', 0),
            iteration_data.get('pca_components'),
            iteration_data['pipeline_hash'],
            iteration_data.get('training_memory_mb'),
            iteration_data.get('stage2_memory_mb'),
            iteration_data.get('training_time_sec'),
            iteration_data.get('stage2_time_sec'),
            iteration_data.get('timeout', 0),
            iteration_data.get('stage2_tp'),
            iteration_data.get('stage2_fp'),
            iteration_data.get('stage2_tn'),
            iteration_data.get('stage2_fn')
        ))
        conn.commit()
    finally:
        conn.close()


def insert_stage2_epoch(epoch_data: Dict) -> None:
    """Insert a stage 2 DNN training epoch record into the database.
    
    Args:
        epoch_data: Dictionary containing epoch data with keys:
            - timestamp (str): ISO format timestamp
            - ensemble_id (str): Unique ensemble identifier
            - epoch (int): Epoch number
            - train_loss (float): Training loss
            - val_loss (float): Validation loss
            - train_auc (float): Training AUC score
            - val_auc (float): Validation AUC score
    
    Raises:
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        conn.execute('''
            INSERT INTO stage2_log (
                timestamp, ensemble_id, epoch, train_loss, val_loss,
                train_auc, val_auc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            epoch_data['timestamp'],
            epoch_data['ensemble_id'],
            epoch_data['epoch'],
            epoch_data['train_loss'],
            epoch_data['val_loss'],
            epoch_data['train_auc'],
            epoch_data['val_auc']
        ))
        conn.commit()
    finally:
        conn.close()


def query_ensemble_data(limit: Optional[int] = None) -> pd.DataFrame:
    """Query ensemble iteration data from the database.
    
    Args:
        limit: Optional limit on number of rows to return (most recent first)
    
    Returns:
        DataFrame containing ensemble iteration data, empty if no data exists
    
    Raises:
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        query = 'SELECT * FROM ensemble_log ORDER BY iteration_num DESC'
        if limit is not None:
            query += f' LIMIT {limit}'
        
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def query_stage2_data(ensemble_id: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Query stage 2 DNN training data for a specific ensemble.
    
    Args:
        ensemble_id: Unique ensemble identifier
        limit: Optional limit on number of rows to return (most recent first)
    
    Returns:
        DataFrame containing stage 2 epoch data, empty if no data exists
    
    Raises:
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        query = '''
            SELECT * FROM stage2_log 
            WHERE ensemble_id = ? 
            ORDER BY epoch DESC
        '''
        if limit is not None:
            query += f' LIMIT {limit}'
        
        df = pd.read_sql_query(query, conn, params=(ensemble_id,))
        return df
    finally:
        conn.close()


def get_summary_stats() -> Dict:
    """Get aggregate statistics from the database.
    
    Returns:
        Dictionary containing:
            - total_iterations (int): Total number of iterations logged
            - total_accepted (int): Number of accepted ensembles
            - best_cv_score (float): Best CV score achieved
            - best_combined_score (float): Best combined score achieved
            - current_temperature (float): Most recent temperature value
            - last_update (str): Timestamp of most recent iteration
    
    Raises:
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        
        # Get total iterations and accepted count
        cursor.execute('SELECT COUNT(*), SUM(accepted) FROM ensemble_log')
        total_iterations, total_accepted = cursor.fetchone()
        total_iterations = total_iterations or 0
        total_accepted = total_accepted or 0
        
        # Get best scores
        cursor.execute('SELECT MAX(cv_score), MAX(combined_score) FROM ensemble_log')
        best_cv_score, best_combined_score = cursor.fetchone()
        
        # Get most recent temperature and timestamp
        cursor.execute('''
            SELECT temperature, timestamp 
            FROM ensemble_log 
            ORDER BY iteration_num DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        current_temperature = result[0] if result else None
        last_update = result[1] if result else None
        
        return {
            'total_iterations': total_iterations,
            'total_accepted': total_accepted,
            'best_cv_score': best_cv_score,
            'best_combined_score': best_combined_score,
            'current_temperature': current_temperature,
            'last_update': last_update
        }
    finally:
        conn.close()


def get_all_ensemble_ids() -> List[str]:
    """Get list of all unique ensemble IDs that have been logged.
    
    Returns:
        List of ensemble ID strings, sorted by first appearance
    
    Raises:
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT ensemble_id 
            FROM ensemble_log 
            WHERE accepted = 1
            ORDER BY MIN(iteration_num)
        ''')
        ensemble_ids = [row[0] for row in cursor.fetchall()]
        return ensemble_ids
    finally:
        conn.close()


def database_exists() -> bool:
    """Check if the database file exists.
    
    Returns:
        True if database file exists, False otherwise
    """
    return Path(DB_PATH).exists()


def get_database_size() -> float:
    """Get the size of the database file in MB.
    
    Returns:
        Size in megabytes, or 0 if database doesn't exist
    """
    db_path = Path(DB_PATH)
    if db_path.exists():
        return db_path.stat().st_size / (1024 * 1024)
    return 0.0


# ==============================================================================
# BATCH STATUS TRACKING
# ==============================================================================

def clear_batch_status() -> None:
    """Clear all batch status entries.
    
    Call this at the start of each new batch to reset worker tracking.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        conn.execute('DELETE FROM batch_status')
        conn.commit()
    finally:
        conn.close()


def update_worker_status(
    worker_id: int,
    iteration_num: int,
    batch_num: int,
    status: str,
    classifier_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    runtime_sec: Optional[float] = None,
    pipeline_hash: Optional[str] = None
) -> None:
    """Update or insert worker status in the batch_status table.
    
    Args:
        worker_id: Unique worker identifier (0 to N_WORKERS-1)
        iteration_num: Iteration number being processed
        batch_num: Batch number (increments with each batch)
        status: One of 'running', 'completed', 'timeout', 'error'
        classifier_type: Type of classifier being trained (optional)
        start_time: ISO timestamp when worker started (required for new entries)
        end_time: ISO timestamp when worker completed (optional)
        runtime_sec: Runtime in seconds (optional)
        pipeline_hash: Hash of pipeline configuration (optional)
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        # Check if worker exists
        cursor = conn.cursor()
        cursor.execute('SELECT worker_id FROM batch_status WHERE worker_id = ?', (worker_id,))
        exists = cursor.fetchone() is not None
        
        current_time = datetime.now().isoformat()
        
        if exists:
            # Update existing worker
            conn.execute('''
                UPDATE batch_status
                SET iteration_num = ?,
                    batch_num = ?,
                    status = ?,
                    classifier_type = COALESCE(?, classifier_type),
                    end_time = COALESCE(?, end_time),
                    runtime_sec = COALESCE(?, runtime_sec),
                    pipeline_hash = COALESCE(?, pipeline_hash),
                    last_update = ?
                WHERE worker_id = ?
            ''', (
                iteration_num, batch_num, status, classifier_type,
                end_time, runtime_sec, pipeline_hash, current_time, worker_id
            ))
        else:
            # Insert new worker
            if start_time is None:
                start_time = current_time
            
            conn.execute('''
                INSERT INTO batch_status (
                    worker_id, iteration_num, batch_num, status, classifier_type,
                    start_time, end_time, runtime_sec, pipeline_hash, last_update
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                worker_id, iteration_num, batch_num, status, classifier_type,
                start_time, end_time, runtime_sec, pipeline_hash, current_time
            ))
        
        conn.commit()
    finally:
        conn.close()


def get_batch_status() -> pd.DataFrame:
    """Get current batch status for all workers.
    
    Returns:
        DataFrame with columns: worker_id, iteration_num, batch_num, status,
        classifier_type, start_time, end_time, runtime_sec, pipeline_hash,
        last_update. Empty DataFrame if no data exists.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        df = pd.read_sql_query(
            'SELECT * FROM batch_status ORDER BY worker_id',
            conn
        )
        return df
    finally:
        conn.close()


def get_batch_summary() -> Dict:
    """Get summary statistics for the current batch.
    
    Returns:
        Dictionary containing:
            - total_workers: Total number of workers in batch
            - running: Number of workers currently running
            - completed: Number of workers completed successfully
            - timeout: Number of workers that timed out
            - error: Number of workers that errored
            - avg_runtime: Average runtime for completed workers (seconds)
            - batch_num: Current batch number
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        
        # Get counts by status
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) as timeout,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error,
                AVG(CASE WHEN status = 'completed' THEN runtime_sec ELSE NULL END) as avg_runtime,
                MAX(batch_num) as batch_num
            FROM batch_status
        ''')
        
        result = cursor.fetchone()
        
        return {
            'total_workers': result[0] or 0,
            'running': result[1] or 0,
            'completed': result[2] or 0,
            'timeout': result[3] or 0,
            'error': result[4] or 0,
            'avg_runtime': result[5] or 0.0,
            'batch_num': result[6] or 0
        }
    finally:
        conn.close()
