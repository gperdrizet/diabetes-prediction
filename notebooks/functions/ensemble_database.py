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

# Hardcoded database path
DB_PATH = '/workspaces/diabetes-prediction/data/ensemble_training.db'

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
            cv_score REAL NOT NULL,
            diversity_score REAL NOT NULL,
            combined_score REAL NOT NULL,
            temperature REAL NOT NULL,
            accepted INTEGER NOT NULL,
            acceptance_reason TEXT,
            num_models INTEGER NOT NULL,
            transformers_used TEXT,
            pipeline_hash TEXT NOT NULL
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
    
    # Create indexes for common queries
    conn.execute('CREATE INDEX IF NOT EXISTS idx_iteration_num ON ensemble_log(iteration_num)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ensemble_id ON ensemble_log(ensemble_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_stage2_ensemble ON stage2_log(ensemble_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_stage2_epoch ON stage2_log(epoch)')
    
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
            - cv_score (float): Cross-validation score
            - diversity_score (float): Ensemble diversity score
            - combined_score (float): Combined objective score
            - temperature (float): Current simulated annealing temperature
            - accepted (int): 1 if accepted, 0 if rejected
            - acceptance_reason (str): Reason for acceptance/rejection
            - num_models (int): Number of models in ensemble
            - transformers_used (str): Comma-separated transformer names
            - pipeline_hash (str): Hash of pipeline configuration
    
    Raises:
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        conn.execute('''
            INSERT INTO ensemble_log (
                timestamp, iteration_num, ensemble_id, cv_score, diversity_score,
                combined_score, temperature, accepted, acceptance_reason,
                num_models, transformers_used, pipeline_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            iteration_data['timestamp'],
            iteration_data['iteration_num'],
            iteration_data['ensemble_id'],
            iteration_data['cv_score'],
            iteration_data['diversity_score'],
            iteration_data['combined_score'],
            iteration_data['temperature'],
            iteration_data['accepted'],
            iteration_data.get('acceptance_reason', ''),
            iteration_data['num_models'],
            iteration_data.get('transformers_used', ''),
            iteration_data['pipeline_hash']
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


def reset_database() -> None:
    """Reset the database by dropping and recreating all tables.
    
    WARNING: This is a DESTRUCTIVE operation that permanently deletes all data.
    This operation is IRREVOCABLE. Use with extreme caution.
    
    Raises:
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        # Drop existing tables
        conn.execute('DROP TABLE IF EXISTS ensemble_log')
        conn.execute('DROP TABLE IF EXISTS stage2_log')
        conn.commit()
        conn.close()
        
        # Reinitialize with fresh tables
        init_database()
        
        print("WARNING: Database has been reset. All previous data has been permanently deleted.")
    except Exception as e:
        conn.close()
        raise e


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
