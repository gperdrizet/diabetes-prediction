"""Database tracking for ensemble training.

This module provides clean, type-safe database operations for logging
ensemble training iterations and Stage 2 DNN training.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd


class EnsembleDatabase:
    """SQLite database manager for ensemble training.
    
    Uses WAL mode for concurrent read/write access, allowing
    the dashboard to query data while training is in progress.
    """
    
    def __init__(self, db_path: Path):
        """Initialize database manager.
        
        Parameters
        ----------
        db_path : Path
            Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.timeout = 30.0  # Seconds
    
    def reset(self) -> None:
        """Delete the database file if it exists to start fresh."""
        if self.db_path.exists():
            self.db_path.unlink()
            print(f"Deleted existing database: {self.db_path}")
        else:
            print(f"No existing database found at: {self.db_path}")
    
    def initialize(self) -> None:
        """Initialize database with required tables and indexes.
        
        Creates three tables:
        - ensemble_log: Hill climbing iteration data
        - stage2_log: Stage 2 DNN training epoch data
        - batch_status: Active worker tracking
        
        Enables WAL mode for concurrent access and creates indexes.
        Safe to call multiple times - only creates tables if they don't exist.
        """
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect and enable WAL mode
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
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
        
        # Create batch_status table
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
        
        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_iteration_num ON ensemble_log(iteration_num)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ensemble_id ON ensemble_log(ensemble_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_stage2_ensemble ON stage2_log(ensemble_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_stage2_epoch ON stage2_log(epoch)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_status(batch_num)')
        
        conn.commit()
        conn.close()
        
        print(f"Database initialized at: {self.db_path}")
    
    def insert_iteration(self, iteration_data: Dict) -> None:
        """Insert a hill climbing iteration record.
        
        Parameters
        ----------
        iteration_data : dict
            Dictionary containing iteration data with required keys:
            timestamp, iteration_num, ensemble_id, stage1_val_auc, stage2_val_auc,
            diversity_score, temperature, accepted, num_models, pipeline_hash.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            conn.execute('''
                INSERT INTO ensemble_log (
                    timestamp, iteration_num, ensemble_id, stage1_val_auc, stage2_val_auc,
                    diversity_score, temperature, accepted, rejection_reason,
                    num_models, classifier_type, transformers_used, use_pca, pca_components,
                    pipeline_hash, training_memory_mb, stage2_memory_mb, training_time_sec, 
                    stage2_time_sec, timeout, stage2_tp, stage2_fp, stage2_tn, stage2_fn
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
    
    def insert_stage2_epoch(self, epoch_data: Dict) -> None:
        """Insert a Stage 2 DNN training epoch record.
        
        Parameters
        ----------
        epoch_data : dict
            Dictionary with keys: timestamp, ensemble_id, epoch,
            train_loss, val_loss, train_auc, val_auc.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
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
    
    def query_iterations(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Query ensemble iteration data.
        
        Parameters
        ----------
        limit : int or None, default=None
            Maximum number of rows to return (most recent first).
        
        Returns
        -------
        df : pd.DataFrame
            Iteration data, empty if no data exists.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            query = 'SELECT * FROM ensemble_log ORDER BY iteration_num DESC'
            if limit is not None:
                query += f' LIMIT {limit}'
            
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()
    
    def query_stage2_epochs(self, ensemble_id: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Query Stage 2 DNN training data for a specific ensemble.
        
        Parameters
        ----------
        ensemble_id : str
            Unique ensemble identifier.
        limit : int or None, default=None
            Maximum number of rows to return (most recent first).
        
        Returns
        -------
        df : pd.DataFrame
            Stage 2 epoch data, empty if no data exists.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
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
    
    def get_accepted_ensemble_ids(self) -> List[str]:
        """Get list of all accepted ensemble IDs.
        
        Returns
        -------
        ensemble_ids : list of str
            List of ensemble IDs, sorted by first appearance.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ensemble_id 
                FROM ensemble_log 
                WHERE accepted = 1
                GROUP BY ensemble_id
                ORDER BY MIN(iteration_num)
            ''')
            ensemble_ids = [row[0] for row in cursor.fetchall()]
            return ensemble_ids
        finally:
            conn.close()
    
    def clear_batch_status(self) -> None:
        """Clear all batch status entries.
        
        Call this at the start of each new batch to reset worker tracking.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            conn.execute('DELETE FROM batch_status')
            conn.commit()
        finally:
            conn.close()
    
    def update_worker_status(
        self,
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
        """Update or insert worker status.
        
        Parameters
        ----------
        worker_id : int
            Unique worker identifier (0 to N_WORKERS-1).
        iteration_num : int
            Iteration number being processed.
        batch_num : int
            Batch number.
        status : str
            One of 'running', 'completed', 'timeout', 'error'.
        classifier_type : str, optional
            Type of classifier being trained.
        start_time : str, optional
            ISO timestamp when worker started.
        end_time : str, optional
            ISO timestamp when worker completed.
        runtime_sec : float, optional
            Runtime in seconds.
        pipeline_hash : str, optional
            Hash of pipeline configuration.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
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
    
    def get_batch_status(self) -> pd.DataFrame:
        """Get current batch status for all workers.
        
        Returns
        -------
        df : pd.DataFrame
            Batch status data, empty if no data exists.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            df = pd.read_sql_query(
                'SELECT * FROM batch_status ORDER BY worker_id',
                conn
            )
            return df
        finally:
            conn.close()
    
    def exists(self) -> bool:
        """Check if the database file exists.
        
        Returns
        -------
        exists : bool
            True if database file exists.
        """
        return self.db_path.exists()
    
    def get_size_mb(self) -> float:
        """Get the size of the database file in MB.
        
        Returns
        -------
        size_mb : float
            Size in megabytes, or 0 if database doesn't exist.
        """
        if self.db_path.exists():
            return self.db_path.stat().st_size / (1024 * 1024)
        return 0.0
