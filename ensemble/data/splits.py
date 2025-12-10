"""Data splitting utilities for ensemble training.

This module provides clean functions for creating the fixed three-way data split
used in ensemble training: training pool (60%), validation stage 1 (35%), 
validation stage 2 (5%).
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplits:
    """Manages the three-way data split for ensemble training.
    
    The data is split into:
    - Training pool (60%): For training Stage 1 models with sampling
    - Validation Stage 1 (35%): For evaluating Stage 1 models and training Stage 2
    - Validation Stage 2 (5%): Held-out set for final Stage 2 evaluation
    
    All splits are stratified to preserve class distribution.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        label_column: str,
        random_state: int = 42,
        train_pool_size: float = 0.60,
        val_stage1_size: float = 0.35,
        val_stage2_size: float = 0.05
    ):
        """Initialize data splits.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full training dataset with labels.
        label_column : str
            Name of the label column.
        random_state : int, default=42
            Random state for reproducible splits.
        train_pool_size : float, default=0.60
            Fraction of data for training pool.
        val_stage1_size : float, default=0.35
            Fraction of data for Stage 1 validation.
        val_stage2_size : float, default=0.05
            Fraction of data for Stage 2 validation.
        """
        # Validate sizes sum to 1.0
        total = train_pool_size + val_stage1_size + val_stage2_size
        assert abs(total - 1.0) < 1e-6, f"Split sizes must sum to 1.0, got {total}"
        
        self.label_column = label_column
        self.random_state = random_state
        
        # Separate features and labels
        X_full = data.drop(columns=[label_column])
        y_full = data[label_column]
        
        # First split: training pool vs validation combined
        val_combined_size = val_stage1_size + val_stage2_size
        self.X_train_pool, X_val_combined, self.y_train_pool, y_val_combined = train_test_split(
            X_full,
            y_full,
            test_size=val_combined_size,
            random_state=random_state,
            stratify=y_full
        )
        
        # Second split: Stage 1 validation vs Stage 2 validation
        # Calculate relative size: val_stage2 / (val_stage1 + val_stage2)
        relative_stage2_size = val_stage2_size / val_combined_size
        self.X_val_s1, self.X_val_s2, self.y_val_s1, self.y_val_s2 = train_test_split(
            X_val_combined,
            y_val_combined,
            test_size=relative_stage2_size,
            random_state=random_state,
            stratify=y_val_combined
        )
        
        # Store sizes for reference
        self._sizes = {
            'train_pool': len(self.X_train_pool),
            'val_stage1': len(self.X_val_s1),
            'val_stage2': len(self.X_val_s2),
            'total': len(X_full)
        }
        
        # Store class distributions
        self._class_distributions = {
            'train_pool': self.y_train_pool.mean(),
            'val_stage1': self.y_val_s1.mean(),
            'val_stage2': self.y_val_s2.mean(),
            'full': y_full.mean()
        }
    
    def get_train_pool(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training pool data (60%).
        
        Returns
        -------
        X_train_pool : pd.DataFrame
            Training features.
        y_train_pool : pd.Series
            Training labels.
        """
        return self.X_train_pool, self.y_train_pool
    
    def get_val_stage1(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get Stage 1 validation data (35%).
        
        Returns
        -------
        X_val_s1 : pd.DataFrame
            Validation features.
        y_val_s1 : pd.Series
            Validation labels.
        """
        return self.X_val_s1, self.y_val_s1
    
    def get_val_stage2(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get Stage 2 validation data (5% held-out).
        
        Returns
        -------
        X_val_s2 : pd.DataFrame
            Held-out validation features.
        y_val_s2 : pd.Series
            Held-out validation labels.
        """
        return self.X_val_s2, self.y_val_s2
    
    def get_all_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                       pd.Series, pd.Series, pd.Series]:
        """Get all data splits at once.
        
        Returns
        -------
        X_train_pool : pd.DataFrame
            Training pool features (60%).
        X_val_s1 : pd.DataFrame
            Stage 1 validation features (35%).
        X_val_s2 : pd.DataFrame
            Stage 2 validation features (5%).
        y_train_pool : pd.Series
            Training pool labels.
        y_val_s1 : pd.Series
            Stage 1 validation labels.
        y_val_s2 : pd.Series
            Stage 2 validation labels.
        """
        return (
            self.X_train_pool, self.X_val_s1, self.X_val_s2,
            self.y_train_pool, self.y_val_s1, self.y_val_s2
        )
    
    def summary(self) -> str:
        """Get summary of data splits.
        
        Returns
        -------
        summary : str
            Human-readable summary of splits.
        """
        lines = [
            "Data Splits Summary",
            "=" * 60,
            f"Total samples: {self._sizes['total']:,}",
            "",
            "Split sizes:",
            f"  Training pool:    {self._sizes['train_pool']:,} "
            f"({self._sizes['train_pool']/self._sizes['total']*100:.1f}%)",
            f"  Val Stage 1:      {self._sizes['val_stage1']:,} "
            f"({self._sizes['val_stage1']/self._sizes['total']*100:.1f}%)",
            f"  Val Stage 2:      {self._sizes['val_stage2']:,} "
            f"({self._sizes['val_stage2']/self._sizes['total']*100:.1f}%)",
            "",
            "Class distributions (positive rate):",
            f"  Full dataset:     {self._class_distributions['full']:.3f}",
            f"  Training pool:    {self._class_distributions['train_pool']:.3f}",
            f"  Val Stage 1:      {self._class_distributions['val_stage1']:.3f}",
            f"  Val Stage 2:      {self._class_distributions['val_stage2']:.3f}",
            "=" * 60
        ]
        return "\n".join(lines)


def create_three_way_split(
    data: pd.DataFrame,
    label_column: str,
    random_state: int = 42,
    train_pool_size: float = 0.60,
    val_stage1_size: float = 0.35,
    val_stage2_size: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Create three-way stratified split (convenience function).
    
    This is a convenience wrapper around DataSplits for simple use cases.
    
    Parameters
    ----------
    data : pd.DataFrame
        Full training dataset with labels.
    label_column : str
        Name of the label column.
    random_state : int, default=42
        Random state for reproducible splits.
    train_pool_size : float, default=0.60
        Fraction of data for training pool.
    val_stage1_size : float, default=0.35
        Fraction of data for Stage 1 validation.
    val_stage2_size : float, default=0.05
        Fraction of data for Stage 2 validation.
    
    Returns
    -------
    X_train_pool : pd.DataFrame
        Training pool features (60%).
    X_val_s1 : pd.DataFrame
        Stage 1 validation features (35%).
    X_val_s2 : pd.DataFrame
        Stage 2 validation features (5%).
    y_train_pool : pd.Series
        Training pool labels.
    y_val_s1 : pd.Series
        Stage 1 validation labels.
    y_val_s2 : pd.Series
        Stage 2 validation labels.
    """
    splits = DataSplits(
        data=data,
        label_column=label_column,
        random_state=random_state,
        train_pool_size=train_pool_size,
        val_stage1_size=val_stage1_size,
        val_stage2_size=val_stage2_size
    )
    
    return splits.get_all_splits()
