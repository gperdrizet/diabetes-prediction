"""Data sampling for ensemble diversity.

Provides stratified row sampling and random column sampling to create
diverse training sets for Stage 1 models.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ensemble.config import SamplingConfig


class DataSampler:
    """Handles row and column sampling for ensemble diversity.
    
    Each model in the ensemble is trained on a random subsample of both
    rows (stratified by target) and columns to increase diversity and
    reduce overfitting.
    
    Attributes:
        config: SamplingConfig with row/column sample ranges
        random_state: Random seed for reproducibility
        rng: NumPy random number generator
    
    Example:
        >>> from ensemble.config import SamplingConfig
        >>> config = SamplingConfig(
        ...     row_sample_range=(0.10, 0.40),
        ...     column_sample_range=(0.30, 0.70)
        ... )
        >>> sampler = DataSampler(config, random_state=42)
        >>> X_sample, y_sample = sampler.sample_rows(X_train, y_train)
        >>> columns = sampler.sample_columns(X_sample)
        >>> X_final = X_sample[columns]
    """
    
    def __init__(self, config: SamplingConfig, random_state: int):
        """Initialize the data sampler.
        
        Args:
            config: SamplingConfig with sampling ranges
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def sample_rows(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Sample rows with stratification by target variable.
        
        Samples a random fraction of rows (within configured range) while
        maintaining the class distribution of the target variable.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_sample, y_sample) with stratified sampling
            
        Example:
            >>> X_sample, y_sample = sampler.sample_rows(X_train, y_train)
            >>> print(f"Sampled {len(X_sample)} of {len(X_train)} rows")
        """
        # Choose random sample fraction within configured range
        row_min, row_max = self.config.row_sample_range
        sample_fraction = self.rng.uniform(row_min, row_max)
        
        # Stratified sampling to maintain class distribution
        X_sample, _, y_sample, _ = train_test_split(
            X, y,
            train_size=sample_fraction,
            stratify=y,
            random_state=self.rng.randint(0, 2**31)
        )
        
        return X_sample, y_sample
    
    def sample_columns(self, X: pd.DataFrame) -> List[str]:
        """Sample columns randomly without replacement.
        
        Samples a random fraction of columns (within configured range)
        to create diverse feature subsets for each model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of column names to use
            
        Example:
            >>> columns = sampler.sample_columns(X_train)
            >>> X_subset = X_train[columns]
            >>> print(f"Selected {len(columns)} of {X_train.shape[1]} columns")
        """
        # Choose random sample fraction within configured range
        col_min, col_max = self.config.column_sample_range
        sample_fraction = self.rng.uniform(col_min, col_max)
        
        # Calculate number of columns to sample
        n_columns = X.shape[1]
        n_sample = max(1, int(n_columns * sample_fraction))
        
        # Random column selection without replacement
        selected_columns = self.rng.choice(
            X.columns,
            size=n_sample,
            replace=False
        ).tolist()
        
        return selected_columns
    
    def sample_both(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Sample both rows and columns in one call.
        
        Convenience method that combines row and column sampling.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_sample, y_sample) with both row and column sampling
            
        Example:
            >>> X_sample, y_sample = sampler.sample_both(X_train, y_train)
            >>> print(f"Sampled shape: {X_sample.shape}")
        """
        # First sample rows (stratified)
        X_row_sample, y_sample = self.sample_rows(X, y)
        
        # Then sample columns
        selected_columns = self.sample_columns(X_row_sample)
        X_final = X_row_sample[selected_columns]
        
        return X_final, y_sample
    
    def get_sample_info(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Get information about what would be sampled without actually sampling.
        
        Useful for debugging and understanding sampling behavior.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dict with sampling information
            
        Example:
            >>> info = sampler.get_sample_info(X_train, y_train)
            >>> print(f"Will sample {info['row_pct']:.1%} rows")
        """
        row_min, row_max = self.config.row_sample_range
        col_min, col_max = self.config.column_sample_range
        
        # Expected values (midpoint of ranges)
        expected_row_pct = (row_min + row_max) / 2
        expected_col_pct = (col_min + col_max) / 2
        
        expected_rows = int(len(X) * expected_row_pct)
        expected_cols = int(X.shape[1] * expected_col_pct)
        
        return {
            'row_range': (row_min, row_max),
            'col_range': (col_min, col_max),
            'expected_row_pct': expected_row_pct,
            'expected_col_pct': expected_col_pct,
            'expected_rows': expected_rows,
            'expected_cols': expected_cols,
            'original_shape': X.shape,
            'expected_shape': (expected_rows, expected_cols)
        }
