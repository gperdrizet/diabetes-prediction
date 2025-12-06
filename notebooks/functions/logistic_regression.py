"""Functions for logistic regression model optimization and runtime experiments."""

import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Union, List, Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


class IDColumnDropper(BaseEstimator, TransformerMixin):
    """Drops the 'id' column from the dataframe."""
    
    def __init__(self, id_column='id'):
        self.id_column = id_column
    
    def fit(self, X, y=None):
        """No fitting necessary for dropping columns."""
        return self
    
    def transform(self, X):
        """Drop the ID column if it exists."""
        if isinstance(X, pd.DataFrame):
            if self.id_column in X.columns:
                return X.drop(columns=[self.id_column])
            return X
        else:
            # If not a DataFrame, return as is
            return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips features to a multiple of the interquartile range (IQR)."""
    
    def __init__(self, iqr_multiplier=2.0):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        """Calculate the clipping bounds based on IQR."""

        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.iqr_multiplier * IQR
        self.upper_bounds_ = Q3 + self.iqr_multiplier * IQR

        return self
    
    def transform(self, X):
        """Apply clipping to the data."""

        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class ConstantFeatureRemover(BaseEstimator, TransformerMixin):
    """Removes features with constant values (zero variance)."""
    
    def __init__(self):
        self.constant_features_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
    
    def fit(self, X, y=None):
        """Identify constant-valued features."""
        
        # Calculate variance for each feature
        variances = np.var(X, axis=0)
        
        # Identify constant features (variance close to zero)
        self.constant_features_ = np.where(np.isclose(variances, 0.0))[0]
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.n_features_in_ - len(self.constant_features_)
        
        return self
    
    def transform(self, X):
        """Remove constant-valued features."""
        
        # Create mask for non-constant features
        mask = np.ones(self.n_features_in_, dtype=bool)
        mask[self.constant_features_] = False
        
        return X[:, mask]


def random_search_test(
    train_df_path: str,
    label: str,
    pipeline: Pipeline,
    param_distributions: Union[List[Dict[str, Any]], Dict[str, Any]],
    sample_sizes: List[int],
    n_iters: List[int],
    random_state: int,
    n_replicates: int = 3,
    n_jobs: int = 3,
    data_dir: Union[str, Path] = '../data',
    results_filename: str = 'runtime_experiment_results.csv'
) -> Tuple[pd.DataFrame, int]:
    """
    Run runtime and memory experiments with different sample sizes and iteration counts.
    
    Measures both execution time and peak memory usage for RandomizedSearchCV
    with various sample sizes and iteration counts. Runs multiple replicates of each
    condition to capture variability.
    
    Parameters
    ----------
    train_df_path : str
        Path to the training data CSV file.
    label : str
        Name of the target column.
    pipeline : sklearn.pipeline.Pipeline
        The sklearn pipeline to optimize.
    param_distributions : list or dict
        Parameter distributions for RandomizedSearchCV.
    sample_sizes : list
        List of sample sizes to test.
    n_iters : list
        List of iteration counts to test.
    random_state : int
        Base random state for reproducibility.
    n_replicates : int, optional
        Number of replicates to run for each condition (default: 3).
    n_jobs : int, optional
        Number of parallel jobs for RandomizedSearchCV (default: 3).
    data_dir : str or Path, optional
        Directory to save results (default: '../data').
    results_filename : str, optional
        Filename for saving results (default: 'runtime_experiment_results.csv').
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing experiment results with columns:
        sample_size, n_iter, replicate, runtime_seconds, peak_memory_mb.
    int
        Full dataset size for later predictions.
    """

    # Store results
    results = []
    
    # Get full dataset size for later prediction
    full_dataset_size = len(pd.read_csv(train_df_path))
    
    # Run experiments
    for sample_size in sample_sizes:
        for n_iter in n_iters:
            for replicate in range(1, n_replicates + 1):
                
                # Use different random state for each replicate to get different samples
                replicate_random_state = random_state + replicate
                
                # Sample the data
                train_sample = pd.read_csv(train_df_path).drop(columns=['id']).sample(
                    n=sample_size, 
                    random_state=replicate_random_state
                )
                
                X_sample = train_sample.drop(columns=[label])
                y_sample = train_sample[label]
                
                # Create a simple search
                simple_search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    scoring='roc_auc',
                    n_jobs=n_jobs,
                    cv=3,
                    random_state=replicate_random_state,
                    verbose=0
                )
                
                # Start memory tracking
                tracemalloc.start()
                
                # Time the fitting
                start_time = time.time()
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    simple_search.fit(X_sample, y_sample)
                
                elapsed_time = time.time() - start_time
                
                # Get peak memory usage
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Convert to MB
                peak_memory_mb = peak_memory / (1024 * 1024)
                
                results.append({
                    'sample_size': sample_size,
                    'n_iter': n_iter,
                    'replicate': replicate,
                    'runtime_seconds': elapsed_time,
                    'peak_memory_mb': peak_memory_mb
                })
                
                print(f"Sample size: {sample_size:5d} | Iterations: {n_iter:3d} | Replicate: {replicate} | Time: {elapsed_time:6.2f}s | Memory: {peak_memory_mb:7.2f} MB", end='\r')
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to disk
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_path = data_dir / results_filename
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return results_df, full_dataset_size


def build_runtime_model(results_df: pd.DataFrame) -> Tuple[LinearRegression, Dict[str, float], pd.DataFrame]:
    """
    Build a linear regression model to predict runtime.
    
    Simple model: runtime = intercept + c1 × (sample_size/1000)² × n_iter
    
    This captures the key scaling: runtime grows quadratically with sample size
    (due to PCA) and linearly with iterations.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: sample_size, n_iter, runtime_seconds.
    
    Returns
    -------
    sklearn.linear_model.LinearRegression
        Fitted runtime prediction model (with fit_intercept=True).
    dict
        Dictionary containing model statistics:
        - intercept: Model intercept
        - quadratic_interaction_coef: Coefficient for (sample_size/1000)² × n_iter
        - r2_score: R² score of the model
        - rmse: Root mean squared error
    pd.DataFrame
        Results DataFrame with added columns: predicted_runtime, residual.
    """

    # Create quadratic interaction feature: (sample_size in thousands)² × n_iter
    results_with_features = results_df.copy()
    sample_size_thousands = results_df['sample_size'] / 1000.0
    results_with_features['quadratic_interaction'] = (sample_size_thousands ** 2) * results_df['n_iter']
    
    # Extract features and target
    X_train_runtime = results_with_features[['quadratic_interaction']].values
    y_train_runtime = results_with_features['runtime_seconds'].values
    
    # Fit linear model with intercept
    runtime_model = LinearRegression(fit_intercept=True)
    runtime_model.fit(X_train_runtime, y_train_runtime)
    
    # Calculate statistics
    r2_score = runtime_model.score(X_train_runtime, y_train_runtime)
    predictions = runtime_model.predict(X_train_runtime)
    rmse = np.sqrt(np.mean((y_train_runtime - predictions) ** 2))
    
    stats = {
        'intercept': runtime_model.intercept_,
        'quadratic_interaction_coef': runtime_model.coef_[0],
        'r2_score': r2_score,
        'rmse': rmse
    }
    
    # Add predictions and residuals to results
    results_with_predictions = results_with_features.copy()
    results_with_predictions['predicted_runtime'] = predictions
    results_with_predictions['residual'] = (
        results_with_predictions['runtime_seconds'] - 
        results_with_predictions['predicted_runtime']
    )
    
    return runtime_model, stats, results_with_predictions


def calculate_optimal_iterations(
    runtime_model: LinearRegression, 
    sample_size: int, 
    runtime_limit_minutes: float
) -> Tuple[int, Dict[str, float]]:
    """
    Calculate optimal number of iterations to fit within runtime limit.
    
    Model form: runtime = intercept + c × (S/1000)² × N
    where S = sample_size, N = n_iter
    
    Solving for N: N = (runtime - intercept) / [c × (S/1000)²]
    
    Parameters
    ----------
    runtime_model : sklearn.linear_model.LinearRegression
        Fitted runtime prediction model.
    sample_size : int
        Size of the dataset to use for training.
    runtime_limit_minutes : float
        Maximum runtime allowed in minutes.
    
    Returns
    -------
    int
        Optimal number of iterations (rounded down conservatively).
    dict
        Dictionary containing calculation details.
    """

    s = sample_size
    t = runtime_limit_minutes * 60
    b = runtime_model.intercept_
    m = runtime_model.coef_[0]

    n = ((t-b)*(1/m)) / ((s/1000)**2)
    
    # Round down to be conservative, but ensure at least 1 iteration
    n = max(1, int(np.floor(n)))
    
    # Predict actual runtime
    feature = ((s/1000) ** 2) * n
    predicted_runtime = runtime_model.predict([[feature]])[0]
    
    return n, {
        'runtime_limit_seconds': t,
        'sample_size': s,
        'optimal_n_iter': n,
        'predicted_runtime_seconds': predicted_runtime,
        'predicted_runtime_minutes': predicted_runtime / 60,
    }


def calculate_optimal_sample_size(
    memory_model: LinearRegression,
    memory_limit_gb: float
) -> Tuple[int, Dict[str, float]]:
    """
    Calculate optimal sample size to fit within memory limit.
    
    Model form: memory = intercept + c × (S/1000)²
    where S = sample_size (in samples)
    
    Solving for S: S = 1000 × sqrt((memory - intercept) / c)
    
    Parameters
    ----------
    memory_model : sklearn.linear_model.LinearRegression
        Fitted memory prediction model.
    memory_limit_gb : float
        Maximum memory allowed in gigabytes.
    
    Returns
    -------
    int
        Optimal sample size (rounded down conservatively).
    dict
        Dictionary containing calculation details.
    """
    
    # Convert GB to MB
    memory_limit_mb = memory_limit_gb * 1024
    
    b = memory_model.intercept_
    c = memory_model.coef_[0]
    
    # Solve for sample size: S = 1000 × sqrt((M - b) / c)
    # where M is memory limit in MB
    s = 1000 * np.sqrt((memory_limit_mb - b) / c)
    
    # Round down to be conservative, but ensure at least 1 sample
    s = max(1, int(np.floor(s)))
    
    # Predict actual memory usage
    feature = (s / 1000) ** 2
    predicted_memory_mb = memory_model.predict([[feature]])[0]
    predicted_memory_gb = predicted_memory_mb / 1024
    
    return s, {
        'memory_limit_gb': memory_limit_gb,
        'memory_limit_mb': memory_limit_mb,
        'optimal_sample_size': s,
        'predicted_memory_mb': predicted_memory_mb,
        'predicted_memory_gb': predicted_memory_gb,
    }


def plot_runtime_model(
    runtime_model: LinearRegression, 
    model_stats: Dict[str, float], 
    results_df: pd.DataFrame, 
    results_with_predictions: pd.DataFrame, 
    sample_sizes: List[int]
) -> None:
    """
    Create a three-panel visualization of the runtime model.
    
    Parameters
    ----------
    runtime_model : sklearn.linear_model.LinearRegression
        Fitted runtime prediction model.
    model_stats : dict
        Dictionary containing model statistics (intercept, interaction_coef, r2_score).
    results_df : pd.DataFrame
        DataFrame containing experimental results with columns: sample_size, n_iter, runtime_seconds.
    results_with_predictions : pd.DataFrame
        DataFrame containing predictions and residuals.
    sample_sizes : list
        List of sample sizes used in the experiment.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Add figure suptitle with metrics
    fig.suptitle(f"RandomizedSearchCV runtime model\nR² = {model_stats['r2_score']:.4f}, RMSE = {model_stats['rmse']:.2f} sec.")
    
    # Plot 1: Runtime vs iterations for each sample size
    ax1 = axes[0]
    ax1.set_title('Runtime vs iterations & sample size')
    
    for i, sample_size in enumerate(sorted(sample_sizes)):
        data = results_df[results_df['sample_size'] == sample_size]
        ax1.scatter(
            data['n_iter'], 
            data['runtime_seconds'], 
            label=f'{sample_size:,} samples'
        )
        
        # Add trend line from model: runtime = intercept + c × (S/1000)² × N
        n_iter_range = np.linspace(data['n_iter'].min(), data['n_iter'].max(), 100)
        S_k = sample_size / 1000.0  # Convert to thousands
        quadratic_interactions = (S_k ** 2) * n_iter_range
        predicted_runtime = runtime_model.predict(quadratic_interactions.reshape(-1, 1))
        ax1.plot(n_iter_range, predicted_runtime, '--', alpha=0.7)
    
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.legend()
    
    # Plot 2: Actual vs Predicted Runtime
    ax2 = axes[1]
    ax2.set_title('Actual vs predicted runtime')
    ax2.scatter(
        results_with_predictions['runtime_seconds'],
        results_with_predictions['predicted_runtime'],
        c='black',
        label='Data'
    )
    
    # Add perfect prediction line
    min_val = min(
        results_with_predictions['runtime_seconds'].min(), 
        results_with_predictions['predicted_runtime'].min()
    )
    max_val = max(
        results_with_predictions['runtime_seconds'].max(), 
        results_with_predictions['predicted_runtime'].max()
    )
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax2.set_xlabel('Actual runtime (seconds)')
    ax2.set_ylabel('Predicted runtime (seconds)')
    ax2.legend()
    
    # Plot 3: Residuals
    ax3 = axes[2]
    ax3.set_title('Prediction residuals')
    ax3.scatter(
        results_with_predictions['predicted_runtime'],
        results_with_predictions['residual'],
        c='black'
    )
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted runtime (seconds)')
    ax3.set_ylabel('Residual (actual - predicted)')
    
    plt.tight_layout()


def build_memory_model(results_df: pd.DataFrame) -> Tuple[LinearRegression, Dict[str, float], pd.DataFrame]:
    """
    Build a linear regression model to predict peak memory usage.
    
    Simple model: memory = intercept + c × (sample_size/1000)²
    
    This captures the key scaling: memory grows quadratically with sample size
    (due to PCA), but is independent of the number of iterations.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: sample_size, peak_memory_mb.
    
    Returns
    -------
    sklearn.linear_model.LinearRegression
        Fitted memory prediction model (with fit_intercept=True).
    dict
        Dictionary containing model statistics:
        - intercept: Model intercept
        - quadratic_coef: Coefficient for (sample_size/1000)²
        - r2_score: R² score of the model
        - rmse: Root mean squared error
    pd.DataFrame
        Results DataFrame with added columns: predicted_memory, residual.
    """

    # Create quadratic feature: (sample_size in thousands)²
    results_with_features = results_df.copy()
    sample_size_thousands = results_df['sample_size'] / 1000.0
    results_with_features['sample_size_squared'] = sample_size_thousands ** 2
    
    # Extract features and target
    X_train_memory = results_with_features[['sample_size_squared']].values
    y_train_memory = results_with_features['peak_memory_mb'].values
    
    # Fit linear model with intercept
    memory_model = LinearRegression(fit_intercept=True)
    memory_model.fit(X_train_memory, y_train_memory)
    
    # Calculate statistics
    r2_score = memory_model.score(X_train_memory, y_train_memory)
    predictions = memory_model.predict(X_train_memory)
    rmse = np.sqrt(np.mean((y_train_memory - predictions) ** 2))
    
    stats = {
        'intercept': memory_model.intercept_,
        'quadratic_coef': memory_model.coef_[0],
        'r2_score': r2_score,
        'rmse': rmse
    }
    
    # Add predictions and residuals to results
    results_with_predictions = results_with_features.copy()
    results_with_predictions['predicted_memory'] = predictions
    results_with_predictions['residual'] = (
        results_with_predictions['peak_memory_mb'] - 
        results_with_predictions['predicted_memory']
    )
    
    return memory_model, stats, results_with_predictions


def plot_memory_model(
    memory_model: LinearRegression, 
    model_stats: Dict[str, float], 
    results_df: pd.DataFrame, 
    results_with_predictions: pd.DataFrame
) -> None:
    """
    Create a three-panel visualization of the memory model.
    
    Parameters
    ----------
    memory_model : sklearn.linear_model.LinearRegression
        Fitted memory prediction model.
    model_stats : dict
        Dictionary containing model statistics (intercept, interaction_coef, r2_score).
    results_df : pd.DataFrame
        DataFrame containing experimental results with columns: sample_size, n_iter, peak_memory_mb.
    results_with_predictions : pd.DataFrame
        DataFrame containing predictions and residuals.
    sample_sizes : list
        List of sample sizes used in the experiment.
    """

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Add figure suptitle with metrics
    fig.suptitle(f"RandomizedSearchCV memory model\nR² = {model_stats['r2_score']:.4f}, RMSE = {model_stats['rmse']:.2f} MB")
    
    # Plot 1: Memory vs sample size
    ax1 = axes[0]
    ax1.set_title('Memory vs sample size')
    
    # Scatter all data points
    ax1.scatter(
        results_df['sample_size'], 
        results_df['peak_memory_mb'],
        c='black',
        label='Data'
    )
    
    # Add trend line from model: memory = intercept + c × (S/1000)²
    sample_size_range = np.linspace(results_df['sample_size'].min(), results_df['sample_size'].max(), 100)
    S_k = sample_size_range / 1000.0  # Convert to thousands
    sample_size_squared = S_k ** 2
    predicted_memory = memory_model.predict(sample_size_squared.reshape(-1, 1))
    ax1.plot(sample_size_range, predicted_memory, 'r--', linewidth=2, label='Model fit')
    
    ax1.set_xlabel('Sample size')
    ax1.set_ylabel('Peak memory (MB)')
    ax1.legend()
    
    # Plot 2: Actual vs Predicted Memory
    ax2 = axes[1]
    ax2.set_title('Actual vs predicted memory')
    ax2.scatter(
        results_with_predictions['peak_memory_mb'],
        results_with_predictions['predicted_memory'],
        c='black',
        label='Data'
    )
    
    # Add perfect prediction line
    min_val = min(
        results_with_predictions['peak_memory_mb'].min(), 
        results_with_predictions['predicted_memory'].min()
    )
    max_val = max(
        results_with_predictions['peak_memory_mb'].max(), 
        results_with_predictions['predicted_memory'].max()
    )
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax2.set_xlabel('Actual memory (MB)')
    ax2.set_ylabel('Predicted memory (MB)')
    ax2.legend()
    
    # Plot 3: Residuals
    ax3 = axes[2]
    ax3.set_title('Prediction residuals')
    ax3.scatter(
        results_with_predictions['predicted_memory'],
        results_with_predictions['residual'],
        c='black'
    )
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted memory (MB)')
    ax3.set_ylabel('Residual (actual - predicted)')
    
    plt.tight_layout()
