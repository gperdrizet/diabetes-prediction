"""Stage 2 DNN hyperparameter optimization.

This module provides functions for optimizing Stage 2 DNN hyperparameters
using Keras Tuner with a focused search space.
"""

import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras import callbacks

from .model import build_for_tuning


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_trials: int = 8,
    executions_per_trial: int = 1,
    max_epochs: int = 50,
    project_name: str = 'stage2_tuning',
    directory: Path = None
) -> Tuple[Dict[str, Any], float]:
    """Optimize stage 2 DNN hyperparameters using Keras Tuner.
    
    SEARCH SPACE (anti-overfitting focus):
    - Architecture types: funnel, constant, pyramid (3 types)
    - Layers: 1-3 hidden layers
    - Base units: 16, 32, 64, 128
    - Dropout: continuous 0.2-0.7 (strong regularization)
    - L2 regularization: log continuous 1e-4 to 1e-2
    - Learning rate: log continuous 1e-5 to 1e-4
    - Objective: val_loss (not val_auc) to prevent overfitting
    
    NOTE: This function requires keras_tuner to be installed.
    Install with: pip install keras-tuner
    
    Parameters
    ----------
    X_train : np.ndarray
        Training predictions from stage 1 models, shape (n_samples, n_models).
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray
        Validation predictions from stage 1 models.
    y_val : np.ndarray
        Validation labels.
    max_trials : int, default=8
        Number of hyperparameter combinations to try.
    executions_per_trial : int, default=1
        Number of training runs per combination.
    max_epochs : int, default=50
        Maximum epochs per trial (early stopping will trigger earlier).
    project_name : str, default='stage2_tuning'
        Name for tuning project.
    directory : Path or None, default=None
        Directory for tuning results.
    
    Returns
    -------
    best_hyperparameters : dict
        Best hyperparameters found (architecture_type, n_layers, base_units,
        dropout, l2_reg, learning_rate).
    best_val_auc : float
        Best validation AUC achieved.
    """
    # Import keras_tuner only when this function is called
    try:
        from keras_tuner import RandomSearch, Objective
    except ImportError:
        raise ImportError(
            "keras_tuner is required for hyperparameter optimization. "
            "Install with: pip install keras-tuner"
        )
    
    # Suppress optimizer loading warning
    warnings.filterwarnings('ignore', message='Skipping variable loading for optimizer')
    
    n_models = X_train.shape[1]
    
    if directory is None:
        directory = Path('../models/keras_tuner')
    directory.mkdir(parents=True, exist_ok=True)
    
    def build_model_wrapper(hp):
        """Wrapper for Keras Tuner."""
        model, _ = build_for_tuning(hp, n_models)
        return model
    
    # Create tuner
    tuner = RandomSearch(
        build_model_wrapper,
        objective=Objective('val_loss', direction='min'),  # Optimize val_loss!
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=str(directory),
        project_name=project_name,
        overwrite=True
    )
    
    # Early stopping and learning rate reduction callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Run tuning
    print(f"  Starting Keras Tuner search...")
    print(f"  Settings: {max_trials} trials × {executions_per_trial} executions = {max_trials * executions_per_trial} trainings")
    
    start_time = time.time()
    
    tuner.search(
        X_train, y_train,
        epochs=max_epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
        batch_size=64  # Smaller batch for better generalization
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"  Tuner search complete in {elapsed_time/60:.1f} minutes!")
    
    # Get best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Evaluate best model
    y_pred_val = best_model.predict(X_val, verbose=0).flatten()
    best_val_auc = roc_auc_score(y_val, y_pred_val)
    
    # Extract hyperparameters
    best_hyperparameters = {
        'architecture_type': best_hps.get('architecture_type'),
        'n_layers': best_hps.get('n_layers'),
        'base_units': best_hps.get('base_units'),
        'dropout': best_hps.get('dropout'),
        'l2_reg': best_hps.get('l2_reg'),
        'learning_rate': best_hps.get('learning_rate')
    }
    
    print(f"  Best validation AUC: {best_val_auc:.6f}")
    print(f"  Best hyperparameters:")
    print(f"    Architecture: {best_hyperparameters['architecture_type']}")
    print(f"    Layers: {best_hyperparameters['n_layers']}")
    print(f"    Base units: {best_hyperparameters['base_units']}")
    print(f"    Dropout: {best_hyperparameters['dropout']:.3f}")
    print(f"    L2 reg: {best_hyperparameters['l2_reg']:.6f}")
    print(f"    Learning rate: {best_hyperparameters['learning_rate']:.6f}")
    
    return best_hyperparameters, best_val_auc
