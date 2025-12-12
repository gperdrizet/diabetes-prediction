"""Stage 2 DNN meta-learner functions for ensemble training and optimization.

This module provides functions for building, training, and optimizing the stage 2
deep neural network that combines predictions from stage 1 models.
"""

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import roc_auc_score, confusion_matrix

# Use absolute imports instead of relative imports
import ensemble_database

# Note: keras_tuner only needed for optimize_stage2_hyperparameters function
# Import on-demand if that function is used


def build_stage2_dnn(
    n_models: int,
    config: dict = None,
    n_layers: int = 2,
    units_per_layer: int = 128,
    dropout: float = 0.3,
    batch_norm: bool = True,
    activation: str = 'relu',
    learning_rate: float = 0.001
) -> models.Sequential:
    """Build stage 2 DNN meta-learner.
    
    Parameters
    ----------
    n_models : int
        Number of stage 1 models (input dimension).
    config : dict, optional
        Config dict with architecture and training specs.
        If provided, overrides other parameters.
    n_layers : int, default=2
        Number of hidden layers (ignored if config provided).
    units_per_layer : int, default=128
        Units per hidden layer (ignored if config provided).
    dropout : float, default=0.3
        Dropout rate (ignored if config provided).
    batch_norm : bool, default=True
        Whether to use batch normalization (ignored if config provided).
    activation : str, default='relu'
        Activation function (ignored if config provided).
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer (ignored if config provided).
    
    Returns
    -------
    model : Sequential
        Compiled Keras model.
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(n_models,)))
    
    if config is not None:
        # Build from config
        arch = config['architecture']
        train_cfg = config['training']
        
        # Hidden layers from config
        for layer_cfg in arch['hidden_layers']:
            model.add(layers.Dense(
                layer_cfg['units'],
                activation=layer_cfg['activation']
            ))
            model.add(layers.Dropout(layer_cfg['dropout']))
        
        # Output layer
        output_cfg = arch['output']
        model.add(layers.Dense(
            output_cfg['units'],
            activation=output_cfg['activation']
        ))
        
        # Compile with config settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=train_cfg['learning_rate']),
            loss=train_cfg['loss'],
            metrics=[keras.metrics.AUC(name='auc')]
        )
    else:
        # Build from legacy parameters
        for i in range(n_layers):
            model.add(layers.Dense(units_per_layer, activation=activation))
            
            if batch_norm:
                model.add(layers.BatchNormalization())
            
            model.add(layers.Dropout(dropout))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[keras.metrics.AUC(name='auc')]
        )
    
    return model


def build_architecture_units(n_models: int, arch_type: str, base_units: int, n_layers: int) -> List[int]:
    """Generate layer unit configuration based on architecture type.
    
    Parameters
    ----------
    n_models : int
        Number of input models (input dimension).
    arch_type : str
        Architecture type: 'uniform', 'pyramid', or 'funnel'.
    base_units : int
        Base number of units for scaling.
    n_layers : int
        Number of hidden layers.
    
    Returns
    -------
    units_per_layer : list of int
        Units for each layer.
    """
    if arch_type == 'uniform':
        # All layers same size
        return [base_units] * n_layers
    
    elif arch_type == 'pyramid':
        # Decreasing by half each layer
        units = []
        current = base_units
        for _ in range(n_layers):
            units.append(max(8, current))  # Minimum 8 units
            current = current // 2
        return units
    
    elif arch_type == 'funnel':
        # Start at input size, decrease to base_units
        if n_layers == 1:
            return [base_units]
        units = np.linspace(n_models, base_units, n_layers, dtype=int)
        return [max(8, int(u)) for u in units]  # Minimum 8 units
    
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def build_regularized_model_for_tuning(hp, n_models):
    """Build model with strong regularization for Keras Tuner (internal use).
    
    This is the hyperparameter model builder used during optimization.
    After optimization, use build_model_from_config() to rebuild the winner.
    
    Parameters
    ----------
    hp : HyperParameters
        Keras Tuner hyperparameters object.
    n_models : int
        Number of stage 1 models (input dimension).
    
    Returns
    -------
    model : Sequential
        Compiled Keras model.
    metadata : dict
        Hyperparameter values for logging.
    """
    # Expanded search space with regularization
    arch_type = hp.Choice('architecture_type', values=['funnel', 'constant', 'pyramid'])
    n_layers = hp.Int('n_layers', min_value=1, max_value=3)
    base_units = hp.Choice('base_units', values=[16, 32, 64, 128])
    
    # Strong regularization
    dropout = hp.Float('dropout', min_value=0.2, max_value=0.7)
    l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log')
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, sampling='log')
    
    # Generate layer units based on architecture type
    if arch_type == 'funnel':
        # Decreasing units: [base, base//2, base//4, ...]
        if n_layers == 1:
            units_per_layer = [base_units]
        elif n_layers == 2:
            units_per_layer = [base_units, base_units // 2]
        else:  # n_layers == 3
            units_per_layer = [base_units, base_units // 2, base_units // 4]
    
    elif arch_type == 'constant':
        # Same units per layer: [base, base, base, ...]
        units_per_layer = [base_units] * n_layers
    
    elif arch_type == 'pyramid':
        # Increasing then decreasing: [base//2, base, base//2] or similar
        if n_layers == 1:
            units_per_layer = [base_units]
        elif n_layers == 2:
            units_per_layer = [base_units // 2, base_units]
        else:  # n_layers == 3
            units_per_layer = [base_units // 2, base_units, base_units // 2]
    
    # Build model
    from tensorflow.keras import regularizers
    model = models.Sequential()
    model.add(layers.Input(shape=(n_models,)))
    
    for units in units_per_layer:
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')]
    )
    
    return model, {
        'architecture_type': arch_type,
        'n_layers': n_layers,
        'base_units': base_units,
        'units_per_layer': units_per_layer,
        'dropout': dropout,
        'l2_reg': l2_reg,
        'learning_rate': learning_rate
    }


def build_model_from_config(config: dict, n_models: int) -> Tuple[models.Sequential, List[int]]:
    """Build model from hyperparameter configuration (for retraining).
    
    Use this to rebuild the winning model after hyperparameter optimization.
    
    Parameters
    ----------
    config : dict
        Configuration with keys: architecture_type, n_layers, base_units,
        dropout, l2_reg, learning_rate.
    n_models : int
        Number of stage 1 models (input dimension).
    
    Returns
    -------
    model : Sequential
        Compiled Keras model.
    units_per_layer : list of int
        Units for each layer (for inspection).
    """
    # Extract hyperparameters
    arch_type = config['architecture_type']
    n_layers = config['n_layers']
    base_units = config['base_units']
    dropout = config['dropout']
    l2_reg = config['l2_reg']
    learning_rate = config['learning_rate']
    
    # Generate layer units based on architecture type
    if arch_type == 'funnel':
        if n_layers == 1:
            units_per_layer = [base_units]
        elif n_layers == 2:
            units_per_layer = [base_units, base_units // 2]
        else:  # n_layers == 3
            units_per_layer = [base_units, base_units // 2, base_units // 4]
    
    elif arch_type == 'constant':
        units_per_layer = [base_units] * n_layers
    
    elif arch_type == 'pyramid':
        if n_layers == 1:
            units_per_layer = [base_units]
        elif n_layers == 2:
            units_per_layer = [base_units // 2, base_units]
        else:  # n_layers == 3
            units_per_layer = [base_units // 2, base_units, base_units // 2]
    
    # Build model
    from tensorflow.keras import regularizers
    model = models.Sequential()
    model.add(layers.Input(shape=(n_models,)))
    
    for units in units_per_layer:
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc'), 'accuracy']
    )
    
    return model, units_per_layer


def optimize_stage2_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_trials: int = 8,
    executions_per_trial: int = 1,
    max_epochs: int = 50,
    project_name: str = 'stage2_regularized_tuning',
    directory: Path = None
) -> Tuple[Dict[str, Any], float]:
    """Optimize stage 2 DNN hyperparameters using Keras Tuner with regularization.
    
    UPDATED SEARCH SPACE (anti-overfitting focus):
    - Architecture types: funnel, constant, pyramid (3 types)
    - Layers: 1-3 hidden layers
    - Base units: 16, 32, 64, 128 (smaller for regularization)
    - Dropout: continuous 0.2-0.7 (stronger regularization)
    - L2 regularization: log continuous 1e-4 to 1e-2
    - Learning rate: log continuous 1e-5 to 1e-4 (lower for stability)
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
    project_name : str, default='stage2_regularized_tuning'
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
        import warnings
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
        model, hps = build_regularized_model_for_tuning(hp, n_models)
        return model
    
    # Create tuner
    tuner = RandomSearch(
        build_model_wrapper,
        objective=Objective('val_loss', direction='min'),  # Optimize val_loss, not AUC!
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
    print(f"  Starting Keras Tuner search with regularization...")
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


def train_stage2_dnn(
    model: models.Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 15,
    log_path: Optional[Path] = None,
    iteration: Optional[int] = None
) -> Tuple[models.Sequential, Dict[str, Any]]:
    """Train stage 2 DNN with early stopping and learning rate reduction.
    
    UPDATED DEFAULTS (anti-overfitting focus):
    - Monitor val_loss (not val_auc) to prevent overfitting
    - Patience = 15 (more tolerance for val_loss fluctuations)
    - Batch size = 64 (smaller for better generalization)
    - Max epochs = 200 (early stopping will trigger ~15-30 epochs)
    - ReduceLROnPlateau for adaptive learning rate
    
    Parameters
    ----------
    model : Sequential
        Keras model to train.
    X_train : np.ndarray
        Training predictions from stage 1 models.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray
        Validation predictions from stage 1 models.
    y_val : np.ndarray
        Validation labels.
    epochs : int, default=200
        Maximum number of epochs.
    batch_size : int, default=64
        Batch size.
    patience : int, default=15
        Early stopping patience (for val_loss).
    log_path : Path or None, default=None
        Path to log training metrics (used as ensemble_id string).
    iteration : int or None, default=None
        Current iteration (for logging).
    
    Returns
    -------
    model : Sequential
        Trained model.
    history : dict
        Training history.
    """
    # Callbacks - monitor val_loss instead of val_auc
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            restore_best_weights=True,
            verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            mode='min',
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    # Custom callback for SQLite logging
    if log_path is not None:
        # log_path parameter now used to pass ensemble_id as string
        ensemble_id = str(log_path) if log_path else f"iter_{iteration}"
        
        class LoggingCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                try:
                    epoch_data = {
                        'timestamp': datetime.now().isoformat(),
                        'ensemble_id': ensemble_id,
                        'epoch': epoch,
                        'train_loss': float(logs.get('loss', 0)),
                        'val_loss': float(logs.get('val_loss', 0)),
                        'train_auc': float(logs.get('auc', 0)),
                        'val_auc': float(logs.get('val_auc', 0))
                    }
                    ensemble_database.insert_stage2_epoch(epoch_data)
                except Exception as e:
                    print(f"Warning: Failed to log epoch {epoch}: {e}")
        
        callback_list.append(LoggingCallback())
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=0
    )
    
    return model, history.history


def evaluate_ensemble(
    stage1_models: List[Any],
    stage2_model: models.Sequential,
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """Evaluate ensemble performance (stage 1 + stage 2).
    
    Parameters
    ----------
    stage1_models : list of pipelines
        List of stage 1 sklearn pipelines.
    stage2_model : Sequential
        Stage 2 Keras model.
    X : np.ndarray or pd.DataFrame
        Input features.
    y : np.ndarray or pd.Series
        True labels.
    
    Returns
    -------
    roc_auc : float
        ROC-AUC score on the validation set.
    """
    # Generate stage 1 predictions
    stage1_predictions = []
    
    for model in stage1_models:
        try:
            # Get probability predictions
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
            else:
                pred_proba = model.decision_function(X)
            
            stage1_predictions.append(pred_proba)
        except Exception as e:
            print(f"Error generating predictions: {e}")
            # Use zeros if prediction fails
            stage1_predictions.append(np.zeros(len(X)))
    
    # Stack predictions
    stage1_predictions = np.column_stack(stage1_predictions)
    
    # Get stage 2 predictions
    stage2_predictions = stage2_model.predict(stage1_predictions, verbose=0).flatten()
    
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(y, stage2_predictions)
    
    return roc_auc


def evaluate_ensemble_with_cm(
    stage1_models: List[Any],
    stage2_model: models.Sequential,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, int, int, int, int]:
    """Evaluate ensemble performance with confusion matrix.
    
    Parameters
    ----------
    stage1_models : list of pipelines
        List of stage 1 sklearn pipelines.
    stage2_model : Sequential
        Stage 2 Keras model.
    X : np.ndarray or pd.DataFrame
        Input features.
    y : np.ndarray or pd.Series
        True labels.
    threshold : float, default=0.5
        Classification threshold for confusion matrix.
    
    Returns
    -------
    roc_auc : float
        ROC-AUC score on the validation set.
    tn : int
        True negatives.
    fp : int
        False positives.
    fn : int
        False negatives.
    tp : int
        True positives.
    """
    # Generate stage 1 predictions
    stage1_predictions = []
    
    for model in stage1_models:
        try:
            # Get probability predictions
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
            else:
                pred_proba = model.decision_function(X)
            
            stage1_predictions.append(pred_proba)
        except Exception as e:
            print(f"Error generating predictions: {e}")
            # Use zeros if prediction fails
            stage1_predictions.append(np.zeros(len(X)))
    
    # Stack predictions
    stage1_predictions = np.column_stack(stage1_predictions)
    
    # Get stage 2 predictions
    stage2_predictions = stage2_model.predict(stage1_predictions, verbose=0).flatten()
    
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(y, stage2_predictions)
    
    # Calculate confusion matrix
    y_pred_binary = (stage2_predictions >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred_binary)
    
    # Extract values: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    return roc_auc, int(tn), int(fp), int(fn), int(tp)


def save_checkpoint(
    checkpoint_path: Path,
    ensemble_models: List[Any],
    stage2_model: models.Sequential,
    iteration: int,
    temperature: float,
    best_score: float,
    acceptance_history: List[bool],
    metadata: Dict[str, Any]
) -> None:
    """Save ensemble checkpoint for resuming training.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to save checkpoint.
    ensemble_models : list
        List of stage 1 models.
    stage2_model : Sequential
        Stage 2 DNN model.
    iteration : int
        Current iteration.
    temperature : float
        Current temperature.
    best_score : float
        Best ensemble ROC-AUC score.
    acceptance_history : list
        Recent acceptance decisions.
    metadata : dict
        Additional metadata.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save stage 2 model separately
    stage2_path = checkpoint_path.parent / 'stage2_checkpoint.h5'
    stage2_model.save(stage2_path)
    
    # Save checkpoint data
    checkpoint_data = {
        'iteration': iteration,
        'temperature': temperature,
        'best_score': best_score,
        'acceptance_history': acceptance_history,
        'ensemble_size': len(ensemble_models),
        'metadata': metadata,
        'ensemble_models': ensemble_models,
        'stage2_model_path': str(stage2_path)
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load ensemble checkpoint to resume training.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file.
    
    Returns
    -------
    checkpoint_data : dict
        Dictionary containing all checkpoint data.
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Load stage 2 model
    stage2_path = Path(checkpoint_data['stage2_model_path'])
    if stage2_path.exists():
        checkpoint_data['stage2_model'] = keras.models.load_model(stage2_path)
    else:
        print(f"Warning: Stage 2 model not found at {stage2_path}")
        checkpoint_data['stage2_model'] = None
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"  Iteration: {checkpoint_data['iteration']}")
    print(f"  Ensemble size: {checkpoint_data['ensemble_size']}")
    print(f"  Best score: {checkpoint_data['best_score']:.6f}")
    
    return checkpoint_data


def log_stage2_performance(
    log_path: Path,
    iteration: int,
    fold: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_auc: float,
    val_auc: float,
    learning_rate: float
) -> None:
    """Log stage 2 training performance to JSONL file.
    
    Parameters
    ----------
    log_path : Path
        Path to log file.
    iteration : int
        Current iteration.
    fold : int
        Current fold.
    epoch : int
        Current epoch.
    train_loss : float
        Training loss.
    val_loss : float
        Validation loss.
    train_auc : float
        Training AUC.
    val_auc : float
        Validation AUC.
    learning_rate : float
        Learning rate.
    """
    log_entry = {
        'iteration': iteration,
        'fold': fold,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'learning_rate': learning_rate,
        'timestamp': time.time()
    }
    
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
