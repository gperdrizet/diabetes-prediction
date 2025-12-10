"""Stage 2 DNN training and evaluation.

This module provides functions for training the Stage 2 DNN with early stopping,
transfer learning, and evaluation.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
from tensorflow.keras import models, callbacks
from sklearn.metrics import roc_auc_score, confusion_matrix


def train_stage2_dnn(
    model: models.Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 15,
    ensemble_id: Optional[str] = None,
    database_logger: Optional[Any] = None
) -> Tuple[models.Sequential, Dict[str, Any]]:
    """Train stage 2 DNN with early stopping and learning rate reduction.
    
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
    ensemble_id : str or None, default=None
        Ensemble ID for logging.
    database_logger : object or None, default=None
        Database module for logging (must have insert_stage2_epoch method).
    
    Returns
    -------
    model : Sequential
        Trained model.
    history : dict
        Training history.
    """
    # Callbacks - monitor val_loss to prevent overfitting
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
    
    # Custom callback for database logging
    if database_logger is not None and ensemble_id is not None:
        class DatabaseLoggingCallback(callbacks.Callback):
            """Log training metrics to database."""
            
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
                    database_logger.insert_stage2_epoch(epoch_data)
                except Exception as e:
                    print(f"Warning: Failed to log epoch {epoch}: {e}")
        
        callback_list.append(DatabaseLoggingCallback())
    
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


def generate_stage2_training_data(
    stage1_models: List[Any],
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Stage 2 training data from Stage 1 model predictions.
    
    Parameters
    ----------
    stage1_models : list of pipelines
        List of stage 1 sklearn pipelines.
    X : np.ndarray or pd.DataFrame
        Input features.
    y : np.ndarray or pd.Series
        Labels.
    
    Returns
    -------
    X_stage2 : np.ndarray
        Stage 1 predictions, shape (n_samples, n_models).
    y_stage2 : np.ndarray
        Labels.
    """
    # Generate stage 1 predictions
    stage1_predictions = []
    
    for model in stage1_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X)[:, 1]
        else:
            pred = model.decision_function(X)
        stage1_predictions.append(pred)
    
    # Stack predictions
    X_stage2 = np.column_stack(stage1_predictions)
    
    # Convert labels if needed
    y_stage2 = y.values if hasattr(y, 'values') else y
    
    return X_stage2, y_stage2
