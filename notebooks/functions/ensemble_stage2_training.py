"""
Stage 2 DNN training and management.

Handles DNN training with transfer learning for expanding ensembles.
"""

import numpy as np
import psutil
import os
from datetime import datetime

from .ensemble_stage2_model import (
    build_stage2_dnn, train_stage2_dnn, evaluate_ensemble, optimize_stage2_hyperparameters
)
from . import ensemble_database
from . import ensemble_config


def optimize_and_update_config(ensemble_models, X_val_s1, y_val_s1, X_val_s2, y_val_s2,
                               max_trials=30, executions_per_trial=3):
    """
    Run hyperparameter optimization and update ensemble_config.STAGE2_DNN_CONFIG.
    
    This function:
    1. Generates Stage 2 training data from current ensemble (conservative 95/5 split)
    2. Runs Keras Tuner optimization with focused search space
    3. Updates the global STAGE2_DNN_CONFIG dict in-memory
    4. Returns the optimized config for logging
    
    Parameters
    ----------
    ensemble_models : list
        Current ensemble models to use for generating training data
    X_val_s1, y_val_s1 : arrays
        Validation set 1 (will use all for training)
    X_val_s2, y_val_s2 : arrays
        Validation set 2 (will split 90/10 for train/val)
    max_trials : int, default=30
        Number of hyperparameter combinations to try
    executions_per_trial : int, default=3
        Number of training runs per combination
    
    Returns
    -------
    dict
        Optimized configuration dictionary
    """
    print(f"\n{'=' * 80}")
    print(f"RUNNING STAGE 2 DNN HYPERPARAMETER OPTIMIZATION")
    print(f"Ensemble size: {len(ensemble_models)} models")
    print(f"{'=' * 80}")
    
    # Generate Stage 1 predictions on both validation sets
    print("\n  Generating Stage 1 predictions...")
    all_stage1_preds_s1 = []
    for model in ensemble_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val_s1)[:, 1]
        else:
            pred = model.decision_function(X_val_s1)
        all_stage1_preds_s1.append(pred)
    
    all_stage1_preds_s2 = []
    for model in ensemble_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val_s2)[:, 1]
        else:
            pred = model.decision_function(X_val_s2)
        all_stage1_preds_s2.append(pred)
    
    # Stack predictions
    X_stage2_s1 = np.column_stack(all_stage1_preds_s1)
    X_stage2_s2 = np.column_stack(all_stage1_preds_s2)
    y_stage2_s1 = y_val_s1.values
    y_stage2_s2 = y_val_s2.values
    
    # Conservative 95/5 split: X_val_s1 + 90% X_val_s2 for train, 10% X_val_s2 for val
    split_idx = int(len(X_stage2_s2) * 0.9)
    X_stage2_s2_train = X_stage2_s2[:split_idx]
    X_stage2_s2_val = X_stage2_s2[split_idx:]
    y_stage2_s2_train = y_stage2_s2[:split_idx]
    y_stage2_s2_val = y_stage2_s2[split_idx:]
    
    X_train = np.vstack([X_stage2_s1, X_stage2_s2_train])
    y_train = np.concatenate([y_stage2_s1, y_stage2_s2_train])
    X_val = X_stage2_s2_val
    y_val = y_stage2_s2_val
    
    print(f"  Training samples: {len(X_train):,} (X_val_s1 + 90% X_val_s2)")
    print(f"  Validation samples: {len(X_val):,} (10% X_val_s2)")
    print(f"  Running {max_trials} trials with {executions_per_trial} executions each...")
    
    # Run optimization
    from pathlib import Path
    tuner_dir = Path('../models/keras_tuner')
    
    best_model, best_hps = optimize_stage2_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        project_name=f'stage2_online_tuning_{len(ensemble_models)}models',
        directory=tuner_dir
    )
    
    # Build optimized config dictionary
    optimized_config = {
        'architecture': {
            'hidden_layers': [],
            'output': {'units': 1, 'activation': 'sigmoid'}
        },
        'training': {
            'optimizer': 'Adam',
            'learning_rate': best_hps['learning_rate'],
            'loss': 'binary_crossentropy',
            'metrics': ['AUC', 'accuracy'],
            'epochs': 100,
            'batch_size': 128,
            'early_stopping': {
                'monitor': 'val_auc',
                'patience': 10,
                'mode': 'max',
                'restore_best_weights': True
            }
        },
        'retrain_frequency': 10
    }
    
    # Add hidden layers from optimized architecture
    for units in best_hps['units_per_layer']:
        layer_config = {
            'units': int(units),
            'activation': 'relu',
            'dropout': float(best_hps['dropout'])
        }
        optimized_config['architecture']['hidden_layers'].append(layer_config)
    
    # Update global config
    ensemble_config.STAGE2_DNN_CONFIG = optimized_config
    
    # Evaluate best model
    from sklearn.metrics import roc_auc_score
    y_pred = best_model.predict(X_val, verbose=0).flatten()
    val_auc = roc_auc_score(y_val, y_pred)
    
    print(f"\n  Optimization complete!")
    print(f"  Best hyperparameters:")
    print(f"    - Architecture: {best_hps['architecture_type']}")
    print(f"    - Layers: {best_hps['n_layers']} ({best_hps['units_per_layer']})")
    print(f"    - Dropout: {best_hps['dropout']:.3f}")
    print(f"    - Learning rate: {best_hps['learning_rate']:.6f}")
    print(f"  Validation AUC: {val_auc:.6f}")
    print(f"  Config updated in-memory for subsequent training")
    print(f"{'=' * 80}\n")
    
    return optimized_config


def train_or_expand_stage2_model(ensemble_models, stage2_model, X_val_s1, y_val_s1, X_val_s2, y_val_s2,
                                 stage2_epochs, stage2_batch_size, stage2_patience, current_iter,
                                 optimize_every_n_batches=None, run_optimization=True,
                                 optimization_trials=None):
    """
    Train new or expand existing stage 2 DNN with transfer learning.
    
    Parameters
    ----------
    ensemble_models : list
        Current ensemble models
    stage2_model : keras.Model or None
        Existing stage 2 model (None for first training)
    X_val_s1, y_val_s1 : arrays
        Stage 1 validation data for training stage 2
    X_val_s2, y_val_s2 : arrays
        Stage 2 validation data for evaluation
    stage2_epochs : int
        Training epochs
    stage2_batch_size : int
        Batch size for training
    stage2_patience : int
        Early stopping patience
    current_iter : int
        Current iteration number
    optimize_every_n_batches : int or None, default=None
        Run hyperparameter optimization every N batches (10, 20, 30, etc.)
        If None, uses adaptive schedule: batch 10 (50 trials), 20 (40 trials), 30+ (30 trials)
    run_optimization : bool, default=True
        Enable/disable periodic optimization
    optimization_trials : dict or None, default=None
        Custom trial counts per batch. Format: {batch_num: trials}
        Example: {10: 50, 20: 40, 30: 30}
    
    Returns
    -------
    tuple : (stage2_model, final_score, memory_used, elapsed_time)
        - stage2_model: trained DNN model
        - final_score: AUC on stage 2 validation set
        - memory_used: Memory used in MB
        - elapsed_time: Time elapsed in seconds
    """
    print(f"\n{'=' * 80}")
    print(f"BATCH COMPLETE: Training stage 2 DNN on {len(ensemble_models)} models")
    print(f"{'=' * 80}")
    
    # Check if we should run hyperparameter optimization
    batch_number = len(ensemble_models) // ensemble_config.STAGE2_DNN_CONFIG['retrain_frequency']
    
    # Adaptive optimization schedule (GPU-optimized)
    if optimization_trials is None:
        optimization_trials = {
            1: 50,   # First batch: thorough baseline (50 trials × 3 exec = ~45 min on GPU)
            2: 40,   # Second batch: refine (40 trials × 3 exec = ~30 min)
            3: 30,   # Third batch and beyond: adapt (30 trials × 2-3 exec = ~15-20 min)
        }
    
    if optimize_every_n_batches is None:
        # Default: optimize at batches 1, 2, then every 3rd batch (3, 6, 9, etc.)
        should_optimize = (
            run_optimization and 
            batch_number > 0 and
            (batch_number <= 2 or batch_number % 3 == 0)
        )
    else:
        # Custom frequency
        should_optimize = (
            run_optimization and 
            batch_number > 0 and 
            batch_number % optimize_every_n_batches == 0
        )
    
    if should_optimize:
        # Determine trial count for this batch
        trials = optimization_trials.get(batch_number, 30)  # Default to 30 for later batches
        executions = 3 if batch_number <= 2 else 2  # More executions for early batches
        
        optimize_and_update_config(
            ensemble_models, X_val_s1, y_val_s1, X_val_s2, y_val_s2,
            max_trials=trials, executions_per_trial=executions
        )
    
    # Track memory and time
    import time
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 ** 2)  # MB
    
    # Conservative approach: maximize training data while keeping small validation set
    # Use full X_val_s1 (140k) + 90% of X_val_s2 (126k) = 266k for training
    # Keep 10% of X_val_s2 (14k) as held-out validation for early stopping
    
    # Get predictions on X_val_s1 (will use all for training)
    all_stage1_preds_s1 = []
    for model in ensemble_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val_s1)[:, 1]
        else:
            pred = model.decision_function(X_val_s1)
        all_stage1_preds_s1.append(pred)
    
    # Get predictions on X_val_s2 (will split 90/10)
    all_stage1_preds_s2 = []
    for model in ensemble_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val_s2)[:, 1]
        else:
            pred = model.decision_function(X_val_s2)
        all_stage1_preds_s2.append(pred)
    
    # Stack predictions
    X_stage2_s1 = np.column_stack(all_stage1_preds_s1)
    X_stage2_s2 = np.column_stack(all_stage1_preds_s2)
    y_stage2_s1 = y_val_s1.values
    y_stage2_s2 = y_val_s2.values
    
    # Split X_val_s2 into training (90%) and validation (10%)
    split_idx = int(len(X_stage2_s2) * 0.9)
    X_stage2_s2_train = X_stage2_s2[:split_idx]
    X_stage2_s2_val = X_stage2_s2[split_idx:]
    y_stage2_s2_train = y_stage2_s2[:split_idx]
    y_stage2_s2_val = y_stage2_s2[split_idx:]
    
    # Combine X_val_s1 (all) + X_val_s2 (90%) for training
    X_train_s2 = np.vstack([X_stage2_s1, X_stage2_s2_train])
    y_train_s2 = np.concatenate([y_stage2_s1, y_stage2_s2_train])
    
    # Use 10% of X_val_s2 as validation
    X_val_s2_holdout = X_stage2_s2_val
    y_val_s2_holdout = y_stage2_s2_val
    
    if stage2_model is None:
        # First DNN training - use config architecture
        print(f"\n  Building initial stage 2 DNN from config...")
        config = ensemble_config.STAGE2_DNN_CONFIG
        stage2_model = build_stage2_dnn(
            n_models=len(ensemble_models),
            config=config
        )
    else:
        # Transfer learning: build new DNN with more inputs, copy weights where possible
        print(f"\n  Transfer learning: expanding DNN from {stage2_model.input_shape[1]} to {len(ensemble_models)} inputs...")
        
        # Save old weights
        old_weights = stage2_model.get_weights()
        
        # Build new model from config
        config = ensemble_config.STAGE2_DNN_CONFIG
        new_model = build_stage2_dnn(
            n_models=len(ensemble_models),
            config=config
        )
        
        # Transfer weights: copy input layer weights for existing models
        new_weights = new_model.get_weights()
        old_n_models = old_weights[0].shape[0]
        new_weights[0][:old_n_models, :] = old_weights[0]  # Copy old input weights
        new_weights[1] = old_weights[1]  # Copy input bias
        
        # Copy subsequent layer weights if shapes match
        if len(old_weights) > 2 and len(new_weights) > 2:
            for i in range(2, len(old_weights)):
                if old_weights[i].shape == new_weights[i].shape:
                    new_weights[i] = old_weights[i]
        
        new_model.set_weights(new_weights)
        stage2_model = new_model
    
    print(f"\n  Training stage 2 DNN (conservative: 95% train, 5% val)...")
    print(f"    Training samples: {len(X_train_s2):,} (X_val_s1 + 90% X_val_s2)")
    print(f"    Validation samples: {len(X_val_s2_holdout):,} (10% X_val_s2 holdout)")
    
    ensemble_id = f"batch_{len(ensemble_models)}"
    stage2_model, history = train_stage2_dnn(
        model=stage2_model,
        X_train=X_train_s2,
        y_train=y_train_s2,
        X_val=X_val_s2_holdout,
        y_val=y_val_s2_holdout,
        epochs=stage2_epochs,
        batch_size=stage2_batch_size,
        patience=stage2_patience,
        log_path=ensemble_id,
        iteration=current_iter
    )
    
    # Evaluate on held out stage 2 validation
    final_score = evaluate_ensemble(
        stage1_models=ensemble_models,
        stage2_model=stage2_model,
        X=X_val_s2,
        y=y_val_s2
    )
    
    # Track peak memory and elapsed time
    peak_memory = process.memory_info().rss / (1024 ** 2)  # MB
    memory_used = peak_memory - start_memory
    elapsed_time = time.time() - start_time
    
    print(f"\n  Stage 2 DNN trained!")
    print(f"  DNN ensemble AUC: {final_score:.6f}")
    print(f"  Memory used: {memory_used:.1f} MB")
    print(f"  Time elapsed: {elapsed_time:.1f}s")
    
    return stage2_model, final_score, memory_used, elapsed_time


def save_ensemble_bundle(ensemble_models, stage2_model, best_ensemble_score, current_iter,
                        models_dir, random_state, batch_size, n_workers, base_preprocessor,
                        numerical_features, ordinal_features, nominal_features,
                        education_categories, income_categories):
    """
    Save ensemble bundle checkpoint with all models and metadata.
    
    Parameters
    ----------
    ensemble_models : list
        Current ensemble models
    stage2_model : keras.Model
        Current stage 2 DNN
    best_ensemble_score : float
        Best ensemble score so far
    current_iter : int
        Current iteration
    models_dir : Path
        Directory to save bundle
    random_state : int
        Random state used
    batch_size : int
        Parallel batch size
    n_workers : int
        Number of parallel workers
    base_preprocessor : ColumnTransformer
        Base preprocessor
    numerical_features, ordinal_features, nominal_features : lists
        Feature lists
    education_categories, income_categories : lists
        Category lists
    
    Returns
    -------
    Path
        Path to saved bundle file
    """
    import joblib
    
    ensemble_bundle_path = models_dir / f'ensemble_bundle_iter_{current_iter}.joblib'
    
    # Calculate current acceptance rate from database
    conn = ensemble_database.sqlite3.connect(ensemble_database.DB_PATH)
    acceptance_stats = conn.execute("SELECT COUNT(*) as total, SUM(accepted) as accepted FROM ensemble_log").fetchone()
    conn.close()
    current_acceptance_rate = acceptance_stats[1] / acceptance_stats[0] if acceptance_stats[0] > 0 else 0.0
    
    ensemble_bundle = {
        'ensemble_models': ensemble_models,
        'stage2_model': stage2_model,
        'metadata': {
            'ensemble_size': len(ensemble_models),
            'current_iteration': current_iter,
            'best_score': best_ensemble_score,
            'acceptance_rate': current_acceptance_rate,
            'checkpoint_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'random_state': random_state,
            'parallel_batch_size': batch_size,
            'n_workers': n_workers
        },
        'base_preprocessor': base_preprocessor,
        'feature_info': {
            'numerical_features': numerical_features,
            'ordinal_features': ordinal_features,
            'nominal_features': nominal_features,
            'education_categories': education_categories,
            'income_categories': income_categories
        }
    }
    
    joblib.dump(ensemble_bundle, ensemble_bundle_path, compress=3)
    print(f"  Bundle checkpoint saved: {ensemble_bundle_path.name} ({ensemble_bundle_path.stat().st_size / (1024**2):.1f} MB)")
    
    return ensemble_bundle_path
