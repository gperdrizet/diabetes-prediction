"""
Stage 2 DNN training and management.

Handles DNN training with transfer learning for expanding ensembles.
"""

import numpy as np
import psutil
import os
from datetime import datetime

from .ensemble_stage2_model import (
    build_stage2_dnn, build_model_from_config, train_stage2_dnn, evaluate_ensemble, 
    evaluate_ensemble_with_cm, optimize_stage2_hyperparameters
)
from . import ensemble_database
from . import ensemble_config


def optimize_and_update_config(ensemble_models, X_val_s1, y_val_s1, X_val_s2, y_val_s2,
                               max_trials=8, executions_per_trial=1, sample_size=0.30):
    """
    Run hyperparameter optimization and update ensemble_config.STAGE2_DNN_CONFIG.
    
    This function:
    1. Generates Stage 2 training data from current ensemble (conservative 95/5 split)
    2. Runs Keras Tuner optimization with focused search space
    3. Updates the global STAGE2_DNN_CONFIG dict in-memory
    4. Returns the optimized config for logging
    
    OPTIMIZED FOR SPEED (online production):
    - Default 8 trials (down from 30) - sufficient for small hyperparameter space
    - Default 1 execution (down from 3) - early stopping provides regularization
    - Default 30% sample (down from 100%) - faster while maintaining ~1,050+ training samples
    
    Parameters
    ----------
    ensemble_models : list
        Current ensemble models to use for generating training data
    X_val_s1, y_val_s1 : arrays
        Validation set 1 (will sample for training)
    X_val_s2, y_val_s2 : arrays
        Validation set 2 (will sample and split 90/10 for train/val)
    max_trials : int, default=8
        Number of hyperparameter combinations to try
    executions_per_trial : int, default=1
        Number of training runs per combination
    sample_size : float, default=0.30
        Fraction of data to use for optimization (0.30 = 30%)
    
    Returns
    -------
    dict
        Optimized configuration dictionary
    """
    print(f"\n{'=' * 80}")
    print(f"RUNNING STAGE 2 DNN HYPERPARAMETER OPTIMIZATION")
    print(f"Ensemble size: {len(ensemble_models)} models")
    print(f"Optimization settings (FAST for online production):")
    print(f"  Trials: {max_trials}")
    print(f"  Executions per trial: {executions_per_trial}")
    print(f"  Sample size: {sample_size * 100:.0f}%")
    print(f"  Estimated time: ~{max_trials * executions_per_trial * 4:.0f} minutes")
    print(f"  (Based on observed ~7 min/trial on 50% sample, scaled to {sample_size * 100:.0f}% = ~4 min/trial)")
    print(f"{'=' * 80}")
    
    # Sample data for faster optimization
    sample_size_s1 = int(len(X_val_s1) * sample_size)
    sample_size_s2 = int(len(X_val_s2) * sample_size)
    
    X_val_s1_sample = X_val_s1[:sample_size_s1]
    y_val_s1_sample = y_val_s1[:sample_size_s1]
    X_val_s2_sample = X_val_s2[:sample_size_s2]
    y_val_s2_sample = y_val_s2[:sample_size_s2]
    
    # Generate Stage 1 predictions on sampled data
    print(f"\n  Generating Stage 1 predictions on {sample_size * 100:.0f}% sample...")
    all_stage1_preds_s1 = []
    for model in ensemble_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val_s1_sample)[:, 1]
        else:
            pred = model.decision_function(X_val_s1_sample)
        all_stage1_preds_s1.append(pred)
    
    all_stage1_preds_s2 = []
    for model in ensemble_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val_s2_sample)[:, 1]
        else:
            pred = model.decision_function(X_val_s2_sample)
        all_stage1_preds_s2.append(pred)
    
    # Stack predictions
    X_stage2_s1 = np.column_stack(all_stage1_preds_s1)
    X_stage2_s2 = np.column_stack(all_stage1_preds_s2)
    y_stage2_s1 = y_val_s1_sample.values if hasattr(y_val_s1_sample, 'values') else y_val_s1_sample
    y_stage2_s2 = y_val_s2_sample.values if hasattr(y_val_s2_sample, 'values') else y_val_s2_sample
    
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
    
    # Run optimization (returns hyperparameters dict, not model)
    from pathlib import Path
    tuner_dir = Path('../models/keras_tuner')
    
    best_hps, best_val_auc = optimize_stage2_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        max_epochs=50,  # Fast optimization
        project_name=f'stage2_online_tuning_{len(ensemble_models)}models',
        directory=tuner_dir
    )
    
    # Build optimized config dictionary with regularization
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
            'epochs': 200,  # Max epochs (early stopping will trigger earlier)
            'batch_size': 64,  # Smaller batch for better generalization
            'early_stopping': {
                'monitor': 'val_loss',  # Monitor val_loss, not val_auc
                'patience': 15,  # More patience for val_loss
                'mode': 'min',
                'restore_best_weights': True
            }
        },
        'retrain_frequency': 10,
        # Store raw hyperparameters for build_model_from_config()
        'hyperparameters': {
            'architecture_type': best_hps['architecture_type'],
            'n_layers': best_hps['n_layers'],
            'base_units': best_hps['base_units'],
            'dropout': best_hps['dropout'],
            'l2_reg': best_hps['l2_reg'],
            'learning_rate': best_hps['learning_rate']
        }
    }
    
    # Add hidden layers - need to reconstruct units_per_layer
    arch_type = best_hps['architecture_type']
    n_layers = best_hps['n_layers']
    base_units = best_hps['base_units']
    
    # Regenerate layer units (same logic as build_model_from_config)
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
    
    for units in units_per_layer:
        layer_config = {
            'units': int(units),
            'activation': 'relu',
            'dropout': float(best_hps['dropout']),
            'l2_reg': float(best_hps['l2_reg'])  # Store L2 regularization
        }
        optimized_config['architecture']['hidden_layers'].append(layer_config)
    
    # Update global config
    ensemble_config.STAGE2_DNN_CONFIG = optimized_config
    
    print(f"\n  Optimization complete!")
    print(f"  Best hyperparameters:")
    print(f"    - Architecture: {arch_type}")
    print(f"    - Layers: {n_layers} ({units_per_layer})")
    print(f"    - Dropout: {best_hps['dropout']:.3f}")
    print(f"    - L2 reg: {best_hps['l2_reg']:.6f}")
    print(f"    - Learning rate: {best_hps['learning_rate']:.6f}")
    print(f"  Validation AUC: {best_val_auc:.6f}")
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
        If None, uses adaptive schedule: batches 1, 2, then every 3rd (3, 6, 9...)
    run_optimization : bool, default=True
        Enable/disable periodic optimization (enabled with fast settings)
    optimization_trials : dict or None, default=None
        Custom trial counts per batch. Format: {batch_num: trials}
        Default: {1: 10, 2: 8, 3: 5} (fast settings to prevent hanging)
    
    Returns
    -------
    tuple : (stage2_model, final_score, memory_used, elapsed_time, stage2_tp, stage2_fp, stage2_tn, stage2_fn)
        - stage2_model: trained DNN model
        - final_score: AUC on stage 2 validation set
        - memory_used: Memory used in MB
        - elapsed_time: Time elapsed in seconds
        - stage2_tp: Confusion matrix true positives
        - stage2_fp: Confusion matrix false positives
        - stage2_tn: Confusion matrix true negatives
        - stage2_fn: Confusion matrix false negatives
    """
    # Check if we should run hyperparameter optimization
    batch_number = len(ensemble_models) // ensemble_config.STAGE2_DNN_CONFIG['retrain_frequency']
    
    # Adaptive optimization schedule (reduced to prevent hanging)
    if optimization_trials is None:
        optimization_trials = {
            1: 10,   # First batch: baseline exploration (10 trials × 1 exec = ~10-15 min)
            2: 8,    # Second batch: refinement (8 trials × 1 exec = ~8-12 min)
            3: 5,    # Third batch and beyond: quick adaptation (5 trials × 1 exec = ~5-8 min)
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
        trials = optimization_trials.get(batch_number, 5)  # Default to 5 for later batches
        executions = 1  # Single execution per trial (statistical confidence not critical)
        
        print(f"\n{'=' * 80}")
        print(f"BATCH {batch_number}: Running hyperparameter optimization BEFORE training")
        print(f"Ensemble size: {len(ensemble_models)} models")
        print(f"{'=' * 80}")
        
        optimize_and_update_config(
            ensemble_models, X_val_s1, y_val_s1, X_val_s2, y_val_s2,
            max_trials=trials, executions_per_trial=executions
        )
    
    # Now print training header
    print(f"\n{'=' * 80}")
    print(f"BATCH {batch_number}: Training stage 2 DNN on {len(ensemble_models)} models")
    print(f"{'=' * 80}")
    
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
    
    # Build fresh DNN from optimized hyperparameters (re-optimization makes transfer learning pointless)
    print(f"\n  Building stage 2 DNN from optimized hyperparameters...")
    config = ensemble_config.STAGE2_DNN_CONFIG
    
    # Use build_model_from_config for regularized models
    if 'hyperparameters' in config:
        # New regularized workflow
        stage2_model, units_per_layer = build_model_from_config(
            config=config['hyperparameters'],
            n_models=len(ensemble_models)
        )
        print(f"    Architecture: {config['hyperparameters']['architecture_type']}")
        print(f"    Layers: {units_per_layer}")
    else:
        # Legacy workflow (backward compatibility)
        stage2_model = build_stage2_dnn(
            n_models=len(ensemble_models),
            config=config
        )
    
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
    final_score, stage2_tn, stage2_fp, stage2_fn, stage2_tp = evaluate_ensemble_with_cm(
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
    print(f"  Confusion matrix: TP={stage2_tp}, FP={stage2_fp}, TN={stage2_tn}, FN={stage2_fn}")
    print(f"  Memory used: {memory_used:.1f} MB")
    print(f"  Time elapsed: {elapsed_time:.1f}s")
    
    return stage2_model, final_score, memory_used, elapsed_time, stage2_tp, stage2_fp, stage2_tn, stage2_fn


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


def generate_pseudo_labels(
    ensemble_models, 
    stage2_model, 
    test_df, 
    label_column='diagnosed_diabetes',
    confidence_threshold=0.95,
    max_samples=None,
    balance_classes=True,
    target_class_ratio=None
):
    """
    Generate pseudo-labels from high-confidence predictions on unlabeled test data.
    
    Parameters
    ----------
    ensemble_models : list
        Trained Stage 1 ensemble models
    stage2_model : keras.Model
        Trained Stage 2 DNN meta-learner
    test_df : pd.DataFrame
        Unlabeled test data (competition test set)
    label_column : str, default='diagnosed_diabetes'
        Name of label column to create
    confidence_threshold : float, default=0.95
        Minimum prediction probability for inclusion (0.95 = very confident)
    max_samples : int or None, default=None
        Maximum number of pseudo-labeled samples to return
        If None, returns all high-confidence samples
    balance_classes : bool, default=True
        Ensure pseudo-labeled samples have balanced class distribution (50/50)
        If target_class_ratio is provided, this parameter is ignored
    target_class_ratio : float or None, default=None
        Target ratio for positive class (e.g., 0.14 = 14% positive, 86% negative)
        If provided, overrides balance_classes parameter
        If None and balance_classes=True, uses 50/50 split
        If None and balance_classes=False, uses natural distribution from predictions
    
    Returns
    -------
    tuple : (X_pseudo, y_pseudo, stats)
        - X_pseudo: DataFrame of pseudo-labeled features
        - y_pseudo: Series of pseudo-labels (0 or 1)
        - stats: Dict with statistics about pseudo-labeling
    """
    import pandas as pd
    
    print(f"\n{'=' * 80}")
    print("PSEUDO-LABELING: Generating labels from test set")
    print(f"{'=' * 80}")
    print(f"Test set size: {len(test_df):,} samples")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    
    # Generate Stage 1 predictions on test data
    print("Generating Stage 1 predictions...")
    all_stage1_preds = []
    for i, model in enumerate(ensemble_models):
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(test_df)[:, 1]
        else:
            pred = model.decision_function(test_df)
        all_stage1_preds.append(pred)
    
    X_stage1_test = np.column_stack(all_stage1_preds)
    
    # Generate Stage 2 predictions (final ensemble predictions)
    print("Generating Stage 2 predictions...")
    stage2_probs = stage2_model.predict(X_stage1_test, verbose=0).flatten()
    
    # Filter high-confidence predictions
    high_conf_positive = stage2_probs >= confidence_threshold
    high_conf_negative = stage2_probs <= (1 - confidence_threshold)
    high_conf_mask = high_conf_positive | high_conf_negative
    
    n_high_conf = high_conf_mask.sum()
    n_positive = high_conf_positive.sum()
    n_negative = high_conf_negative.sum()
    
    print(f"\nHigh-confidence predictions:")
    print(f"  Total: {n_high_conf:,} ({n_high_conf/len(test_df)*100:.1f}%)")
    print(f"  Positive (p ≥ {confidence_threshold:.2f}): {n_positive:,}")
    print(f"  Negative (p ≤ {1-confidence_threshold:.2f}): {n_negative:,}")
    
    if n_high_conf == 0:
        print("⚠️  WARNING: No high-confidence predictions found!")
        return pd.DataFrame(), pd.Series(dtype=int), {'n_total': 0, 'n_positive': 0, 'n_negative': 0}
    
    # Create pseudo-labels
    pseudo_labels = (stage2_probs >= 0.5).astype(int)
    
    # Filter to high-confidence samples
    X_pseudo = test_df[high_conf_mask].copy()
    y_pseudo = pd.Series(pseudo_labels[high_conf_mask], index=X_pseudo.index, name=label_column)
    confidences = stage2_probs[high_conf_mask]
    
    # Apply class distribution strategy
    if target_class_ratio is not None and n_positive > 0 and n_negative > 0:
        # Match target class ratio (e.g., original training data distribution)
        positive_idx = y_pseudo[y_pseudo == 1].index
        negative_idx = y_pseudo[y_pseudo == 0].index
        
        # Calculate target counts to achieve desired ratio
        # If we have enough samples, use the ratio directly
        # Otherwise, use what's available while maintaining ratio
        total_available = len(y_pseudo)
        target_positive_count = int(total_available * target_class_ratio)
        target_negative_count = total_available - target_positive_count
        
        # Cap at available samples
        actual_positive_count = min(target_positive_count, n_positive)
        actual_negative_count = min(target_negative_count, n_negative)
        
        # If one class is limited, adjust the other to maintain ratio
        if actual_positive_count < target_positive_count:
            # Positive class is limited, reduce negative to maintain ratio
            actual_negative_count = int(actual_positive_count * (1 - target_class_ratio) / target_class_ratio)
            actual_negative_count = min(actual_negative_count, n_negative)
        elif actual_negative_count < target_negative_count:
            # Negative class is limited, reduce positive to maintain ratio
            actual_positive_count = int(actual_negative_count * target_class_ratio / (1 - target_class_ratio))
            actual_positive_count = min(actual_positive_count, n_positive)
        
        np.random.seed(315)  # For reproducibility
        selected_positive = np.random.choice(positive_idx, size=actual_positive_count, replace=False)
        selected_negative = np.random.choice(negative_idx, size=actual_negative_count, replace=False)
        
        selected_idx = np.concatenate([selected_positive, selected_negative])
        X_pseudo = X_pseudo.loc[selected_idx]
        y_pseudo = y_pseudo.loc[selected_idx]
        
        actual_ratio = actual_positive_count / (actual_positive_count + actual_negative_count)
        print(f"\nTarget class distribution matching:")
        print(f"  Target positive ratio: {target_class_ratio:.1%}")
        print(f"  Actual positive ratio: {actual_ratio:.1%}")
        print(f"  Positive samples: {actual_positive_count:,}")
        print(f"  Negative samples: {actual_negative_count:,}")
        print(f"  Total pseudo-labeled: {len(X_pseudo):,}")
        
    elif balance_classes and n_positive > 0 and n_negative > 0:
        # Balance classes (50/50 split)
        min_class_size = min(n_positive, n_negative)
        
        # Sample equal numbers from each class
        positive_idx = y_pseudo[y_pseudo == 1].index
        negative_idx = y_pseudo[y_pseudo == 0].index
        
        np.random.seed(315)  # For reproducibility
        selected_positive = np.random.choice(positive_idx, size=min_class_size, replace=False)
        selected_negative = np.random.choice(negative_idx, size=min_class_size, replace=False)
        
        selected_idx = np.concatenate([selected_positive, selected_negative])
        X_pseudo = X_pseudo.loc[selected_idx]
        y_pseudo = y_pseudo.loc[selected_idx]
        
        print(f"\nClass balancing (50/50):")
        print(f"  Kept {min_class_size:,} samples per class")
        print(f"  Total pseudo-labeled: {len(X_pseudo):,}")
    
    # Apply max_samples limit if specified
    if max_samples is not None and len(X_pseudo) > max_samples:
        # Keep highest confidence samples
        conf_distances = np.abs(confidences.loc[X_pseudo.index] - 0.5)
        conf_order = np.argsort(conf_distances)[::-1]  # Highest confidence first
        keep_idx = conf_order[:max_samples]
        X_pseudo = X_pseudo.iloc[keep_idx]
        y_pseudo = y_pseudo.iloc[keep_idx]
        
        print(f"\nSample limit applied:")
        print(f"  Kept {max_samples:,} highest-confidence samples")
    
    # Compute statistics
    final_confidences = stage2_probs[X_pseudo.index]
    stats = {
        'n_total': len(X_pseudo),
        'n_positive': (y_pseudo == 1).sum(),
        'n_negative': (y_pseudo == 0).sum(),
        'mean_confidence': np.mean(np.maximum(final_confidences, 1 - final_confidences)),
        'min_confidence': confidence_threshold,
        'test_set_size': len(test_df),
        'coverage': len(X_pseudo) / len(test_df) * 100
    }
    
    print(f"\nFinal pseudo-labeled dataset:")
    print(f"  Total samples: {stats['n_total']:,}")
    print(f"  Positive: {stats['n_positive']:,} ({stats['n_positive']/max(stats['n_total'], 1)*100:.1f}%)")
    print(f"  Negative: {stats['n_negative']:,} ({stats['n_negative']/max(stats['n_total'], 1)*100:.1f}%)")
    print(f"  Mean confidence: {stats['mean_confidence']:.3f}")
    print(f"  Coverage: {stats['coverage']:.1f}% of test set")
    print(f"{'=' * 80}\n")
    
    return X_pseudo, y_pseudo, stats


def augment_training_pool_with_pseudo_labels(
    X_train_pool, 
    y_train_pool, 
    X_pseudo, 
    y_pseudo,
    max_pseudo_fraction=0.20
):
    """
    Augment training pool with pseudo-labeled data.
    
    Parameters
    ----------
    X_train_pool : pd.DataFrame
        Original training pool
    y_train_pool : pd.Series
        Original training labels
    X_pseudo : pd.DataFrame
        Pseudo-labeled features
    y_pseudo : pd.Series
        Pseudo-labels
    max_pseudo_fraction : float, default=0.20
        Maximum fraction of pseudo-labeled data (0.20 = 20% of total)
    
    Returns
    -------
    tuple : (X_augmented, y_augmented, stats)
        - X_augmented: Combined training data
        - y_augmented: Combined labels
        - stats: Dict with augmentation statistics
    """
    import pandas as pd
    
    print(f"\n{'=' * 80}")
    print("AUGMENTING TRAINING POOL WITH PSEUDO-LABELS")
    print(f"{'=' * 80}")
    print(f"Original training pool: {len(X_train_pool):,} samples")
    print(f"Pseudo-labeled samples: {len(X_pseudo):,} samples")
    
    # Check if pseudo-labeled data exceeds max fraction
    max_pseudo_samples = int(len(X_train_pool) * max_pseudo_fraction / (1 - max_pseudo_fraction))
    
    if len(X_pseudo) > max_pseudo_samples:
        print(f"\n⚠️  Limiting pseudo-labeled data to {max_pseudo_fraction*100:.0f}% of total:")
        print(f"  Keeping {max_pseudo_samples:,} of {len(X_pseudo):,} pseudo-labeled samples")
        
        # Keep random sample
        np.random.seed(315)
        keep_idx = np.random.choice(len(X_pseudo), size=max_pseudo_samples, replace=False)
        X_pseudo = X_pseudo.iloc[keep_idx]
        y_pseudo = y_pseudo.iloc[keep_idx]
    
    # Combine datasets
    X_augmented = pd.concat([X_train_pool, X_pseudo], ignore_index=True)
    y_augmented = pd.concat([y_train_pool, y_pseudo], ignore_index=True)
    
    # Statistics
    stats = {
        'original_size': len(X_train_pool),
        'pseudo_size': len(X_pseudo),
        'augmented_size': len(X_augmented),
        'pseudo_fraction': len(X_pseudo) / len(X_augmented),
        'original_positive_rate': y_train_pool.mean(),
        'pseudo_positive_rate': y_pseudo.mean() if len(y_pseudo) > 0 else 0,
        'augmented_positive_rate': y_augmented.mean()
    }
    
    print(f"\nAugmented training pool:")
    print(f"  Total size: {stats['augmented_size']:,}")
    print(f"  Original: {stats['original_size']:,} ({(1-stats['pseudo_fraction'])*100:.1f}%)")
    print(f"  Pseudo-labeled: {stats['pseudo_size']:,} ({stats['pseudo_fraction']*100:.1f}%)")
    print(f"\nClass distribution:")
    print(f"  Original positive rate: {stats['original_positive_rate']:.3f}")
    print(f"  Pseudo positive rate: {stats['pseudo_positive_rate']:.3f}")
    print(f"  Augmented positive rate: {stats['augmented_positive_rate']:.3f}")
    print(f"{'=' * 80}\n")
    
    return X_augmented, y_augmented, stats
