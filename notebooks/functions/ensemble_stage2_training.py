"""
Stage 2 DNN training and management.

Handles DNN training with transfer learning for expanding ensembles.
"""

import numpy as np
import psutil
import os
from datetime import datetime

from .ensemble_stage2_model import build_stage2_dnn, train_stage2_dnn, evaluate_ensemble
from . import ensemble_database


def train_or_expand_stage2_model(ensemble_models, stage2_model, X_val_s1, y_val_s1, X_val_s2, y_val_s2,
                                 stage2_epochs, stage2_batch_size, stage2_patience, current_iter):
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
    
    Returns
    -------
    tuple : (stage2_model, final_score)
        - stage2_model: trained DNN model
        - final_score: AUC on stage 2 validation set
    """
    print(f"\n{'=' * 80}")
    print(f"BATCH COMPLETE: Training stage 2 DNN on {len(ensemble_models)} models")
    print(f"{'=' * 80}")
    
    # Track memory usage
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 ** 2)  # MB
    
    # Get all predictions on stage 1 validation set
    all_stage1_preds = []
    for model in ensemble_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val_s1)[:, 1]
        else:
            pred = model.decision_function(X_val_s1)
        all_stage1_preds.append(pred)
    
    X_stage2_train_full = np.column_stack(all_stage1_preds)
    y_stage2_train_full = y_val_s1.values
    
    # Sample for training
    sample_size = min(50000, len(X_stage2_train_full))
    sample_indices = np.random.choice(len(X_stage2_train_full), size=sample_size, replace=False)
    X_stage2_sample = X_stage2_train_full[sample_indices]
    y_stage2_sample = y_stage2_train_full[sample_indices]
    
    # Train/val split
    split_idx = int(len(X_stage2_sample) * 0.8)
    X_train_s2 = X_stage2_sample[:split_idx]
    y_train_s2 = y_stage2_sample[:split_idx]
    X_val_s2_internal = X_stage2_sample[split_idx:]
    y_val_s2_internal = y_stage2_sample[split_idx:]
    
    if stage2_model is None:
        # First DNN training
        print(f"\n  Building initial stage 2 DNN...")
        stage2_model = build_stage2_dnn(
            n_models=len(ensemble_models),
            n_layers=1,
            units_per_layer=32,
            dropout=0.2,
            batch_norm=False,
            activation='relu',
            learning_rate=0.001
        )
    else:
        # Transfer learning: build new DNN with more inputs, copy weights where possible
        print(f"\n  Transfer learning: expanding DNN from {stage2_model.input_shape[1]} to {len(ensemble_models)} inputs...")
        
        # Save old weights
        old_weights = stage2_model.get_weights()
        
        # Build new model
        new_model = build_stage2_dnn(
            n_models=len(ensemble_models),
            n_layers=1,
            units_per_layer=32,
            dropout=0.2,
            batch_norm=False,
            activation='relu',
            learning_rate=0.001
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
    
    print(f"\n  Training stage 2 DNN...")
    print(f"    Training samples: {len(X_train_s2):,}")
    print(f"    Validation samples: {len(X_val_s2_internal):,}")
    
    ensemble_id = f"batch_{len(ensemble_models)}"
    stage2_model, history = train_stage2_dnn(
        model=stage2_model,
        X_train=X_train_s2,
        y_train=y_train_s2,
        X_val=X_val_s2_internal,
        y_val=y_val_s2_internal,
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
    
    # Track peak memory
    peak_memory = process.memory_info().rss / (1024 ** 2)  # MB
    memory_used = peak_memory - start_memory
    
    print(f"\n  Stage 2 DNN trained!")
    print(f"  DNN ensemble AUC: {final_score:.6f}")
    print(f"  Memory used: {memory_used:.1f} MB")
    
    return stage2_model, final_score, memory_used


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
