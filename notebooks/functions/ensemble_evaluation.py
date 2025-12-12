"""
Ensemble evaluation and scoring module.

Handles hybrid scoring logic combining DNN and simple mean approaches.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

# Use absolute imports instead of relative imports
from ensemble_hill_climbing import calculate_ensemble_diversity
from ensemble_stage2_model import evaluate_ensemble


def evaluate_candidate_ensemble(candidate_ensemble, ensemble_models, stage2_model, 
                                X_val_s1, X_val_s2, y_val_s1, y_val_s2):
    """
    Evaluate candidate ensemble using hybrid scoring.
    
    Uses DNN for models trained with current DNN, simple mean for new models.
    
    Parameters
    ----------
    candidate_ensemble : list
        List of all models including candidate
    ensemble_models : list
        List of current ensemble models (without candidate)
    stage2_model : keras.Model or None
        Current stage 2 DNN model
    X_val_s1, y_val_s1 : arrays
        Stage 1 validation data (for diversity calculation)
    X_val_s2, y_val_s2 : arrays
        Stage 2 validation data (for scoring)
    
    Returns
    -------
    tuple : (candidate_score, diversity_score, aggregation_method)
        - candidate_score: ROC-AUC of candidate ensemble
        - diversity_score: Diversity measure of candidate ensemble
        - aggregation_method: String describing scoring method used
    """
    # Determine how many models have been trained with the current DNN
    if stage2_model is None:
        n_dnn_trained = 0
    else:
        n_dnn_trained = stage2_model.input_shape[1]
    
    # Calculate number of "new" models (accepted but not yet in DNN)
    n_new_models = len(candidate_ensemble) - n_dnn_trained
    
    # Hybrid scoring: DNN for old models + simple mean for new models
    if n_dnn_trained == 0:
        # No DNN yet - use simple mean for all
        all_preds = []
        for model in candidate_ensemble:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val_s2)[:, 1]
            else:
                pred = model.decision_function(X_val_s2)
            all_preds.append(pred)
        
        ensemble_pred = np.mean(all_preds, axis=0)
        candidate_score = roc_auc_score(y_val_s2, ensemble_pred)
        aggregation_method = "simple mean (all)"
    
    elif n_new_models == 0:
        # All models are in the DNN - use DNN only
        candidate_score = evaluate_ensemble(
            stage1_models=candidate_ensemble,
            stage2_model=stage2_model,
            X=X_val_s2,
            y=y_val_s2
        )
        aggregation_method = "DNN (all)"
    
    else:
        # Hybrid: DNN for first n_dnn_trained models + simple mean for new models
        dnn_models = ensemble_models[:n_dnn_trained]
        
        # Generate stage 1 predictions for DNN models
        dnn_stage1_preds = []
        for model in dnn_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val_s2)[:, 1]
            else:
                pred = model.decision_function(X_val_s2)
            dnn_stage1_preds.append(pred)
        
        dnn_stage1_preds = np.column_stack(dnn_stage1_preds)
        
        # Get DNN output (weighted predictions)
        dnn_output = stage2_model.predict(dnn_stage1_preds, verbose=0).flatten()
        
        # Get simple mean for new models (including candidate)
        new_models = candidate_ensemble[n_dnn_trained:]
        new_preds = []
        for model in new_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val_s2)[:, 1]
            else:
                pred = model.decision_function(X_val_s2)
            new_preds.append(pred)
        
        new_mean_pred = np.mean(new_preds, axis=0)
        
        # Combine: weighted average
        combined_pred = (dnn_output * n_dnn_trained + new_mean_pred * n_new_models) / len(candidate_ensemble)
        candidate_score = roc_auc_score(y_val_s2, combined_pred)
        aggregation_method = f"hybrid (DNN×{n_dnn_trained} + mean×{n_new_models})"
    
    # Calculate diversity on stage 1 validation set
    all_predictions = []
    for model in candidate_ensemble:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val_s1)[:, 1]
        else:
            pred = model.decision_function(X_val_s1)
        all_predictions.append(pred)
    
    all_predictions = np.column_stack(all_predictions)
    diversity_score = calculate_ensemble_diversity(all_predictions)
    
    return candidate_score, diversity_score, aggregation_method
