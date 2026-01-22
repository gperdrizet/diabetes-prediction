"""Ensemble evaluation utilities with caching support.

Provides optimized evaluation functions for Stage 1 ensembles with
prediction caching to avoid redundant computations during hill climbing.
"""

from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_stage1_ensemble(
    models: List,
    model_hashes: List[str],
    X: np.ndarray,
    y: np.ndarray,
    cache: Optional[Dict[str, np.ndarray]] = None
) -> float:
    """Evaluate Stage 1 ensemble with prediction caching.
    
    Efficiently evaluates ensemble by caching individual model predictions
    and reusing them across multiple ensemble configurations. Critical for
    performance during hill climbing where models are repeatedly evaluated
    in different combinations.
    
    Parameters
    ----------
    models : list of sklearn pipelines
        Trained Stage 1 models to evaluate.
    model_hashes : list of str
        Unique hashes for each model (used as cache keys).
    X : np.ndarray or pd.DataFrame
        Validation features.
    y : np.ndarray or pd.Series
        True labels.
    cache : dict, optional
        Cache mapping model_hash -> predictions. If provided, will
        check cache before computing predictions. Modified in-place.
    
    Returns
    -------
    auc : float
        ROC-AUC score of the averaged ensemble predictions.
        
    Notes
    -----
    - Cache persists across calls, enabling reuse of predictions
    - Models are combined via simple averaging (no Stage 2 weighting)
    - Each model must support predict_proba() interface
    
    Examples
    --------
    >>> # Initialize cache
    >>> cache = {}
    >>> 
    >>> # Evaluate ensemble with caching
    >>> auc = evaluate_stage1_ensemble(
    ...     models=[model1, model2, model3],
    ...     model_hashes=['hash1', 'hash2', 'hash3'],
    ...     X=X_val,
    ...     y=y_val,
    ...     cache=cache
    ... )
    >>> 
    >>> # Add new model - reuses cached predictions for model1, model2, model3
    >>> auc_new = evaluate_stage1_ensemble(
    ...     models=[model1, model2, model3, model4],
    ...     model_hashes=['hash1', 'hash2', 'hash3', 'hash4'],
    ...     X=X_val,
    ...     y=y_val,
    ...     cache=cache  # Only model4 predictions computed
    ... )
    """
    if cache is None:
        cache = {}
    
    predictions = []
    for model, model_hash in zip(models, model_hashes):
        if model_hash in cache:
            # Use cached predictions
            preds = cache[model_hash]
        else:
            # Compute and cache predictions
            preds = model.predict_proba(X)[:, 1]
            cache[model_hash] = preds
        predictions.append(preds)
    
    # Average ensemble predictions
    ensemble_pred = np.mean(predictions, axis=0)
    auc = roc_auc_score(y, ensemble_pred)
    return auc
