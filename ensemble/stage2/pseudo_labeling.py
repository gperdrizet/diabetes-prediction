"""Pseudo-labeling for semi-supervised learning.

This module provides functions for generating pseudo-labels from high-confidence
predictions on unlabeled test data and augmenting the training pool.
"""

from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd


def generate_pseudo_labels(
    ensemble_models: List[Any],
    stage2_model: Any,
    test_df: pd.DataFrame,
    label_column: str = 'diagnosed_diabetes',
    confidence_threshold: float = 0.95,
    max_samples: Optional[int] = None,
    balance_classes: bool = True,
    target_class_ratio: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Generate pseudo-labels from high-confidence predictions on unlabeled test data.
    
    Parameters
    ----------
    ensemble_models : list
        Trained Stage 1 ensemble models.
    stage2_model : keras.Model
        Trained Stage 2 DNN meta-learner.
    test_df : pd.DataFrame
        Unlabeled test data (competition test set).
    label_column : str, default='diagnosed_diabetes'
        Name of label column to create.
    confidence_threshold : float, default=0.95
        Minimum prediction probability for inclusion (0.95 = very confident).
    max_samples : int or None, default=None
        Maximum number of pseudo-labeled samples to return.
        If None, returns all high-confidence samples.
    balance_classes : bool, default=True
        Ensure pseudo-labeled samples have balanced class distribution (50/50).
        If target_class_ratio is provided, this parameter is ignored.
    target_class_ratio : float or None, default=None
        Target ratio for positive class (e.g., 0.14 = 14% positive, 86% negative).
        If provided, overrides balance_classes parameter.
        If None and balance_classes=True, uses 50/50 split.
        If None and balance_classes=False, uses natural distribution from predictions.
    
    Returns
    -------
    X_pseudo : pd.DataFrame
        Pseudo-labeled features.
    y_pseudo : pd.Series
        Pseudo-labels (0 or 1).
    stats : dict
        Statistics about pseudo-labeling.
    """
    print(f"\n{'=' * 80}")
    print("PSEUDO-LABELING: Generating labels from test set")
    print(f"{'=' * 80}")
    print(f"Test set size: {len(test_df):,} samples")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    
    # Generate Stage 1 predictions on test data
    print("Generating Stage 1 predictions...")
    all_stage1_preds = []
    for model in ensemble_models:
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
        return pd.DataFrame(), pd.Series(dtype=int), {
            'n_total': 0,
            'n_positive': 0,
            'n_negative': 0
        }
    
    # Create pseudo-labels
    pseudo_labels = (stage2_probs >= 0.5).astype(int)
    
    # Filter to high-confidence samples
    X_pseudo = test_df[high_conf_mask].copy()
    y_pseudo = pd.Series(pseudo_labels[high_conf_mask], index=X_pseudo.index, name=label_column)
    confidences = stage2_probs[high_conf_mask]
    
    # Apply class distribution strategy
    if target_class_ratio is not None and n_positive > 0 and n_negative > 0:
        # Match target class ratio (e.g., original training data distribution)
        X_pseudo, y_pseudo = _apply_target_ratio(
            X_pseudo, y_pseudo, target_class_ratio, n_positive, n_negative
        )
        
    elif balance_classes and n_positive > 0 and n_negative > 0:
        # Balance classes (50/50 split)
        X_pseudo, y_pseudo = _balance_classes(X_pseudo, y_pseudo, n_positive, n_negative)
    
    # Apply max_samples limit if specified
    if max_samples is not None and len(X_pseudo) > max_samples:
        X_pseudo, y_pseudo = _limit_samples(
            X_pseudo, y_pseudo, confidences, max_samples
        )
    
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


def augment_training_pool(
    X_train_pool: pd.DataFrame,
    y_train_pool: pd.Series,
    X_pseudo: pd.DataFrame,
    y_pseudo: pd.Series,
    max_pseudo_fraction: float = 0.20
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Augment training pool with pseudo-labeled data.
    
    Parameters
    ----------
    X_train_pool : pd.DataFrame
        Original training pool.
    y_train_pool : pd.Series
        Original training labels.
    X_pseudo : pd.DataFrame
        Pseudo-labeled features.
    y_pseudo : pd.Series
        Pseudo-labels.
    max_pseudo_fraction : float, default=0.20
        Maximum fraction of pseudo-labeled data (0.20 = 20% of total).
    
    Returns
    -------
    X_augmented : pd.DataFrame
        Combined training data.
    y_augmented : pd.Series
        Combined labels.
    stats : dict
        Augmentation statistics.
    """
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


# Helper functions

def _apply_target_ratio(
    X_pseudo: pd.DataFrame,
    y_pseudo: pd.Series,
    target_class_ratio: float,
    n_positive: int,
    n_negative: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply target class ratio to pseudo-labeled data."""
    positive_idx = y_pseudo[y_pseudo == 1].index
    negative_idx = y_pseudo[y_pseudo == 0].index
    
    # Calculate target counts to achieve desired ratio
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
    
    return X_pseudo, y_pseudo


def _balance_classes(
    X_pseudo: pd.DataFrame,
    y_pseudo: pd.Series,
    n_positive: int,
    n_negative: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance classes to 50/50 split."""
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
    
    return X_pseudo, y_pseudo


def _limit_samples(
    X_pseudo: pd.DataFrame,
    y_pseudo: pd.Series,
    confidences: np.ndarray,
    max_samples: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Limit to max_samples by keeping highest confidence samples."""
    # Keep highest confidence samples
    conf_distances = np.abs(confidences.loc[X_pseudo.index] - 0.5)
    conf_order = np.argsort(conf_distances)[::-1]  # Highest confidence first
    keep_idx = conf_order[:max_samples]
    X_pseudo = X_pseudo.iloc[keep_idx]
    y_pseudo = y_pseudo.iloc[keep_idx]
    
    print(f"\nSample limit applied:")
    print(f"  Kept {max_samples:,} highest-confidence samples")
    
    return X_pseudo, y_pseudo
