"""
Ensemble initialization module.

Handles data splitting, preprocessor creation, and founder model training.
"""

import time
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

import sys
models_path = Path('../models').resolve()
sys.path.insert(0, str(models_path))
from logistic_regression_transformers import IQRClipper

from .ensemble_hill_climbing import generate_random_pipeline, compute_pipeline_hash, log_iteration


def create_data_splits(train_df, label, random_state):
    """
    Create fixed three-way data split.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    label : str
        Name of the label column
    random_state : int
        Random state for reproducible splits
    
    Returns
    -------
    tuple : (X_train_pool, X_val_s1, X_val_s2, y_train_pool, y_val_s1, y_val_s2)
        - X_train_pool, y_train_pool: 60% for training stage 1 models
        - X_val_s1, y_val_s1: 35% for evaluating stage 1 models and training stage 2
        - X_val_s2, y_val_s2: 5% for evaluating stage 2 model (held out)
    """
    X_full = train_df.drop(columns=[label])
    y_full = train_df[label]
    
    # First split: training pool vs validation (60-40)
    X_train_pool, X_val_combined, y_train_pool, y_val_combined = train_test_split(
        X_full, 
        y_full, 
        test_size=0.4,
        random_state=random_state,
        stratify=y_full
    )
    
    # Second split: stage 1 validation vs stage 2 validation (35-5 from the 40%)
    X_val_s1, X_val_s2, y_val_s1, y_val_s2 = train_test_split(
        X_val_combined,
        y_val_combined,
        test_size=0.125,  # 0.125 * 40% = 5% of total
        random_state=random_state,
        stratify=y_val_combined
    )
    
    return X_train_pool, X_val_s1, X_val_s2, y_train_pool, y_val_s1, y_val_s2


def create_base_preprocessor(numerical_features, ordinal_features, nominal_features, 
                             education_categories, income_categories):
    """
    Create base preprocessor shared across all stage 1 models.
    
    Parameters
    ----------
    numerical_features : list
        List of numerical feature names
    ordinal_features : list
        List of ordinal feature names
    nominal_features : list
        List of nominal feature names
    education_categories : list
        Categories for education level
    income_categories : list
        Categories for income level
    
    Returns
    -------
    ColumnTransformer
        Configured preprocessor
    """
    # Create numerical pipeline
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Create encoders
    ordinal_encoder = OrdinalEncoder(
        categories=education_categories + income_categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    onehot_encoder = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    )
    
    # Create base preprocessor
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('ord', ordinal_encoder, ordinal_features),
            ('nom', onehot_encoder, nominal_features)
        ]
    )
    
    print("\nBase preprocessor created")
    print(f"  Numerical features: {len(numerical_features)}")
    print(f"  Ordinal features: {len(ordinal_features)}")
    print(f"  Nominal features: {len(nominal_features)}")
    
    return base_preprocessor


def train_founder_model(X_train_pool, X_val_s1, X_val_s2, y_train_pool, y_val_s1, y_val_s2,
                        base_preprocessor, random_state, base_temperature, ensemble_dir):
    """
    Train the founder model to establish baseline score.
    
    NOTE: The founder is NOT added to the ensemble - it only establishes
    the initial best score for hill climbing. This simplifies batch indexing.
    
    Parameters
    ----------
    X_train_pool, y_train_pool : arrays
        Training pool data
    X_val_s1, y_val_s1 : arrays
        Stage 1 validation data
    X_val_s2, y_val_s2 : arrays
        Stage 2 validation data (held out)
    base_preprocessor : ColumnTransformer
        Preprocessor for features
    random_state : int
        Random state
    base_temperature : float
        Base temperature for simulated annealing
    ensemble_dir : Path
        Directory to save models
    
    Returns
    -------
    float : Founder AUC on stage 2 validation set (baseline score)
    """
    print("=" * 80)
    print("TRAINING FOUNDER MODEL (baseline only - NOT added to ensemble)")
    print("=" * 80)
    
    # Simple 10% sample of training pool for founder model
    founder_sample_size = int(len(X_train_pool) * 0.10)
    
    # Sample from training pool
    X_train, _, y_train, _ = train_test_split(
        X_train_pool,
        y_train_pool,
        train_size=founder_sample_size,
        stratify=y_train_pool,
        random_state=random_state
    )
    
    print(f"\nTraining founder model")
    print("-" * 80)
    print(f"  Training samples: {len(X_train):,} (10% of {len(X_train_pool):,} pool)")
    
    # Generate random pipeline for founder
    pipeline, metadata = generate_random_pipeline(
        iteration=0,
        random_state=random_state,
        base_preprocessor=base_preprocessor,
        n_input_features=X_train.shape[1]
    )
    
    print(f"  Pipeline config:")
    print(f"    Classifier: {metadata['classifier_type']}")
    print(f"    Transformers: {metadata['transformers_used']}")
    print(f"    Dimensionality reduction: {metadata.get('dim_reduction', 'None')}")
    
    # Train on training sample
    print(f"  Training pipeline...")
    start_time = time.time()
    fitted_pipeline = pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"  Training complete ({training_time:.1f}s)")
    
    # Generate predictions on stage 1 validation set
    if hasattr(fitted_pipeline, 'predict_proba'):
        val_pred_s1 = fitted_pipeline.predict_proba(X_val_s1)[:, 1]
        val_pred_s2 = fitted_pipeline.predict_proba(X_val_s2)[:, 1]
    else:
        val_pred_s1 = fitted_pipeline.decision_function(X_val_s1)
        val_pred_s2 = fitted_pipeline.decision_function(X_val_s2)
    
    # Calculate validation AUCs
    val_auc_s1 = roc_auc_score(y_val_s1, val_pred_s1)
    founder_auc = roc_auc_score(y_val_s2, val_pred_s2)
    
    print(f"  Stage 1 validation AUC: {val_auc_s1:.6f}")
    print(f"  Stage 2 validation AUC: {founder_auc:.6f}")
    
    # Log founder (model not saved - used for baseline score only)
    pipeline_hash = compute_pipeline_hash(fitted_pipeline, metadata)
    log_iteration(
        iteration=0,
        accepted=True,
        rejection_reason='founder_baseline',
        pipeline_hash=pipeline_hash,
        stage1_val_auc=val_auc_s1,
        stage2_val_auc=founder_auc,
        ensemble_size=0,  # NOT included in ensemble
        diversity_score=0.0,
        temperature=base_temperature,
        metadata=metadata,
        ensemble_id="founder"
    )
    
    print(f"\n{'=' * 80}")
    print("FOUNDER MODEL COMPLETE - Baseline score established")
    print(f"{'=' * 80}")
    
    # Return only the score, not the model
    return founder_auc
