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
        - X_val_s1, y_val_s1: 20% for evaluating stage 1 models and training stage 2
        - X_val_s2, y_val_s2: 20% for evaluating stage 2 model (held out)
    """
    X_full = train_df.drop(columns=[label])
    y_full = train_df[label]
    
    # First split: training pool vs validation
    X_train_pool, X_val_combined, y_train_pool, y_val_combined = train_test_split(
        X_full, 
        y_full, 
        test_size=0.4,
        random_state=random_state,
        stratify=y_full
    )
    
    # Second split: stage 1 validation vs stage 2 validation
    X_val_s1, X_val_s2, y_val_s1, y_val_s2 = train_test_split(
        X_val_combined,
        y_val_combined,
        test_size=0.5,
        random_state=random_state,
        stratify=y_val_combined
    )
    
    print(f"\nFixed data split:")
    print("-" * 80)
    print(f"  Training pool: {len(X_train_pool):,} samples (60%)")
    print(f"  Stage 1 validation: {len(X_val_s1):,} samples (20%) - for stage 1 eval & stage 2 training")
    print(f"  Stage 2 validation: {len(X_val_s2):,} samples (20%) - for stage 2 eval (HELD OUT)")
    
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
        ('clipper', IQRClipper(iqr_multiplier=2.0)),
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
    Train the founder model for the ensemble.
    
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
    tuple : (fitted_pipeline, founder_auc)
        - fitted_pipeline: trained founder model
        - founder_auc: AUC on stage 2 validation set
    """
    print("=" * 80)
    print("INITIALIZING FOUNDER MODEL")
    print("=" * 80)
    
    # Random sample size for founder training
    rng = np.random.RandomState(random_state)
    founder_sample_size = rng.randint(10000, 50001)
    
    # Sample from training pool
    X_train, _, y_train, _ = train_test_split(
        X_train_pool,
        y_train_pool,
        train_size=founder_sample_size,
        stratify=y_train_pool
    )
    
    print(f"\nTraining founder model")
    print("-" * 80)
    print(f"  Training samples: {len(X_train):,}")
    
    # Generate random pipeline for founder
    pipeline, metadata = generate_random_pipeline(
        iteration=0,
        random_state=random_state,
        base_preprocessor=base_preprocessor
    )
    
    print(f"  Pipeline config:")
    print(f"    Classifier: {metadata['classifier_type']}")
    print(f"    Transformers: {metadata['transformers_used']}")
    print(f"    Use PCA: {metadata['use_pca']}")
    
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
    
    # Save founder model
    model_path = ensemble_dir / 'founder_model.joblib'
    joblib.dump(fitted_pipeline, model_path)
    
    # Log founder
    pipeline_hash = compute_pipeline_hash(fitted_pipeline, metadata)
    log_iteration(
        iteration=0,
        accepted=True,
        rejection_reason='founder',
        pipeline_hash=pipeline_hash,
        stage1_val_auc=val_auc_s1,
        stage2_val_auc=founder_auc,
        ensemble_size=1,
        diversity_score=0.0,
        temperature=base_temperature,
        metadata=metadata,
        ensemble_id="founder"
    )
    
    print(f"\n{'=' * 80}")
    print("FOUNDER MODEL COMPLETE")
    print(f"{'=' * 80}")
    
    return fitted_pipeline, founder_auc
