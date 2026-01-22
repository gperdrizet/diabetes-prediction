"""Checkpoint management for ensemble training.

Provides utilities for saving training checkpoints with complete
inference artifacts (models, metadata, configuration) for deployment
and Kaggle submissions.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import joblib
import shutil


def save_inference_checkpoint(
    checkpoint_num: int,
    ensemble_models: List,
    stage2_model,
    run_dir: Path,
    stage1_auc: float,
    stage2_auc: float,
    retraining_count: int,
    pseudo_label_stats: Optional[Dict] = None,
    config: Optional[Any] = None,
    logger: Optional[Any] = None
) -> Path:
    """Save complete inference checkpoint for Kaggle submission.
    
    Creates a numbered checkpoint directory containing all artifacts
    needed for standalone inference:
    - ensemble_stage1_models.joblib (Stage 1 models)
    - stage2_model.h5 (Stage 2 DNN model)
    - metadata.json (checkpoint information)
    
    Parameters
    ----------
    checkpoint_num : int
        Checkpoint number (typically accepted_count).
    ensemble_models : list
        List of fitted Stage 1 pipeline models.
    stage2_model : keras.Model
        Fitted Stage 2 DNN model.
    run_dir : Path
        Run directory path.
    stage1_auc : float
        Stage 1 validation AUC.
    stage2_auc : float
        Stage 2 validation AUC.
    retraining_count : int
        Number of Stage 2 retrainings.
    pseudo_label_stats : dict, optional
        Pseudo-labeling statistics.
    config : EnsembleConfig, optional
        Configuration object.
    logger : logging.Logger, optional
        Logger instance for progress messages.
        
    Returns
    -------
    checkpoint_dir : Path
        Path to created checkpoint directory.
        
    Notes
    -----
    - Checkpoints are saved in run_dir/checkpoints/checkpoint_XXX/
    - Stage 2 model saved in H5 format for compatibility
    - Metadata includes architecture, performance metrics, and config
    
    Examples
    --------
    >>> checkpoint_dir = save_inference_checkpoint(
    ...     checkpoint_num=30,
    ...     ensemble_models=current_ensemble,
    ...     stage2_model=stage2_model,
    ...     run_dir=Path('models/run_20251231_120000'),
    ...     stage1_auc=0.7145,
    ...     stage2_auc=0.7289,
    ...     retraining_count=1,
    ...     logger=logger
    ... )
    """
    # Create checkpoint directory
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = checkpoints_dir / f'checkpoint_{checkpoint_num:03d}'
    checkpoint_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    # 1. Save Stage 2 DNN model (H5 format for compatibility)
    stage2_path = checkpoint_dir / 'stage2_model.h5'
    stage2_model.save(stage2_path, save_format='h5')
    stage2_size = stage2_path.stat().st_size / (1024**2)  # MB
    
    # 2. Save Stage 1 ensemble (joblib)
    ensemble_path = checkpoint_dir / 'ensemble_stage1_models.joblib'
    joblib.dump(ensemble_models, ensemble_path)
    ensemble_size = ensemble_path.stat().st_size / (1024**2)  # MB
    
    # 3. Create metadata
    metadata = {
        'checkpoint_num': checkpoint_num,
        'timestamp': datetime.now().isoformat(),
        'ensemble_size': len(ensemble_models),
        'stage1_val_auc': float(stage1_auc),
        'stage2_val_auc': float(stage2_auc),
        'retraining_count': retraining_count,
        'pseudo_labeling': pseudo_label_stats or {'enabled': False},
        'config_snapshot': config.__dict__ if hasattr(config, '__dict__') else str(config)
    }
    
    metadata_path = checkpoint_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    total_size = stage2_size + ensemble_size
    
    # Log checkpoint creation
    if logger:
        logger.info(f'✓ Saved checkpoint_{checkpoint_num:03d}:')
        logger.info(f'  Path: {checkpoint_dir.relative_to(run_dir.parent)}')
        logger.info(f'  Stage 1 ensemble: {ensemble_size:.1f} MB ({len(ensemble_models)} models)')
        logger.info(f'  Stage 2 DNN: {stage2_size:.1f} MB')
        logger.info(f'  Total size: {total_size:.1f} MB')
        logger.info(f'  Save time: {elapsed:.1f}s')
    
    return checkpoint_dir
