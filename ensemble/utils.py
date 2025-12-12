"""Utility functions for ensemble training."""

import hashlib
import json
from typing import Any, Dict
from sklearn.pipeline import Pipeline


def compute_pipeline_hash(pipeline: Pipeline, metadata: Dict[str, Any] = None) -> str:
    """Compute a hash of the pipeline configuration for tracking.
    
    Parameters
    ----------
    pipeline : Pipeline
        sklearn pipeline.
    metadata : dict, optional
        Pipeline metadata. If None, will extract from pipeline.
    
    Returns
    -------
    hash_str : str
        SHA256 hash of pipeline configuration.
    """
    if metadata is None:
        metadata = {}
    
    # Create string representation of pipeline config
    config_str = json.dumps({
        'transformers': metadata.get('transformers_used', []),
        'classifier': metadata.get('classifier_type', ''),
        'dim_reduction': metadata.get('dim_reduction'),
        'row_sample_pct': round(metadata.get('row_sample_pct', 0), 4),
        'col_sample_pct': round(metadata.get('col_sample_pct', 0), 4),
        'pipeline_steps': [step[0] for step in pipeline.steps] if hasattr(pipeline, 'steps') else []
    }, sort_keys=True)
    
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
