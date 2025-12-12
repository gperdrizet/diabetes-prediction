"""Stage 1: Diverse sklearn model training.

This subpackage handles all Stage 1 (base model) training:
- Classifier pool with hyperparameter sampling
- Custom sklearn transformers for feature engineering
- Pipeline builder for generating random diverse pipelines
- Training with timeout protection
"""

from ensemble.stage1.classifiers import ClassifierPool
from ensemble.stage1.transformers import get_transformer
from ensemble.stage1.pipeline_builder import PipelineBuilder

__all__ = [
    'ClassifierPool',
    'get_transformer',
    'PipelineBuilder'
]
