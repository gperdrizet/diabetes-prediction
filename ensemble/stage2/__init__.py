"""Stage 2: Deep neural network meta-learner.

This subpackage handles Stage 2 DNN training:
- Model architecture and building
- Training with early stopping
- Keras Tuner optimization
- Pseudo-labeling for data augmentation
"""

from .model import (
    build_stage2_dnn,
    build_from_config,
    build_architecture_units,
    build_for_tuning,
    build_from_hyperparameters
)
from .trainer import (
    train_stage2_dnn,
    evaluate_ensemble,
    evaluate_ensemble_with_cm,
    generate_stage2_training_data
)
from .optimizer import optimize_hyperparameters
from .pseudo_labeling import generate_pseudo_labels, augment_training_pool

__all__ = [
    # Model building
    'build_stage2_dnn',
    'build_from_config',
    'build_architecture_units',
    'build_for_tuning',
    'build_from_hyperparameters',
    # Training and evaluation
    'train_stage2_dnn',
    'evaluate_ensemble',
    'evaluate_ensemble_with_cm',
    'generate_stage2_training_data',
    # Optimization
    'optimize_hyperparameters',
    # Pseudo-labeling
    'generate_pseudo_labels',
    'augment_training_pool'
]
