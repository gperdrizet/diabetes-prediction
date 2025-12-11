"""Parallel execution for ensemble training.

This package provides batch scheduling, worker management, and timeout
handling for parallel training of ensemble candidates.
"""

from .scheduler import (
    prepare_training_batch,
    get_batch_info
)

from .worker import (
    train_single_candidate,
    train_batch_parallel
)

__all__ = [
    'prepare_training_batch',
    'get_batch_info',
    'train_single_candidate',
    'train_batch_parallel'
]
