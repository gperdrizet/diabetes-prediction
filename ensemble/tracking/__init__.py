"""Tracking and monitoring utilities.

This subpackage handles:
- SQLite database operations
- Structured logging
"""

from .database import EnsembleDatabase
from .checkpoints import save_inference_checkpoint
from .logger import (
    setup_logger,
    log_phase_start,
    log_phase_end,
    log_iteration,
    log_training_progress,
    log_performance_metrics,
    log_error,
    log_warning,
    log_success
)

__all__ = [
    # Database
    'EnsembleDatabase',
    # Checkpoints
    'save_inference_checkpoint',
    # Logging
    'setup_logger',
    'log_phase_start',
    'log_phase_end',
    'log_iteration',
    'log_training_progress',
    'log_performance_metrics',
    'log_error',
    'log_warning',
    'log_success'
]
