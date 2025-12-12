"""Structured logging utilities for ensemble training.

This module provides clean logging functions to replace scattered print statements
with proper structured logging.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = 'ensemble',
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Setup a logger with consistent formatting.
    
    Parameters
    ----------
    name : str, default='ensemble'
        Logger name.
    level : int, default=logging.INFO
        Logging level.
    log_file : Path, optional
        Path to log file. If provided, logs will be written to both console and file.
        File will be overwritten (mode='w') to start fresh each run.
    
    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to start fresh
    logger.handlers.clear()
    
    # Format: [2025-12-10 10:30:45] INFO: Message
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if requested) - use 'w' mode to overwrite and start fresh
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_phase_start(logger: logging.Logger, phase_name: str, details: str = "") -> None:
    """Log the start of a major phase.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    phase_name : str
        Name of the phase.
    details : str, optional
        Additional details.
    """
    separator = "=" * 80
    logger.info(separator)
    logger.info(f"{phase_name.upper()}")
    if details:
        logger.info(details)
    logger.info(separator)


def log_phase_end(logger: logging.Logger, phase_name: str, elapsed_time: float = None) -> None:
    """Log the end of a major phase.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    phase_name : str
        Name of the phase.
    elapsed_time : float, optional
        Time elapsed in seconds.
    """
    separator = "=" * 80
    logger.info(separator)
    msg = f"{phase_name.upper()} COMPLETE"
    if elapsed_time is not None:
        msg += f" ({elapsed_time:.1f}s)"
    logger.info(msg)
    logger.info(separator)


def log_iteration(
    logger: logging.Logger,
    iteration: int,
    accepted: bool,
    reason: str,
    metrics: Dict[str, Any]
) -> None:
    """Log a hill climbing iteration.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    iteration : int
        Iteration number.
    accepted : bool
        Whether candidate was accepted.
    reason : str
        Acceptance/rejection reason.
    metrics : dict
        Dictionary with metrics (e.g., stage1_auc, stage2_auc, diversity).
    """
    status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
    logger.info(f"Iteration {iteration}: {status} - {reason}")
    
    # Log key metrics
    if 'stage1_auc' in metrics:
        logger.info(f"  Stage 1 AUC: {metrics['stage1_auc']:.6f}")
    if 'stage2_auc' in metrics:
        logger.info(f"  Stage 2 AUC: {metrics['stage2_auc']:.6f}")
    if 'diversity' in metrics:
        logger.info(f"  Diversity: {metrics['diversity']:.6f}")
    if 'temperature' in metrics:
        logger.info(f"  Temperature: {metrics['temperature']:.6f}")


def log_training_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    message: str = "Training progress"
) -> None:
    """Log training progress.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    current : int
        Current iteration/epoch.
    total : int
        Total iterations/epochs.
    message : str, default='Training progress'
        Progress message.
    """
    pct = (current / total * 100) if total > 0 else 0
    logger.info(f"{message}: {current}/{total} ({pct:.1f}%)")


def log_performance_metrics(
    logger: logging.Logger,
    metrics: Dict[str, float],
    prefix: str = ""
) -> None:
    """Log performance metrics in a structured way.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    metrics : dict
        Dictionary of metric name to value.
    prefix : str, optional
        Prefix for log messages.
    """
    if prefix:
        logger.info(f"{prefix}:")
    
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.6f}")
        else:
            logger.info(f"  {name}: {value}")


def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """Log an error with context.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    error : Exception
        The exception that occurred.
    context : str, optional
        Additional context about where the error occurred.
    """
    if context:
        logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")
    else:
        logger.error(f"{type(error).__name__}: {str(error)}")


def log_warning(logger: logging.Logger, message: str) -> None:
    """Log a warning message.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    message : str
        Warning message.
    """
    logger.warning(f"⚠️  {message}")


def log_success(logger: logging.Logger, message: str) -> None:
    """Log a success message.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    message : str
        Success message.
    """
    logger.info(f"✓ {message}")
