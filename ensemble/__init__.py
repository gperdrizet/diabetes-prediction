"""Ensemble Hill Climbing System.

A refactored ensemble learning system featuring:
- Stage 1: Diverse sklearn models via simulated annealing hill climbing
- Stage 2: Deep neural network meta-learner with pseudo-labeling
- Batch parallel training with timeout protection
- Real-time monitoring via SQLite database and Streamlit dashboard

This package provides clean abstractions for all components with clear
separation of concerns and single-responsibility modules.
"""

__version__ = "2.0.0"
__author__ = "Ensemble System Team"

from ensemble.config import EnsembleConfig

__all__ = ['EnsembleConfig']
