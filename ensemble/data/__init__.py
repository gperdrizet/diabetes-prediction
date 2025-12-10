"""Data management utilities.

This subpackage handles data operations:
- Train/val/test splitting
- Base preprocessing
"""

from .splits import DataSplits, create_three_way_split
from .preprocessing import (
    create_base_preprocessor,
    get_preprocessor_info,
    print_preprocessor_summary
)

__all__ = [
    # Data splitting
    'DataSplits',
    'create_three_way_split',
    # Preprocessing
    'create_base_preprocessor',
    'get_preprocessor_info',
    'print_preprocessor_summary'
]
