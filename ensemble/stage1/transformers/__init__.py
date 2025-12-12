"""Custom transformers for Stage 1 ensemble models.

This package provides transformers organized by purpose:
- preprocessing: CleanNumericTransformer, RandomFeatureSelector, IQRClipper, ConstantFeatureRemover
- feature_engineering: Transformers from parent transformers.py module

The transformers from the parent module (transformers.py) are imported here for convenience.
"""

# Import preprocessing transformers from preprocessing.py
from ensemble.stage1.transformers.preprocessing import (
    CleanNumericTransformer,
    RandomFeatureSelector,
    IQRClipper,
    ConstantFeatureRemover
)

# Import all feature engineering transformers from sibling feature_transformers.py
from ensemble.stage1.feature_transformers import (
    RatioTransformer,
    ProductTransformer,
    DifferenceTransformer,
    SumTransformer,
    ReciprocalTransformer,
    SquareTransformer,
    SquareRootTransformer,
    LogTransformer,
    BinningTransformer,
    KDESmoothingTransformer,
    KMeansClusterTransformer,
    NoiseInjector,
    get_transformer
)

__all__ = [
    # Preprocessing transformers
    'CleanNumericTransformer',
    'RandomFeatureSelector',
    'IQRClipper',
    'ConstantFeatureRemover',
    # Feature engineering transformers
    'RatioTransformer',
    'ProductTransformer',
    'DifferenceTransformer',
    'SumTransformer',
    'ReciprocalTransformer',
    'SquareTransformer',
    'SquareRootTransformer',
    'LogTransformer',
    'BinningTransformer',
    'KDESmoothingTransformer',
    'KMeansClusterTransformer',
    'NoiseInjector',
    'get_transformer'
]
