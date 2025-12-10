"""Base preprocessing utilities for ensemble training.

This module provides clean functions for creating the base preprocessor
that is shared across all Stage 1 models.
"""

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline


def create_base_preprocessor(
    numerical_features: List[str],
    ordinal_features: List[str],
    nominal_features: List[str],
    ordinal_categories: Optional[List[List]] = None
) -> ColumnTransformer:
    """Create base preprocessor shared across all Stage 1 models.
    
    The base preprocessor handles:
    - Numerical features: StandardScaler
    - Ordinal features: OrdinalEncoder with specified categories
    - Nominal features: OneHotEncoder with drop='first'
    
    Parameters
    ----------
    numerical_features : list of str
        Names of numerical feature columns.
    ordinal_features : list of str
        Names of ordinal feature columns.
    nominal_features : list of str
        Names of nominal feature columns.
    ordinal_categories : list of lists, optional
        Categories for each ordinal feature in order.
        If None, will infer from data.
    
    Returns
    -------
    preprocessor : ColumnTransformer
        Configured base preprocessor.
    """
    # Create numerical pipeline
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Create ordinal encoder
    if ordinal_categories is not None:
        ordinal_encoder = OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
    else:
        ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
    
    # Create one-hot encoder
    onehot_encoder = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    )
    
    # Create base preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('ord', ordinal_encoder, ordinal_features),
            ('nom', onehot_encoder, nominal_features)
        ]
    )
    
    return preprocessor


def get_preprocessor_info(preprocessor: ColumnTransformer) -> dict:
    """Get information about a preprocessor's configuration.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessor to inspect.
    
    Returns
    -------
    info : dict
        Dictionary with preprocessor information.
    """
    info = {
        'n_transformers': len(preprocessor.transformers),
        'transformers': []
    }
    
    for name, transformer, columns in preprocessor.transformers:
        transformer_info = {
            'name': name,
            'type': type(transformer).__name__,
            'n_features': len(columns) if isinstance(columns, list) else 1
        }
        info['transformers'].append(transformer_info)
    
    return info


def print_preprocessor_summary(
    preprocessor: ColumnTransformer,
    numerical_features: List[str],
    ordinal_features: List[str],
    nominal_features: List[str]
) -> None:
    """Print a summary of the preprocessor configuration.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessor to summarize.
    numerical_features : list of str
        Names of numerical features.
    ordinal_features : list of str
        Names of ordinal features.
    nominal_features : list of str
        Names of nominal features.
    """
    print("\nBase Preprocessor Configuration")
    print("=" * 60)
    print(f"Total input features: {len(numerical_features) + len(ordinal_features) + len(nominal_features)}")
    print()
    print("Feature types:")
    print(f"  Numerical (StandardScaler):  {len(numerical_features)} features")
    if numerical_features:
        print(f"    {', '.join(numerical_features[:5])}" + 
              (f" ... (+{len(numerical_features)-5} more)" if len(numerical_features) > 5 else ""))
    
    print(f"  Ordinal (OrdinalEncoder):    {len(ordinal_features)} features")
    if ordinal_features:
        print(f"    {', '.join(ordinal_features)}")
    
    print(f"  Nominal (OneHotEncoder):     {len(nominal_features)} features")
    if nominal_features:
        print(f"    {', '.join(nominal_features)}")
    
    print("=" * 60)
