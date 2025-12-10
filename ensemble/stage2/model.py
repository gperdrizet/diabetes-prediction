"""Stage 2 DNN model building and architecture.

This module provides clean functions for building the Stage 2 deep neural network
meta-learner that combines predictions from Stage 1 models.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers


def build_stage2_dnn(
    n_models: int,
    n_layers: int = 2,
    units_per_layer: int = 128,
    dropout: float = 0.3,
    batch_norm: bool = True,
    activation: str = 'relu',
    learning_rate: float = 0.001
) -> models.Sequential:
    """Build stage 2 DNN meta-learner (legacy interface).
    
    Parameters
    ----------
    n_models : int
        Number of stage 1 models (input dimension).
    n_layers : int, default=2
        Number of hidden layers.
    units_per_layer : int, default=128
        Units per hidden layer.
    dropout : float, default=0.3
        Dropout rate.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='relu'
        Activation function.
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer.
    
    Returns
    -------
    model : Sequential
        Compiled Keras model.
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(n_models,)))
    
    # Hidden layers
    for i in range(n_layers):
        model.add(layers.Dense(units_per_layer, activation=activation))
        
        if batch_norm:
            model.add(layers.BatchNormalization())
        
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')]
    )
    
    return model


def build_from_config(config: Dict, n_models: int) -> models.Sequential:
    """Build model from Stage2Config.
    
    Parameters
    ----------
    config : Stage2Config
        Configuration object with architecture and training specs.
    n_models : int
        Number of stage 1 models (input dimension).
    
    Returns
    -------
    model : Sequential
        Compiled Keras model.
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(n_models,)))
    
    # Hidden layers from config
    arch = config.architecture
    for layer_cfg in arch.hidden_layers:
        # Dense layer with optional L2 regularization
        if layer_cfg.l2_reg > 0:
            model.add(layers.Dense(
                layer_cfg.units,
                activation=layer_cfg.activation,
                kernel_regularizer=regularizers.l2(layer_cfg.l2_reg)
            ))
        else:
            model.add(layers.Dense(
                layer_cfg.units,
                activation=layer_cfg.activation
            ))
        
        # Dropout
        model.add(layers.Dropout(layer_cfg.dropout))
    
    # Output layer
    output_cfg = arch.output
    model.add(layers.Dense(
        output_cfg.units,
        activation=output_cfg.activation
    ))
    
    # Compile with config settings
    train_cfg = config.training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=train_cfg.learning_rate),
        loss=train_cfg.loss,
        metrics=[keras.metrics.AUC(name='auc')]
    )
    
    return model


def build_architecture_units(
    n_models: int,
    arch_type: str,
    base_units: int,
    n_layers: int
) -> List[int]:
    """Generate layer unit configuration based on architecture type.
    
    Parameters
    ----------
    n_models : int
        Number of input models (input dimension).
    arch_type : str
        Architecture type: 'uniform', 'pyramid', 'funnel', or 'constant'.
    base_units : int
        Base number of units for scaling.
    n_layers : int
        Number of hidden layers.
    
    Returns
    -------
    units_per_layer : list of int
        Units for each layer.
    """
    if arch_type in ('uniform', 'constant'):
        # All layers same size
        return [base_units] * n_layers
    
    elif arch_type == 'pyramid':
        # Increasing then decreasing
        if n_layers == 1:
            return [base_units]
        elif n_layers == 2:
            return [base_units // 2, base_units]
        else:  # n_layers == 3
            return [base_units // 2, base_units, base_units // 2]
    
    elif arch_type == 'funnel':
        # Decreasing units
        if n_layers == 1:
            return [base_units]
        elif n_layers == 2:
            return [base_units, base_units // 2]
        else:  # n_layers == 3
            return [base_units, base_units // 2, base_units // 4]
    
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def build_for_tuning(hp, n_models: int) -> Tuple[models.Sequential, Dict]:
    """Build model for Keras Tuner hyperparameter optimization.
    
    This function is used internally by the optimizer. After optimization,
    use build_from_hyperparameters() to rebuild the winning model.
    
    Parameters
    ----------
    hp : HyperParameters
        Keras Tuner hyperparameters object.
    n_models : int
        Number of stage 1 models (input dimension).
    
    Returns
    -------
    model : Sequential
        Compiled Keras model.
    hyperparameters : dict
        Hyperparameter values used.
    """
    # Sample hyperparameters
    arch_type = hp.Choice('architecture_type', values=['funnel', 'constant', 'pyramid'])
    n_layers = hp.Int('n_layers', min_value=1, max_value=3)
    base_units = hp.Choice('base_units', values=[16, 32, 64, 128])
    dropout = hp.Float('dropout', min_value=0.2, max_value=0.7)
    l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log')
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, sampling='log')
    
    # Generate layer units based on architecture type
    units_per_layer = build_architecture_units(n_models, arch_type, base_units, n_layers)
    
    # Build model
    model = models.Sequential()
    model.add(layers.Input(shape=(n_models,)))
    
    for units in units_per_layer:
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')]
    )
    
    # Extract hyperparameters for logging
    hyperparameters = {
        'architecture_type': arch_type,
        'n_layers': n_layers,
        'base_units': base_units,
        'units_per_layer': units_per_layer,
        'dropout': dropout,
        'l2_reg': l2_reg,
        'learning_rate': learning_rate
    }
    
    return model, hyperparameters


def build_from_hyperparameters(
    hyperparameters: Dict,
    n_models: int
) -> Tuple[models.Sequential, List[int]]:
    """Build model from hyperparameter dictionary (after optimization).
    
    Use this to rebuild the winning model after hyperparameter optimization.
    
    Parameters
    ----------
    hyperparameters : dict
        Hyperparameters with keys: architecture_type, n_layers, base_units,
        dropout, l2_reg, learning_rate.
    n_models : int
        Number of stage 1 models (input dimension).
    
    Returns
    -------
    model : Sequential
        Compiled Keras model.
    units_per_layer : list of int
        Units for each layer (for inspection).
    """
    # Extract hyperparameters
    arch_type = hyperparameters['architecture_type']
    n_layers = hyperparameters['n_layers']
    base_units = hyperparameters['base_units']
    dropout = hyperparameters['dropout']
    l2_reg = hyperparameters['l2_reg']
    learning_rate = hyperparameters['learning_rate']
    
    # Generate layer units
    units_per_layer = build_architecture_units(n_models, arch_type, base_units, n_layers)
    
    # Build model
    model = models.Sequential()
    model.add(layers.Input(shape=(n_models,)))
    
    for units in units_per_layer:
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc'), 'accuracy']
    )
    
    return model, units_per_layer
