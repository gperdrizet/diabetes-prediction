# Ensemble Model Configuration

This directory contains centralized configuration for all ensemble classifiers and their hyperparameters.

## Files

### `ensemble_config.py`
The main configuration file containing all classifier definitions, hyperparameter ranges, and sampling configurations. This file is used by the training pipeline.

**Key components:**
- `ACTIVE_CLASSIFIERS`: List of 10 currently active classifiers
- `DISABLED_CLASSIFIERS`: List of 3 disabled classifiers (too slow)
- `CLASSIFIER_CONFIGS`: Detailed configurations for each classifier including:
  - sklearn class reference
  - Status (active/disabled)
  - Hyperparameter generation functions
  - Performance notes and optimization history
- `SAMPLING_CONFIG`: Row and column sampling ranges
- `FEATURE_ENGINEERING_CONFIG`: Feature transformer settings
- `DIM_REDUCTION_CONFIG`: Dimensionality reduction settings

### `view_config.py`
Standalone viewer script that displays the configuration without requiring sklearn. Can be run directly:

```bash
python3 notebooks/functions/view_config.py
```

## Quick Reference

### Active Classifiers (10)

1. **logistic** - LogisticRegression
   - Fast L2-regularized linear model
   - Optimized: max_iter=100-300, C=0.1-10, removed L1 penalty
   
2. **random_forest** - RandomForestClassifier
   - Ensemble of decision trees, parallelizable
   - n_estimators=10-100, max_depth up to 20
   
3. **linear_svc** - LinearSVC
   - Fast linear SVM
   - Optimized: dual=False for many features, max_iter=200-300
   
4. **sgd_classifier** - SGDClassifier
   - Stochastic gradient descent, fast online learning
   - Multiple loss functions, max_iter=300-800
   
5. **extra_trees** - ExtraTreesClassifier
   - Extremely randomized trees
   - Optimized: n_estimators=5-30, max_depth≤10
   
6. **adaboost** - AdaBoostClassifier
   - Adaptive boosting, sequential
   - n_estimators=10-100, learning_rate=0.1-3.0
   
7. **naive_bayes** - Gaussian/Multinomial/Bernoulli
   - Probabilistic classifier with 3 variants
   - Fast, works well with small data
   
8. **lda** - LinearDiscriminantAnalysis
   - Fast, assumes Gaussian with shared covariance
   - solver=[svd, lsqr, eigen]
   
9. **qda** - QuadraticDiscriminantAnalysis
   - More flexible than LDA, non-linear boundaries
   - reg_param=0.0001-1.0
   
10. **ridge** - RidgeClassifier
    - Fast L2-regularized linear model
    - alpha=0.01-100

### Disabled Classifiers (3)

These cause 30+ minute timeouts and hold up entire batches:

1. **gradient_boosting** - Sequential tree building too slow
2. **mlp** - Neural network training too slow
3. **knn** - Distance calculations too slow

## Sampling Configuration

- **Row Sampling**: 10-40% (2,400-9,600 rows from ~24,000 pool)
  - Higher = stronger models, lower = more diversity
  
- **Column Sampling**: 30-70% of features
  - Random feature selection for diversity

## Feature Engineering

- **Skip Probability**: 30% (use raw features only)
- **Transformers Applied**: 1-3 when not skipping
- **Available**: 16 transformer types including:
  - Arithmetic: ratio, product, difference, sum
  - Math: reciprocal, square, sqrt, log
  - Binning, KDE smoothing, K-Means clustering
  - Kernel approximation: Nystroem, RBF sampler
  - Scaling: power transform, quantile transform
  - Noise injection for diversity

## Usage

### In Notebook
```python
from functions.ensemble_config import print_classifier_summary, ACTIVE_CLASSIFIERS

# Print full summary
print_classifier_summary()

# Get list of active classifiers
classifiers = ACTIVE_CLASSIFIERS
```

### In Training Pipeline
The configuration is automatically imported and used by `ensemble_hill_climbing.py`:

```python
from .ensemble_config import (
    ACTIVE_CLASSIFIERS,
    CLASSIFIER_CONFIGS,
    SAMPLING_CONFIG
)
```

## Modifying Configuration

To experiment with different configurations:

1. **Enable/Disable Classifiers**: 
   - Move classifier names between `ACTIVE_CLASSIFIERS` and `DISABLED_CLASSIFIERS`
   - Update corresponding status in `CLASSIFIER_CONFIGS`

2. **Adjust Hyperparameters**:
   - Modify lambda functions in `CLASSIFIER_CONFIGS[classifier_name]['hyperparameters']`
   - Example: Change C range for logistic regression:
     ```python
     'C': lambda rng: 10 ** rng.uniform(-2, 2),  # 0.01 to 100
     ```

3. **Modify Sampling Ranges**:
   - Update `SAMPLING_CONFIG['row_sample_pct']` or `col_sample_pct`
   - Example: Increase row sampling for stronger models:
     ```python
     'min': 0.20,  # 20% instead of 10%
     'max': 0.60,  # 60% instead of 40%
     ```

4. **Feature Engineering**:
   - Adjust `skip_probability` to control raw vs engineered features
   - Modify `n_transformers` range for complexity control

## Performance Notes

All optimizations are documented in the `notes` field of each classifier configuration. Recent changes:

- **Logistic**: Reduced max_iter from 500-1000 to 100-300 (3-10x faster)
- **Linear SVC**: Set dual=False, reduced max_iter (2-5x faster)
- **Extra Trees**: Reduced n_estimators from 10-100 to 5-30 (2-3x faster)
- **Row Sampling**: Increased from 1.25-15% to 10-40% (stronger models)
- **Temperature**: Increased from 0.01 to 0.05 for better exploration

## Re-enabling Disabled Classifiers

To re-enable a disabled classifier with tighter constraints:

1. Move name from `DISABLED_CLASSIFIERS` to `ACTIVE_CLASSIFIERS`
2. Update status: `'status': 'active'`
3. Tighten hyperparameters to prevent timeouts:
   ```python
   'gradient_boosting': {
       'hyperparameters': {
           'max_iter': lambda rng: rng.choice([10, 20, 30]),  # Very conservative
           'max_depth': lambda rng: rng.choice([3, 5]),  # Shallow only
       }
   }
   ```
