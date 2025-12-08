# Pseudo-Labeling Implementation

## Overview

Pseudo-labeling is a semi-supervised learning technique that leverages unlabeled test data to improve model performance. After each Stage 2 DNN optimization cycle, high-confidence predictions on the test set are added to the training pool for Stage 1 models.

## Implementation

### Data Flow

```
1. Train ensemble (Stage 1 + Stage 2) on labeled data
2. Generate predictions on unlabeled test set  
3. Filter high-confidence predictions (p ≥ 0.95 or p ≤ 0.05)
4. Add pseudo-labeled samples to X_train_pool
5. Retrain Stage 1 models on augmented training pool
6. Stage 1 generates new predictions on validation sets
7. Stage 2 DNN trains on these new Stage 1 predictions
```

### Key Features

**Indirect Feedback Loop**
- Pseudo-labels → Stage 1 training pool (NOT Stage 2)
- Prevents direct circular feedback
- Ensemble acts as filter/regularizer

**High Confidence Threshold**
- Default: 0.95 (very conservative)
- Only includes predictions where model is very certain
- Minimizes propagation of errors

**Class Balancing**
- Ensures equal representation of both classes
- Prevents distribution shift
- Maintains model calibration

**Fraction Limiting**
- Max 15% of training pool can be pseudo-labeled
- Prevents test set from dominating training
- Maintains reliance on labeled data

## Configuration

In `02.1-ensemble_hill_climbing.ipynb`:

```python
# Pseudo-labeling configuration
PSEUDO_LABEL_ENABLED = True           # Enable pseudo-labeling
PSEUDO_CONFIDENCE_THRESHOLD = 0.95    # Only use very confident predictions
PSEUDO_MAX_FRACTION = 0.15            # Max 15% of training pool can be pseudo-labeled
PSEUDO_BALANCE_CLASSES = True         # Ensure balanced pseudo-labeled samples
```

## Functions

### `generate_pseudo_labels()`

Generates pseudo-labels from test set predictions.

**Location:** `notebooks/functions/ensemble_stage2_training.py`

**Parameters:**
- `ensemble_models`: List of trained Stage 1 models
- `stage2_model`: Trained Stage 2 DNN
- `test_df`: Unlabeled test data
- `confidence_threshold`: Minimum confidence (default 0.95)
- `balance_classes`: Balance positive/negative samples (default True)

**Returns:**
- `X_pseudo`: DataFrame of pseudo-labeled features
- `y_pseudo`: Series of pseudo-labels (0 or 1)
- `stats`: Dictionary with statistics

### `augment_training_pool_with_pseudo_labels()`

Combines original training pool with pseudo-labeled data.

**Location:** `notebooks/functions/ensemble_stage2_training.py`

**Parameters:**
- `X_train_pool`: Original training features
- `y_train_pool`: Original training labels  
- `X_pseudo`: Pseudo-labeled features
- `y_pseudo`: Pseudo-labels
- `max_pseudo_fraction`: Maximum fraction (default 0.20)

**Returns:**
- `X_augmented`: Combined training features
- `y_augmented`: Combined labels
- `stats`: Augmentation statistics

## Execution Flow

Pseudo-labeling runs automatically after each Stage 2 DNN retraining:

1. **Trigger:** Every `STAGE2_BATCH_SIZE_MODELS` accepted models (default: 20)
2. **Stage 2 DNN retrained** on current ensemble predictions
3. **Pseudo-labels generated** from test set using full ensemble
4. **Training pool augmented** with high-confidence pseudo-labeled samples
5. **Future Stage 1 models** train on augmented data
6. **Stage 2 DNN** sees new prediction patterns from augmented training

## Expected Benefits

1. **More diverse Stage 1 models** from larger training pool
2. **Better test set adaptation** - model learns test distribution
3. **Improved generalization** through semi-supervised learning
4. **Incremental improvement** as ensemble gets more confident

## Monitoring

Key metrics logged after each pseudo-labeling cycle:

- Number of high-confidence predictions
- Class distribution of pseudo-labels
- Mean prediction confidence
- Coverage of test set
- New training pool size
- Pseudo-label fraction

## Safety Mechanisms

1. **High confidence threshold** (0.95) - only very certain predictions
2. **Class balancing** - prevents distribution shift
3. **Fraction limiting** - max 15% of training can be pseudo-labeled
4. **Separate validation** - validation sets remain untouched
5. **Indirect feedback** - Stage 1 ensemble filters Stage 2 predictions

## Files Modified

- `notebooks/functions/ensemble_stage2_training.py` - Added pseudo-labeling functions
- `notebooks/02.1-ensemble_hill_climbing.ipynb` - Integrated pseudo-labeling into training loop
- Test data loaded from: `https://gperdrizet.github.io/FSA_devops/assets/data/unit3/diabetes_prediction_test.csv`

## Example Output

```
================================================================================
PSEUDO-LABELING: Generating labels from test set
================================================================================
Test set size: 100,000 samples
Confidence threshold: 0.95
Generating Stage 1 predictions...
Generating Stage 2 predictions...

High-confidence predictions:
  Total: 45,230 (45.2%)
  Positive (p ≥ 0.95): 22,115
  Negative (p ≤ 0.05): 23,115

Class balancing:
  Kept 22,115 samples per class
  Total pseudo-labeled: 44,230

Final pseudo-labeled dataset:
  Total samples: 44,230
  Positive: 22,115 (50.0%)
  Negative: 22,115 (50.0%)
  Mean confidence: 0.973
  Coverage: 44.2% of test set
================================================================================

================================================================================
AUGMENTING TRAINING POOL WITH PSEUDO-LABELS
================================================================================
Original training pool: 280,000 samples
Pseudo-labeled samples: 44,230 samples

Augmented training pool:
  Total size: 324,230
  Original: 280,000 (86.4%)
  Pseudo-labeled: 44,230 (13.6%)

Class distribution:
  Original positive rate: 0.082
  Pseudo positive rate: 0.500
  Augmented positive rate: 0.141
================================================================================
```
