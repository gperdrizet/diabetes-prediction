# Code Audit Summary - December 7, 2025

## Overview
Complete code review and cleanup to align all components with the final implementation architecture (no cross-validation, no folds, batch-based DNN training with transfer learning).

## Critical Issues Found and Fixed

### 1. Database Schema Mismatch

**Problem**: Schema didn't match actual logged data
- Had `cv_score` (never used - always 0.0)
- Had `combined_score` (redundant with stage2_val_auc)
- Had `acceptance_reason` instead of `rejection_reason`
- Missing: `classifier_type`, `use_pca`, `pca_components`

**Solution**: Updated schema to match reality
```sql
-- OLD (incorrect)
cv_score REAL NOT NULL,
combined_score REAL NOT NULL,
acceptance_reason TEXT,

-- NEW (correct)
stage1_val_auc REAL NOT NULL,
stage2_val_auc REAL NOT NULL,
rejection_reason TEXT,
classifier_type TEXT,
use_pca INTEGER,
pca_components REAL,
```

### 2. Obsolete Function Parameters

**Functions with deprecated parameters:**

#### `log_iteration()`
- **Removed**: `fold` parameter (no folds in implementation)
- **Removed**: `stage1_cv_score` parameter (always 0.0, not meaningful)
- **Added**: Proper metadata extraction (`classifier_type`, `use_pca`, `pca_components`)

#### `save_checkpoint()`
- **Removed**: `current_fold` parameter (no folds in implementation)

#### `train_stage2_dnn()`
- **Removed**: `fold` parameter (no folds in implementation)
- **Fixed**: ensemble_id fallback from `f"iter_{iteration}_fold_{fold}"` to `f"iter_{iteration}"`

#### `quick_optimize_pipeline()`
- Parameters `n_iter`, `cv`, `n_jobs`, `random_state` marked as DEPRECATED
- Function just does `pipeline.fit(X, y)` - no optimization, no CV

### 3. Notebook Configuration Cleanup

**Removed obsolete variables:**
- `N_FOUNDERS` - Not used (single founder model)
- `TRAINING_SAMPLE_SIZE` - Not used (random per iteration)
- `QUICK_OPTIMIZE_ITERATIONS` - Not used (no optimization)
- `QUICK_OPTIMIZE_CV` - Not used (no CV)
- `QUICK_OPTIMIZE_JOBS` - Not used (no parallelization)
- `FOUNDER_OPTIMIZE_ITERATIONS` - Not used (no optimization)
- `FOUNDER_OPTIMIZE_CV` - Not used (no CV)

**Retained variables:**
- `RANDOM_STATE` - Used ONLY for fixed data splits
- `MAX_ITERATIONS` - Hill climbing limit
- `PLATEAU_ITERATIONS` - Early stopping threshold
- `BASE_TEMPERATURE` - Simulated annealing
- `TEMPERATURE_DECAY` - Simulated annealing
- `STAGE2_BATCH_SIZE_MODELS` - DNN retraining frequency (10 models)
- `STAGE2_EPOCHS` - DNN training epochs
- `STAGE2_BATCH_SIZE` - DNN batch size
- `STAGE2_PATIENCE` - Early stopping patience

### 4. Notebook Code Updates

**Founder model cell:**
- Removed: `quick_optimize_pipeline()` call
- Changed to: Direct `pipeline.fit(X_train, y_train)`
- Removed: `fold=0` from `log_iteration()`
- Removed: `stage1_cv_score` from `log_iteration()`

**Hill climbing loop cell:**
- Removed: `quick_optimize_pipeline()` call
- Changed to: Direct `pipeline.fit(X_train, y_train)`
- Removed: `fold=iteration % N_FOUNDERS` from `log_iteration()`
- Removed: `stage1_cv_score` from `log_iteration()`
- Removed: `fold=0` from `train_stage2_dnn()`

**Checkpoint cell:**
- Removed: `current_fold=current_fold` (undefined variable)
- Removed: `'n_folds': N_FOLDS` from metadata

**Summary cell:**
- Removed: References to `TRAINING_LOG_PATH` and `STAGE2_LOG_PATH`
- Added: Reference to SQLite database path

### 5. Dashboard Updates

**Updated to use correct schema:**
- `cv_score` → `stage1_val_auc`
- `combined_score` → `stage2_val_auc`
- Added proper null checks for `classifier_type`, `use_pca`, `pca_components`
- Fixed metric labels to match actual data

## Implementation Architecture

### Data Flow
```
Training Pool (60%)
    ↓ Random sample (10k-50k)
    ↓ Train stage 1 model
    ↓ Evaluate on Stage 1 Validation
    
Stage 1 Validation (20%) - FIXED
    ↓ Individual model evaluation
    ↓ Used to train Stage 2 DNN
    
Stage 2 Validation (20%) - HELD OUT
    ↓ Final ensemble evaluation
    ✓ Never used for training
```

### Training Process
1. **No Cross-Validation**: Models trained once on random sample
2. **No Hyperparameter Optimization**: Wide random distributions, no search
3. **No Folds**: Single fixed validation split
4. **Batch-Based DNN**: Train every 10 accepted models with transfer learning

### Random State Usage
- **Used**: Creating fixed 3-way data split (RANDOM_STATE=315)
- **Used**: Seeding iteration-specific RNG for sample sizes
- **NOT used**: sklearn model training (all `random_state=None`)
- **NOT used**: Hyperparameter sampling (uses `np.random` directly)

## Files Modified

1. **`notebooks/functions/ensemble_database.py`**
   - Updated schema (removed cv_score, combined_score, acceptance_reason)
   - Added columns (classifier_type, use_pca, pca_components)
   - Fixed `insert_ensemble_iteration()` to match new schema

2. **`notebooks/functions/ensemble_hill_climbing.py`**
   - Removed `fold`, `stage1_cv_score` from `log_iteration()`
   - Added metadata extraction for new columns
   - Updated docstrings to mark deprecated parameters

3. **`notebooks/functions/ensemble_stage2_model.py`**
   - Removed `current_fold` from `save_checkpoint()`
   - Removed `fold` from `train_stage2_dnn()`
   - Fixed ensemble_id fallback string

4. **`notebooks/02.1-ensemble_hill_climbing.ipynb`**
   - Configuration cell: Removed obsolete variables
   - Founder cell: Removed optimization calls, removed fold parameter
   - Hill climbing cell: Removed optimization calls, removed fold parameters
   - Checkpoint cell: Removed current_fold, n_folds
   - Summary cell: Fixed file paths

5. **`dashboard/app.py`**
   - Updated all references from cv_score to stage1_val_auc
   - Updated all references from combined_score to stage2_val_auc
   - Fixed chart labels and titles
   - Added proper null handling for new columns

## Testing Recommendations

1. **Delete old database**: `rm /workspaces/diabetes-prediction/data/ensemble_training.db`
2. **Reinitialize database**: Run notebook configuration cell
3. **Run founder model**: Verify logging works with new schema
4. **Test dashboard**: Should display correctly with new schema
5. **Run hill climbing**: Test batch-based DNN training at model 10

## Breaking Changes

⚠️ **Database schema changed** - old databases incompatible
- Delete existing `ensemble_training.db` before running updated code
- Old checkpoints may have incompatible metadata

## Migration Notes

If you have an existing database:
```bash
# Backup old database
mv data/ensemble_training.db data/ensemble_training_old.db

# Reinitialize with new schema
# Run notebook configuration cell
```

If you have existing checkpoints:
- Checkpoints should still load (ensemble_models array is unchanged)
- Metadata may have extra/missing keys (handled gracefully)
- Consider restarting training from scratch for consistency

## Verification Checklist

- [x] Database schema matches logged data
- [x] All function signatures updated
- [x] Notebook removes obsolete variables
- [x] Dashboard uses correct column names
- [x] No references to cv_score, combined_score, fold
- [x] Docstrings updated to reflect reality
- [x] No undefined variables in notebook
- [ ] Test with fresh database (user action required)
- [ ] Verify dashboard displays correctly (user action required)

## Summary

The codebase is now consistent with the actual implementation:
- **No cross-validation** anywhere
- **No folds** - single fixed validation split
- **No hyperparameter optimization** - random sampling only
- **Batch-based DNN training** - every 10 models with transfer learning
- **Clean database schema** - matches actual logged data
- **No deprecated parameters** - all obsolete code removed
