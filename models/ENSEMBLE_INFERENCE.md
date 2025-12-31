# Ensemble Model Inference Guide

This guide explains how to prepare checkpoint assets and run inference for Kaggle submissions.

## Overview

The ensemble model uses a **checkpoint-based** inference system that allows you to:
- Save complete model snapshots after each Stage 2 retraining
- Submit intermediate results without waiting for full training completion
- Choose which checkpoint to submit based on validation performance
- Upload checkpoints to Kaggle as datasets for inference

## Checkpoint Structure

Each checkpoint is a self-contained directory with everything needed for inference:

```
checkpoint_003/
├── ensemble_stage1_models.joblib      # Stage 1 models wrapped in EnsembleClassifier
├── stage2_model.h5                    # Stage 2 DNN model (H5 format)
├── ensemble_classifier_transformers.py # Custom transformer definitions
└── metadata.json                       # Checkpoint information
```

### Checkpoint Naming

Checkpoints are numbered by ensemble size:
- `checkpoint_003` = 3 accepted models
- `checkpoint_006` = 6 accepted models  
- `checkpoint_009` = 9 accepted models
- etc.

The checkpoint number matches the number of Stage 1 models in the ensemble at the time of saving.

## Workflow

### 1. Train and Create Checkpoints

Run the training notebook ([notebooks/03.1-ensemble_training_refactored.ipynb](notebooks/03.1-ensemble_training_refactored.ipynb)):

- Checkpoints are **automatically saved** after each Stage 2 DNN retraining
- By default, retraining happens every 3 accepted models (configurable via `RETRAIN_FREQUENCY`)
- Checkpoints are saved to: `models/run_YYYYMMDD_HHMMSS/checkpoints/checkpoint_XXX/`

The checkpoint saving function logs output like:

```
✓ Saved checkpoint_003:
  Path: models/run_20251231_120000/checkpoints/checkpoint_003
  Stage 1 ensemble: 15.2 MB (3 models)
  Stage 2 DNN: 0.8 MB
  Transformers: 12.4 KB
  Total size: 16.0 MB
  Save time: 2.3s
```

### 2. Prepare Assets for Kaggle

Use the preparation script to stage checkpoint files:

```bash
# Activate virtual environment
source .venv/bin/activate

# Prepare latest checkpoint from most recent run
python scripts/prepare_kaggle_assets.py --checkpoint latest

# OR prepare specific checkpoint number
python scripts/prepare_kaggle_assets.py --checkpoint 006

# OR specify run directory and checkpoint
python scripts/prepare_kaggle_assets.py --checkpoint 009 --run-dir models/run_20251231_120000
```

This creates a staging directory:

```
kaggle_assets/checkpoint_XXX/
├── ensemble_stage1_models.joblib
├── stage2_model.h5
├── ensemble_classifier_transformers.py
├── metadata.json
└── README.md  # Generated instructions
```

The script outputs:

```
======================================================================
Assets staged successfully!
======================================================================

Files staged in: /home/user/diabetes-prediction/kaggle_assets/checkpoint_006

File list:
  ✓ ensemble_stage1_models.joblib                   32.45 MB
  ✓ stage2_model.h5                                  1.23 MB
  ✓ ensemble_classifier_transformers.py             12.4 KB
  ✓ metadata.json                                    0.8 KB
  ✓ README.md                                        2.1 KB

Total size: 33.68 MB

======================================================================
Next Steps:
======================================================================
1. Go to https://www.kaggle.com/datasets
2. Click 'New Dataset'
3. Upload all files from: /path/to/kaggle_assets/checkpoint_006
4. Name the dataset: diabetes-ensemble-checkpoint-006
5. In your Kaggle notebook, add this dataset as a data source
6. Set CHECKPOINT = '006' in the inference notebook
7. Run inference to generate submission.csv
======================================================================
```

### 3. Upload to Kaggle

1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload all 4 files from the staging directory
4. Name the dataset following the pattern: **`diabetes-ensemble-checkpoint-XXX`**
   - Example: `diabetes-ensemble-checkpoint-006`
   - The name must match the checkpoint number for the inference code to work
5. Set visibility (public/private)
6. Add description from the generated `README.md`
7. Click **"Create"**

### 4. Run Inference on Kaggle

#### Option A: Use Provided Inference Notebook

1. Upload [notebooks/03.3-ensemble_inference.ipynb](notebooks/03.3-ensemble_inference.ipynb) to Kaggle
2. Add your checkpoint dataset as a data source
3. Set configuration at the top:
   ```python
   KAGGLE = True
   CHECKPOINT = '006'  # Your checkpoint number
   ```
4. Run all cells
5. Download `submission.csv`

#### Option B: Manual Inference Code

Create a new Kaggle notebook with:

```python
import sys
import joblib
import pandas as pd
from pathlib import Path

# Configuration
KAGGLE = True
CHECKPOINT = '006'  # Your checkpoint number

# Set paths
checkpoint_dataset = f'diabetes-ensemble-checkpoint-{CHECKPOINT}'
module_path = Path(f'/kaggle/input/{checkpoint_dataset}')
sys.path.insert(0, str(module_path))

# Import ensemble classifier
from ensemble_classifier import EnsembleClassifier

# Load test data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Load model and metadata
import json
with open(module_path / 'metadata.json') as f:
    metadata = json.load(f)

print(f"Checkpoint: {metadata['checkpoint_num']:03d}")
print(f"Ensemble size: {metadata['ensemble_size']} models")
print(f"Stage 1 AUC: {metadata['stage1_val_auc']:.4f}")
print(f"Stage 2 AUC: {metadata['stage2_val_auc']:.4f}")

# Load ensemble model
model_path = module_path / 'ensemble_stage1_models.joblib'
stage2_model_path = module_path / 'stage2_model.h5'

model = joblib.load(model_path)
model.stage2_model_path = str(stage2_model_path)

# Make predictions (probabilities, not classes)
print(f"\nRunning inference on {len(test_df):,} samples...")
predictions_proba = model.predict_proba(test_df)
predictions = predictions_proba[:, 1]  # Probability of positive class

print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

# Create submission
submission_df = pd.DataFrame({
    'id': test_df['id'].astype(int),
    'diagnosed_diabetes': predictions  # Float probabilities
})

submission_df.to_csv('submission.csv', index=False)
print(f"\n✓ Submission saved: submission.csv ({len(submission_df)} rows)")
```

### 5. Local Testing (Optional)

Before uploading to Kaggle, test the checkpoint locally:

1. Open [notebooks/03.3-ensemble_inference.ipynb](notebooks/03.3-ensemble_inference.ipynb)
2. Set configuration:
   ```python
   KAGGLE = False
   CHECKPOINT = '006'  # Or 'latest'
   ```
3. Run all cells
4. Verify output:
   - Checkpoint metadata displays correctly
   - Predictions are probabilities in [0, 1] range
   - Submission file created in `data/ensemble_submission.csv`

## Checkpoint Selection Strategy

### By Validation Performance

Check the dashboard or database for validation AUC:

```sql
SELECT checkpoint_num, stage1_val_auc, stage2_val_auc, ensemble_size
FROM metadata_summary
ORDER BY stage2_val_auc DESC;
```

Or review training logs:

```bash
grep "Saved checkpoint" logs/ensemble_training.log
```

### By Ensemble Size

Larger ensembles may have better performance but:
- Longer inference time on Kaggle
- Larger dataset upload size

Consider submitting multiple checkpoints:
- Small checkpoint (e.g., `003`) for fast baseline
- Medium checkpoint (e.g., `009`) for balanced performance
- Large checkpoint (e.g., `015`) for maximum performance

## Multiple Submissions

You can create multiple Kaggle datasets from different checkpoints:

```bash
# Prepare multiple checkpoints
python scripts/prepare_kaggle_assets.py --checkpoint 003 --output-dir kaggle_assets/checkpoint_003
python scripts/prepare_kaggle_assets.py --checkpoint 006 --output-dir kaggle_assets/checkpoint_006
python scripts/prepare_kaggle_assets.py --checkpoint 009 --output-dir kaggle_assets/checkpoint_009
```

Then upload each as a separate dataset:
- `diabetes-ensemble-checkpoint-003`
- `diabetes-ensemble-checkpoint-006`
- `diabetes-ensemble-checkpoint-009`

Run inference with each and compare Kaggle leaderboard scores.

## Troubleshooting

### Checkpoint Not Found

**Error**: `FileNotFoundError: No checkpoints found`

**Solution**: Run training notebook until at least one Stage 2 retraining completes (requires `RETRAIN_FREQUENCY` accepted models).

### Module Import Error on Kaggle

**Error**: `ModuleNotFoundError: No module named 'ensemble_classifier'`

**Solution**: 
1. Verify all 4 files were uploaded to Kaggle dataset
2. Check dataset name matches: `diabetes-ensemble-checkpoint-XXX`
3. Verify `CHECKPOINT` variable matches dataset suffix

### Stage 2 Model Load Error

**Error**: `Failed to load Stage 2 model`

**Solution**:
1. Verify `stage2_model.h5` was uploaded
2. Check TensorFlow is available in Kaggle kernel
3. Try setting path manually: `model.stage2_model_path = '/kaggle/input/.../stage2_model.h5'`

### Wrong Prediction Format

**Error**: Kaggle expects probabilities but getting class labels

**Solution**: 
- Use `model.predict_proba(test_df)[:, 1]` not `model.predict(test_df)`
- Submission should have float values in [0, 1], not integers {0, 1}

### Memory Error on Kaggle

**Error**: Kernel runs out of memory during inference

**Solution**:
1. Use a smaller checkpoint (fewer models)
2. Process test data in batches if very large
3. Enable Kaggle GPU/TPU for Stage 2 inference

## File Size Considerations

Typical checkpoint sizes:
- 3 models: ~15-20 MB
- 6 models: ~30-40 MB  
- 9 models: ~45-60 MB
- 12 models: ~60-80 MB

Kaggle dataset size limit: **20 GB** (generous for this use case)

Kaggle notebook output limit: **500 MB** (should be fine for submission CSV)

## Advanced: Checkpoint Metadata

Each `metadata.json` contains:

```json
{
  "checkpoint_num": 6,
  "timestamp": "2025-12-31T12:34:56",
  "ensemble_size": 6,
  "stage1_val_auc": 0.8523,
  "stage2_val_auc": 0.8671,
  "retraining_count": 2,
  "pseudo_labeling": {
    "enabled": false
  },
  "config_snapshot": {
    "retrain_frequency": 3,
    "batch_size": 3,
    "max_iterations": 25
  }
}
```

Use this to track:
- When checkpoint was created
- Validation performance
- Whether pseudo-labeling was used
- Training configuration

## Summary

1. **Train**: Run training notebook → checkpoints auto-saved
2. **Prepare**: `python scripts/prepare_kaggle_assets.py --checkpoint XXX`
3. **Upload**: Kaggle Datasets → upload 4 files
4. **Infer**: Kaggle Notebook → run inference code
5. **Submit**: Download submission.csv → submit to competition

The checkpoint system allows continuous improvement while submitting intermediate results to the competition.
