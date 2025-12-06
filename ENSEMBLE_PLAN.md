# Online Hill Climbing Ensemble Implementation Plan

## Overview

Build a two-stage ensemble using online hill climbing with adaptive simulated annealing to iteratively add diverse stage 1 models. Train 5 separate founder models (one per fold) to establish baseline, use exponential stage 2 re-optimization schedule with manual stop/restart capability, terminate when performance plateaus for 100 iterations, and implement comprehensive performance logging for future web dashboard integration.

## Architecture

### Stage 1: Diverse Base Models
- **Founder ensemble**: 5 models (one per fold) trained on full dataset
- **Hill climbing**: Iteratively add models with diverse feature engineering and sampling
- **Row sampling**: 30-50% early iterations (0-100), 10-30% later iterations (diversity)
- **Column sampling**: 50-95% random feature selection
- **Feature engineering**: 1-3 custom transformers per pipeline (ratios, products, differences, etc.)
- **Classifiers**: LogisticRegression, RandomForest, GradientBoosting, SVC, MLP, KNN, ExtraTrees, AdaBoost

### Stage 2: DNN Meta-Learner
- **Input**: Predictions from all stage 1 models `[n_samples, n_models]`
- **Architecture**: 1-3 dense layers (32-256 units), dropout (0.2-0.5), batch normalization
- **Optimization**: Keras Tuner RandomSearch every time ensemble size doubles (5→10→20→40)
- **Training**: Early stopping on validation AUC with GPU acceleration

## Implementation Components

### 1. Custom Feature Engineering Transformers
**File**: `notebooks/functions/ensemble_transformers.py`

Transformers:
- `RatioTransformer`: Pairwise ratios (division-by-zero protection)
- `ProductTransformer`: Pairwise products
- `DifferenceTransformer`: Pairwise differences
- `SumTransformer`: Pairwise sums
- `ReciprocalTransformer`: 1/(x+epsilon)
- `SquareTransformer`: x²
- `SquareRootTransformer`: √|x| × sign(x)
- `LogTransformer`: log(|x|+1) × sign(x)
- `BinningTransformer`: Equal-width or quantile (3-10 bins)
- `KDESmoothingTransformer`: Gaussian KDE smoothing
- `RandomFeatureSelector`: Random column sampling (50-95%)

### 2. Hill Climbing & Pipeline Generation
**File**: `notebooks/functions/ensemble_hill_climbing.py`

Functions:
- `generate_random_pipeline()`: Create diverse stage 1 pipelines
- `calculate_ensemble_diversity()`: Mean pairwise prediction correlation
- `quick_optimize_pipeline()`: RandomizedSearchCV (10 iterations, 3-fold CV)
- `adaptive_simulated_annealing_acceptance()`: $P = e^{\Delta/T}$ with adaptive temperature
- `log_iteration()`: Record all iteration details to JSONL

### 3. Stage 2 DNN Model
**File**: `notebooks/functions/ensemble_stage2_model.py`

Functions:
- `build_stage2_dnn()`: Construct TensorFlow sequential model
- `optimize_stage2_hyperparameters()`: Keras Tuner RandomSearch
- `train_stage2_dnn()`: Fit with early stopping and GPU
- `evaluate_ensemble()`: Compute validation ROC-AUC
- `save_checkpoint()`: Serialize ensemble state
- `load_checkpoint()`: Restore and resume
- `log_stage2_performance()`: Record DNN training metrics

### 4. Training Notebook
**File**: `notebooks/02.1-ensemble_hill_climbing.ipynb`

Workflow:
1. Load training data, create 5-fold stratified split
2. **Founder ensemble (iterations 0-4)**:
   - Train 5 models (one per fold) on full dataset
   - Optimize with 20 RandomizedSearch iterations
   - Generate predictions on validation folds
   - Train simple stage 2 DNN per fold
   - Establish baseline: `best_score = mean(founder_val_aucs)`
3. **Hill climbing loop (iterations 5-500)**:
   - Check for checkpoint, prompt resume/restart
   - Rotate validation fold: `current_fold = iteration % 5`
   - Generate random pipeline with adaptive row sampling
   - Quick optimize on 4 training folds
   - Generate predictions on validation fold
   - Retrain stage 2 DNN, evaluate ROC-AUC
   - Apply adaptive simulated annealing acceptance
   - Log all iteration details (accepted and rejected)
   - Save accepted models to `models/ensemble_stage1_models/`
   - **Every time ensemble size doubles**: Save checkpoint, prompt user for optimization/continue/stop
   - **Terminate**: When best ROC-AUC hasn't improved for 100 iterations

### 5. Final Training & Inference
**Files**: `notebooks/02.2-ensemble_final_training.ipynb`, `02.3-ensemble_inference.ipynb`

Final training:
- Load ensemble pool, generate predictions on all folds
- Final stage 2 hyperparameter optimization (30 trials)
- Train final DNN (max 100 epochs, early stopping patience=15)
- Evaluate per-fold and overall ROC-AUC (mean ± std)
- Save model, config, and training history

Inference:
- Load all stage 1 models
- Parallel prediction on 300k test samples (24 workers)
- Stage 2 DNN prediction in batches (GPU)
- Measure inference time/memory
- Save submission CSV

### 6. Documentation
**File**: `models/ensemble_description.md` (auto-generated)

Includes:
- Founder ensemble details
- Hill climbing trajectory plots
- Diversity evolution
- Ensemble composition analysis
- Stage 2 optimization timeline
- Validation performance
- Inference metrics
- **Logging schema for dashboard**

## Logging Schema for Web Dashboard

### Training Log
**File**: `data/ensemble_training_log.jsonl`

Fields per iteration:
```json
{
  "iteration": 42,
  "timestamp": "2025-12-06T10:30:45.123456",
  "fold": 2,
  "accepted": true,
  "rejection_reason": null,
  "stage1_pipeline_hash": "abc123...",
  "stage1_cv_score": 0.6234,
  "stage1_val_auc": 0.6189,
  "stage2_val_auc": 0.6512,
  "ensemble_size": 25,
  "diversity_score": 0.3456,
  "temperature": 0.0082,
  "row_sample_pct": 0.23,
  "col_sample_pct": 0.87,
  "classifier_type": "RandomForestClassifier",
  "transformers_used": ["RatioTransformer", "LogTransformer"]
}
```

### Stage 2 Training Log
**File**: `data/stage2_training_log.jsonl`

Fields per epoch:
```json
{
  "iteration": 42,
  "fold": 2,
  "epoch": 15,
  "train_loss": 0.4523,
  "val_loss": 0.4612,
  "train_auc": 0.6834,
  "val_auc": 0.6512,
  "learning_rate": 0.001,
  "timestamp": "2025-12-06T10:31:02.456789"
}
```

## Key Design Decisions

### 1. Founder Member Strategy: C
Train 5 separate founder models (one per fold) to establish per-fold baselines from the start.

### 2. Stage 2 Re-optimization: B (Modified)
Exponential schedule when ensemble size doubles (5→10→20→40), with manual intervention points allowing users to:
- Continue climbing
- Run hyperparameter optimization
- Stop and save checkpoint

### 3. Termination Criteria: C
Stop when best ROC-AUC hasn't improved for 100 iterations (plateau detection).

### 4. Row Sampling Strategy: B
Bias toward larger samples early (30-50% for iterations 0-100), then smaller samples for diversity (10-30%).

### 5. Validation Strategy
K-fold rotation with stratified 5-fold split, cycling through folds as iterations progress.

### 6. Simulated Annealing: C (Adaptive)
Temperature adapts based on acceptance rate:
- If acceptance < 10% over last 20 iterations: increase temperature by 20%
- Otherwise: exponential decay $T_i = T_0 \times 0.995^i$ starting from $T_0 = 0.01$

## Hardware Utilization

- **CPU**: 24 cores for parallel stage 1 training and inference
- **Memory**: 256 GB for large ensemble and data handling
- **GPU**: NVIDIA P100 (16GB VRAM) for stage 2 DNN training and inference

## Expected Outcomes

1. **Ensemble size**: 20-100 diverse stage 1 models
2. **Diversity**: Low prediction correlation between models
3. **Performance**: Improved ROC-AUC over single model baseline (0.6440)
4. **Inference**: Fast parallel prediction on 300k test samples
5. **Monitoring**: Real-time dashboard tracking via JSONL logs
6. **Reproducibility**: Full checkpoint/resume capability

## Next Steps

1. ✅ Document plan
2. ⏳ Implement custom feature engineering transformers
3. ⏳ Implement hill climbing functions
4. ⏳ Implement stage 2 DNN model functions
5. ⏳ Create training notebook with founder ensemble
6. ⏳ Create final training and inference notebooks
7. ⏳ Test end-to-end pipeline
8. ⏳ Deploy and monitor
