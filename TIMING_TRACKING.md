# Timing Tracking Documentation

## Overview

Timing tracking has been added to the ensemble hill climbing training process to provide insights into computational performance and identify bottlenecks.

## What's Tracked

### 1. Parallel Pipeline Training Time (`training_time_sec`)
- Time spent training candidate pipelines in parallel
- Measured for each iteration
- Captured by the `train_single_candidate()` worker function
- Includes: model fitting, validation, and result collection

### 2. Stage 2 DNN Training Time (`stage2_time_sec`)
- Time spent training or expanding the Stage 2 meta-learner DNN
- Only measured when Stage 2 training occurs (accepted candidates)
- Captured by the `train_or_expand_stage2_model()` function
- Includes: transfer learning, model compilation, and training

## Implementation Details

### Database Schema
```sql
ALTER TABLE ensemble_iterations ADD COLUMN training_time_sec REAL;
ALTER TABLE ensemble_iterations ADD COLUMN stage2_time_sec REAL;
```

### Code Changes

#### 1. Parallel Training (`ensemble_parallel.py`)
```python
def train_single_candidate(...):
    # Already had timing, now explicitly returns it
    return {
        'pipeline': pipeline,
        'val_auc': val_auc,
        'metadata': metadata,
        'training_time': training_time,  # Now used
        'memory_mb': memory_used,
        'training_time_sec': training_time  # Explicitly named
    }
```

#### 2. Stage 2 Training (`ensemble_stage2_training.py`)
```python
import time

def train_or_expand_stage2_model(...):
    start_time = time.time()
    
    # ... training logic ...
    
    elapsed_time = time.time() - start_time
    return stage2_model, final_score, memory_used, elapsed_time
```

#### 3. Logging (`ensemble_hill_climbing.py`)
```python
def log_iteration(..., training_time_sec=None, stage2_time_sec=None):
    iteration_data = {
        # ... existing fields ...
        'training_time_sec': training_time_sec,
        'stage2_time_sec': stage2_time_sec
    }
```

#### 4. Notebook Integration
The refactored notebook extracts timing from results:
```python
# Extract from parallel training results
training_time_sec = result.get('training_time_sec', None)

# Capture from Stage 2 training
stage2_model, final_score, stage2_memory_mb, stage2_time_sec = train_or_expand_stage2_model(...)

# Log both metrics
log_iteration(..., 
    training_time_sec=training_time_sec,
    stage2_time_sec=stage2_time_sec
)
```

## Dashboard Visualizations

A new **"⏱️ Timing"** tab has been added to the dashboard with:

### Header Metrics
1. **Avg Training Time** - Average time per parallel training iteration
2. **Total Training Time** - Cumulative time spent on all training (in minutes)
3. **Avg Stage 2 Time** - Average time per Stage 2 DNN training
4. **Total Stage 2 Time** - Cumulative time spent on Stage 2 training (in minutes)

### Charts

#### 1. Training Time Over Iterations
- Line chart showing parallel training time trends
- Helps identify if training is slowing down over time
- Useful for capacity planning

#### 2. Training Time by Classifier Type
- Bar chart with error bars showing min/max range
- Compares average training time across different classifier types
- Identifies which models are computationally expensive

#### 3. Stage 2 DNN Training Time
- Line chart showing Stage 2 training time evolution
- Shows trends as the ensemble grows
- Indicates if transfer learning overhead increases

#### 4. Time Efficiency
- Scatter plot: seconds per AUC improvement
- Color-coded by classifier type
- Lower values indicate better efficiency
- Formula: `training_time_sec / abs(auc_improvement)`

## Usage

### Running Training with Timing
No code changes needed - timing is automatically tracked:

```python
# Simply run the refactored notebook
%run 02.1-ensemble_hill_climbing_refactored.ipynb
```

### Viewing Timing Data

1. **In the Notebook**: Timing is printed during training
   ```
   Iteration 5: LogisticRegression | Stage 1 AUC: 0.843210 | Memory: 245.3 MB | Time: 12.4s
   ```

2. **In the Dashboard**: Navigate to the "⏱️ Timing" tab
   ```bash
   cd dashboard
   streamlit run app.py
   ```

3. **Query the Database**:
   ```python
   import sqlite3
   import pandas as pd
   
   conn = sqlite3.connect('ensemble_hill_climbing.db')
   df = pd.read_sql_query("""
       SELECT iteration, classifier_type, 
              training_time_sec, stage2_time_sec
       FROM ensemble_iterations
       ORDER BY iteration
   """, conn)
   ```

## Performance Insights

### Typical Timings (Example)
- **Parallel Training**: 5-30 seconds per iteration (depends on classifier type)
  - LogisticRegression: ~5-8s
  - RandomForest: ~10-15s  
  - XGBoost: ~15-25s
  - LightGBM: ~12-20s

- **Stage 2 DNN**: 3-15 seconds per training
  - Initial training: ~10-15s
  - Transfer learning: ~3-8s

### Optimization Opportunities
Use timing data to:
1. Identify slow classifier types that might be removed
2. Optimize parallel worker count based on iteration time
3. Determine if Stage 2 transfer learning is efficient
4. Calculate total expected training time for N iterations
5. Compare time vs. accuracy trade-offs

## Backward Compatibility

- Database columns are nullable - old data without timing still works
- Dashboard gracefully handles missing timing data
- Modules return timing but it's optional in logging
- Existing training runs won't break

## Related Documentation

- See `MEMORY_TRACKING.md` for memory usage tracking
- See module docstrings in `functions/ensemble_*.py` for implementation details
- See dashboard code in `dashboard/app.py` for visualization logic
