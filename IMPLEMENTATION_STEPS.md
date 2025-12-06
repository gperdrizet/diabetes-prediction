# Hill Climbing Ensemble Implementation Steps

## Overview

This document outlines the complete implementation sequence for integrating SQLite logging into the hill climbing ensemble training system and building a Streamlit monitoring dashboard.

**Implementation Date**: December 6, 2025  
**Status**: In Progress

## Prerequisites

- Core ensemble modules already implemented:
  - `notebooks/functions/ensemble_transformers.py` (11 custom transformers)
  - `notebooks/functions/ensemble_hill_climbing.py` (pipeline generation, simulated annealing)
  - `notebooks/functions/ensemble_stage2_model.py` (DNN meta-learner functions)
  - `notebooks/02.1-ensemble_hill_climbing.ipynb` (training notebook)
- Hardware: 24 CPU cores, 256GB RAM, NVIDIA P100 GPU
- Dependencies: scikit-learn 1.2.2, TensorFlow/Keras, SQLite3

## Implementation Sequence

### Phase 1: SQLite Database Integration (Steps 1-5)

These steps prepare the training system to log to SQLite instead of JSONL files.

#### Step 1: Document Implementation Plan ✓
**File**: `IMPLEMENTATION_STEPS.md` (this file)  
**Status**: Complete

Create comprehensive documentation of the complete implementation sequence including SQLite integration and dashboard development.

#### Step 2: Implement Database Manager Module
**File**: `notebooks/functions/ensemble_database.py`  
**Estimated Time**: ~10 minutes

Create SQLite database manager with the following functions:

```python
# Core functions to implement:
- init_database() -> None
  - Create ensemble_log and stage2_log tables
  - Enable WAL mode for concurrent access
  - Create indexes on iteration_num, ensemble_id, epoch
  - Hardcode DB_PATH = '/workspaces/diabetes-prediction/data/ensemble_training.db'

- insert_ensemble_iteration(iteration_data: dict) -> None
  - Insert hill climbing iteration data
  - Use context manager for thread safety

- insert_stage2_epoch(epoch_data: dict) -> None
  - Insert stage 2 DNN training epoch data
  - Use context manager for thread safety

- query_ensemble_data(limit: int = None) -> pd.DataFrame
  - Retrieve ensemble iteration data
  - Optional limit for most recent N rows

- query_stage2_data(ensemble_id: str, limit: int = None) -> pd.DataFrame
  - Retrieve stage 2 training data for specific ensemble
  - Optional limit for most recent N epochs

- get_summary_stats() -> dict
  - Return aggregate statistics (total iterations, best score, etc.)

- reset_database() -> None
  - Drop and recreate tables (DESTRUCTIVE)
  - For manual database cleanup only
```

**Database Schema**:

```sql
-- ensemble_log table
CREATE TABLE ensemble_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    iteration_num INTEGER NOT NULL,
    ensemble_id TEXT NOT NULL,
    cv_score REAL NOT NULL,
    diversity_score REAL NOT NULL,
    combined_score REAL NOT NULL,
    temperature REAL NOT NULL,
    accepted INTEGER NOT NULL,
    acceptance_reason TEXT,
    num_models INTEGER NOT NULL,
    transformers_used TEXT,
    pipeline_hash TEXT NOT NULL
);

CREATE INDEX idx_iteration_num ON ensemble_log(iteration_num);
CREATE INDEX idx_ensemble_id ON ensemble_log(ensemble_id);

-- stage2_log table
CREATE TABLE stage2_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ensemble_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    train_loss REAL NOT NULL,
    val_loss REAL NOT NULL,
    train_auc REAL NOT NULL,
    val_auc REAL NOT NULL
);

CREATE INDEX idx_stage2_ensemble ON stage2_log(ensemble_id);
CREATE INDEX idx_stage2_epoch ON stage2_log(epoch);
```

#### Step 3: Update Hill Climbing Logging
**File**: `notebooks/functions/ensemble_hill_climbing.py`  
**Estimated Time**: ~5 minutes

Modify the `log_iteration()` function:

**Changes**:
- Remove `log_path` parameter
- Import `ensemble_database` module
- Serialize `transformers_used` list to comma-separated string
- Call `ensemble_database.insert_ensemble_iteration()` instead of JSONL append
- Wrap in try/except for graceful error handling

**Before**:
```python
def log_iteration(log_path, iteration_data):
    with open(log_path, 'a') as f:
        f.write(json.dumps(iteration_data) + '\n')
```

**After**:
```python
import ensemble_database

def log_iteration(iteration_data):
    try:
        # Serialize transformers list
        if 'transformers_used' in iteration_data:
            iteration_data['transformers_used'] = ','.join(iteration_data['transformers_used'])
        ensemble_database.insert_ensemble_iteration(iteration_data)
    except Exception as e:
        print(f"Warning: Failed to log iteration {iteration_data.get('iteration_num')}: {e}")
```

#### Step 4: Update Stage 2 DNN Logging
**File**: `notebooks/functions/ensemble_stage2_model.py`  
**Estimated Time**: ~5 minutes

Modify the `train_stage2_dnn()` function to add per-epoch logging:

**Changes**:
- Import `ensemble_database` module
- After each epoch, call `ensemble_database.insert_stage2_epoch()` with epoch metrics
- Include: timestamp, ensemble_id, epoch, train_loss, val_loss, train_auc, val_auc
- Wrap in try/except for graceful error handling

**Example epoch logging**:
```python
try:
    epoch_data = {
        'timestamp': datetime.now().isoformat(),
        'ensemble_id': ensemble_id,
        'epoch': epoch,
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'train_auc': history.history['auc'][-1],
        'val_auc': history.history['val_auc'][-1]
    }
    ensemble_database.insert_stage2_epoch(epoch_data)
except Exception as e:
    print(f"Warning: Failed to log epoch {epoch}: {e}")
```

#### Step 5: Update Training Notebook
**File**: `notebooks/02.1-ensemble_hill_climbing.ipynb`  
**Estimated Time**: ~5 minutes

**Changes**:
1. Add database import in imports cell:
   ```python
   from functions import ensemble_database
   ```

2. Add database initialization cell (before training loop):
   ```python
   # Initialize SQLite database
   ensemble_database.init_database()
   print("Database initialized at:", ensemble_database.DB_PATH)
   ```

3. Remove `ENSEMBLE_LOG_PATH` and `STAGE2_LOG_PATH` from configuration cell

4. Remove `log_path` parameter from all `log_iteration()` calls in hill climbing loop

### Phase 2: Start Training Run (Step 6)

#### Step 6: Execute First Hill Climbing Run
**Action**: User manually runs training notebook  
**Expected Duration**: Hours to days (long-running process)

**What happens**:
- Founder ensemble is created (5 diverse models)
- Hill climbing loop begins with simulated annealing
- Each iteration logs to SQLite database in real-time
- Stage 2 DNN training epochs logged for each accepted ensemble
- Training continues until stopped manually or target iterations reached

**Monitoring during training**:
- Database file created at: `/workspaces/diabetes-prediction/data/ensemble_training.db`
- Can query database directly: `sqlite3 data/ensemble_training.db "SELECT * FROM ensemble_log ORDER BY iteration_num DESC LIMIT 10;"`
- Dashboard development can proceed in parallel

### Phase 3: Dashboard Implementation (Steps 7-10)

These steps create the Streamlit monitoring dashboard while training runs in background.

#### Step 7: Implement Streamlit Dashboard
**File**: `dashboard/app.py`  
**Estimated Time**: ~30 minutes

Create Streamlit application with the following components:

**Core Features**:
1. **Auto-refresh**: Use `streamlit-autorefresh` for live updates
2. **Hardcoded database path**: `DB_PATH = '/workspaces/diabetes-prediction/data/ensemble_training.db'`
3. **Cached queries**: `@st.cache_data(ttl=60)` for all database queries
4. **Header metrics**: Total iterations, best CV score, current temperature, time since last update
5. **Tabbed interface**:
   - **Performance Tab**: CV score over iterations, combined score over iterations
   - **Diversity Tab**: Diversity score over iterations, diversity vs CV score scatter
   - **Composition Tab**: Transformer usage frequency, models per iteration
   - **Stage 2 Tab**: DNN training curves (loss, AUC) for selected ensemble
6. **CSV export**: Download current ensemble_log and stage2_log tables
7. **Database reset**: Multi-step confirmation (text input "DELETE DATABASE" + button click)

**Error Handling**:
- Check if database file exists
- Handle empty tables gracefully
- Display informative messages for missing data
- Try/except around all database operations

**Example structure**:
```python
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sqlite3
import pandas as pd
from pathlib import Path

# Hardcoded database path
DB_PATH = '/workspaces/diabetes-prediction/data/ensemble_training.db'

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

# Cached query functions
@st.cache_data(ttl=60)
def get_ensemble_data():
    # Query ensemble_log table
    pass

# Header with metrics
col1, col2, col3, col4 = st.columns(4)
# ... display metrics

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Diversity", "Composition", "Stage 2"])

# CSV export in sidebar
# Database reset in sidebar
```

#### Step 8: Create Dashboard Requirements
**File**: `dashboard/requirements.txt`  
**Estimated Time**: ~2 minutes

Minimal dependencies matching Kaggle environment:

```
streamlit>=1.28.0
streamlit-autorefresh>=0.0.1
pandas>=1.5.0
numpy>=1.23.0
```

**Note**: SQLite3 is part of Python standard library (no separate install needed)

#### Step 9: Create Dashboard Documentation
**File**: `dashboard/README.md`  
**Estimated Time**: ~5 minutes

**Contents**:
- Dashboard purpose and features
- Installation instructions (`pip install -r requirements.txt`)
- Usage instructions (`streamlit run app.py`)
- Database path information (hardcoded)
- Database reset warnings (DESTRUCTIVE, IRREVOCABLE)
- Troubleshooting common issues

#### Step 10: Test Dashboard with Live Data
**Action**: Manual testing and validation  
**Estimated Time**: ~10 minutes

**Testing steps**:
1. Install dashboard dependencies: `pip install -r dashboard/requirements.txt`
2. Start dashboard: `streamlit run dashboard/app.py`
3. Verify auto-refresh works (60s interval)
4. Check all tabs display correct data
5. Test CSV export functionality
6. Verify database reset safety (multi-step confirmation)
7. Monitor dashboard while training runs

## Database Configuration

**Path**: `/workspaces/diabetes-prediction/data/ensemble_training.db` (hardcoded)  
**Mode**: WAL (Write-Ahead Logging) for concurrent read/write access  
**Timeout**: 30 seconds for lock acquisition  
**Cache TTL**: 60 seconds (matches long iteration times)

## Safety Features

1. **Database Reset Protection**:
   - Two-step confirmation required
   - User must type "DELETE DATABASE" in text input
   - Then click "Confirm Reset" button
   - Warning messages emphasize irreversible nature

2. **Graceful Error Handling**:
   - All database operations wrapped in try/except
   - Informative error messages displayed to user
   - Training continues even if logging fails

3. **Concurrent Access**:
   - WAL mode allows dashboard to read while training writes
   - Context managers ensure proper connection cleanup
   - Indexes optimize query performance

## Success Criteria

- [ ] Database module creates tables with correct schema
- [ ] Hill climbing iterations log to SQLite in real-time
- [ ] Stage 2 DNN epochs log to SQLite during training
- [ ] Training notebook runs without errors
- [ ] Dashboard displays live data from training run
- [ ] Auto-refresh updates dashboard every 60 seconds
- [ ] CSV export downloads complete data
- [ ] Database reset requires multi-step confirmation
- [ ] No concurrent access issues (WAL mode working)

## Rollback Plan

If SQLite integration causes issues:
1. Keep original JSONL logging code in git history
2. Can revert ensemble_hill_climbing.py and ensemble_stage2_model.py
3. Database file can be safely deleted (all data is also in checkpoints)
4. Training can resume from last checkpoint

## Next Steps After Implementation

1. Monitor first training run for several iterations
2. Verify database growth is reasonable (not too large)
3. Check dashboard performance with large datasets (1000+ iterations)
4. Consider adding database maintenance (VACUUM, ANALYZE) for long runs
5. Optionally add more visualizations based on training insights
