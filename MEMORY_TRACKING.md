# Memory Tracking Implementation

## Overview
Added comprehensive memory usage tracking to the ensemble hill climbing training process. Memory consumption is now monitored for both parallel pipeline training and Stage 2 DNN training.

## Changes Made

### 1. Database Schema Updates
**File:** `notebooks/functions/ensemble_database.py`
- Added `training_memory_mb` column to `ensemble_log` table for pipeline training memory
- Added `stage2_memory_mb` column to `ensemble_log` table for DNN training memory
- Updated `insert_ensemble_iteration()` to accept and store memory metrics

### 2. Parallel Training Memory Tracking
**File:** `notebooks/functions/ensemble_parallel.py`
- Added `psutil` library for process memory monitoring
- Modified `train_single_candidate()` to track memory before and after training
- Returns memory usage in MB as part of result dictionary
- Memory is tracked per individual pipeline training process

### 3. Stage 2 DNN Memory Tracking
**File:** `notebooks/functions/ensemble_stage2_training.py`
- Added memory tracking to `train_or_expand_stage2_model()`
- Monitors memory consumption during DNN construction and training
- Returns memory usage alongside model and score
- Tracks both initial training and transfer learning scenarios

### 4. Logging Updates
**File:** `notebooks/functions/ensemble_hill_climbing.py`
- Updated `log_iteration()` function signature to accept memory parameters
- Memory metrics are now logged to database for each iteration
- Optional parameters allow backward compatibility

### 5. Notebook Updates
**File:** `notebooks/02.1-ensemble_hill_climbing_refactored.ipynb`
- Extracts memory usage from training results
- Passes memory metrics to logging function
- Displays memory usage in console output during training
- Tracks memory separately for accepted models and DNN retraining events

### 6. Dashboard Enhancements
**File:** `dashboard/app.py`
- Added new "Memory Usage" tab (tab5)
- Displays memory metrics in header (avg, peak training, avg stage 2)
- **Memory visualizations:**
  - Pipeline training memory over iterations
  - Memory usage by classifier type (with error bars)
  - Stage 2 DNN training memory over batches
  - Memory efficiency metric (MB per 0.01% AUC improvement)

## Dashboard Features

### Header Metrics
- **Avg Training Memory:** Average memory consumed per pipeline training
- **Peak Training Memory:** Maximum memory used by any pipeline
- **Avg Stage 2 Memory:** Average memory for DNN training events

### Memory Usage Tab
1. **Pipeline Training Memory Graph**
   - Scatter plot showing memory usage per iteration
   - Color-coded by acceptance (green=accepted, red=rejected)
   - Average memory line for reference

2. **Memory by Classifier Type**
   - Bar chart showing average memory per classifier
   - Error bars indicating max memory
   - Sample counts displayed

3. **Stage 2 DNN Memory**
   - Bar chart of memory usage at each DNN retraining
   - Shows memory growth as ensemble expands
   - Summary statistics (min/avg/max)

4. **Memory Efficiency Analysis**
   - Tracks MB per 0.01% AUC improvement
   - Helps identify most efficient models
   - Lower values indicate better efficiency

## Dependencies
- **psutil:** Added for process memory monitoring
  - Install: `pip install psutil`
  - Used in both parallel training and stage 2 training modules

## Usage

### Starting Fresh
1. Reset database to get new schema: Training will automatically reset
2. Run updated training notebook
3. Launch dashboard to view memory metrics

### Backward Compatibility
- Memory columns allow NULL values
- Existing code will work without memory tracking
- Dashboard gracefully handles missing memory data

## Memory Tracking Details

### Training Memory
- Measured in MB (megabytes)
- Captured per process in parallel training
- Includes sklearn pipeline fit operations
- Accounts for feature engineering transformations

### Stage 2 Memory
- Measured during DNN construction and training
- Includes TensorFlow/Keras operations
- Tracks transfer learning overhead
- Grows as ensemble size increases (more input dimensions)

## Future Enhancements
- Add memory prediction based on classifier type and configuration
- Track GPU memory if CUDA becomes available
- Add memory alerts/warnings for resource management
- Correlate memory usage with training time
