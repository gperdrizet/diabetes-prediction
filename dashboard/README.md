# Ensemble Training Dashboard

Real-time monitoring dashboard for the ensemble hill climbing training process using Streamlit.

## Features

- **Auto-refresh**: Updates every 60 seconds to display live training progress
- **Performance Metrics**: Track Stage 1 and Stage 2 validation AUC over iterations
- **Diversity Analysis**: Monitor diversity scores and their relationship with performance
- **Composition Insights**: Visualize transformer usage, classifier distributions, and PCA statistics
- **Stage 2 DNN Training**: View detailed training curves for each DNN batch with transfer learning events
- **Batch Boundaries**: See when DNN retraining occurs (every 10 accepted models)
- **CSV Export**: Download complete training logs for offline analysis
- **Database Reset**: Safely clear training data with multi-step confirmation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: SQLite3 is part of Python's standard library and doesn't require separate installation.

## Usage

1. Start the dashboard:
```bash
streamlit run app.py
```

2. The dashboard will automatically open in your default browser at `http://localhost:8501`

3. The dashboard auto-refreshes every 60 seconds to display the latest training data

## Database Configuration

- **Path**: `/workspaces/diabetes-prediction/data/ensemble_training.db` (hardcoded)
- **Mode**: WAL (Write-Ahead Logging) for concurrent read/write access
- **Tables**: 
  - `ensemble_log`: Hill climbing iteration data
  - `stage2_log`: Stage 2 DNN training epoch data

## Dashboard Sections

### Header Metrics
- Total iterations (accepted + rejected)
- Current ensemble size (accepted models only)
- Best Stage 2 validation AUC
- Current temperature
- Current aggregation method (Simple Mean / DNN Weighted)
- Time since last update

### Performance Tab
- Stage 2 validation AUC timeline with acceptance/rejection markers
- Stage 1 validation AUC timeline (individual model performance)
- Batch boundaries showing DNN retraining events
- Ensemble size growth over iterations

### Diversity Tab
- Diversity score over iterations
- Diversity vs Stage 1 AUC scatter plot
- Classifier type distribution in accepted models

### Composition Tab
- Transformer usage frequency across accepted models
- Classifier type distribution (pie chart)
- PCA usage statistics

### Stage 2 DNN Tab
- Training curves (loss and AUC) for each DNN batch
- Dropdown selector for different training batches
- Transfer learning information
- DNN architecture growth visualization

## Data Export

Use the sidebar to export training data:
- Click "Download Ensemble Log CSV" to export `ensemble_log` table
- Click "Download Stage 2 Log CSV" to export `stage2_log` table

Files are timestamped automatically.

## Database Reset

⚠️ **WARNING: DESTRUCTIVE OPERATION** ⚠️

To reset the database:
1. Type `DELETE DATABASE` in the sidebar text input (exact match required)
2. Click the "Confirm Reset Database" button
3. All training data will be permanently deleted
4. Restart the training notebook to reinitialize the database

**This operation is irreversible!** Make sure to export data before resetting.

## Troubleshooting

### Database not found
- **Error**: "Database not found at: /workspaces/diabetes-prediction/data/ensemble_training.db"
- **Solution**: Start the training notebook (`02.1-ensemble_hill_climbing.ipynb`) to create the database

### No training data
- **Error**: "No training data found in database"
- **Solution**: Wait for the first hill climbing iteration to complete. The dashboard will auto-refresh.

### Dashboard not updating
- **Issue**: Data appears stale
- **Solution**: 
  - Check that training is still running
  - Verify database file is being updated (check file modification time)
  - Manually refresh the browser page

### Auto-refresh not working
- **Issue**: Dashboard requires manual refresh
- **Solution**: 
  - Check browser console for errors
  - Ensure `streamlit-autorefresh` is installed
  - Restart the dashboard

## Architecture Notes

### Batch-based Training
The dashboard visualizes the batch-based DNN training strategy:
- **Models 1-9**: Simple mean aggregation (fast)
- **Model 10**: First DNN training (10 inputs)
- **Models 11-19**: DNN weighted aggregation
- **Model 20**: DNN retraining with transfer learning (20 inputs)
- **Pattern continues**: Every 10 accepted models

### Transfer Learning
The dashboard shows how the DNN architecture grows:
- Previous DNN weights are copied to the new model
- New inputs (for new ensemble members) are randomly initialized
- This allows faster convergence than training from scratch

### Data Split
The training uses a 3-way fixed data split:
- **Training pool (60%)**: Random samples for Stage 1 model training
- **Stage 1 validation (20%)**: FIXED - for Stage 1 evaluation and Stage 2 training
- **Stage 2 validation (20%)**: HELD OUT - for final ensemble evaluation

The dashboard displays:
- **Stage 1 validation AUC**: Individual model performance on Stage 1 val set
- **Stage 2 validation AUC**: Ensemble performance on HELD OUT Stage 2 val set

## Performance

- **Cache TTL**: 60 seconds (matches auto-refresh interval)
- **Concurrent Access**: WAL mode allows dashboard to read while training writes
- **Query Optimization**: Indexes on `iteration_num`, `ensemble_id`, `epoch` for fast queries

## Development

To modify the dashboard:
1. Edit `app.py`
2. Save changes
3. Streamlit will auto-reload in development mode

For production deployment, consider:
- Adjusting auto-refresh interval
- Adding authentication
- Implementing query pagination for large datasets
- Adding VACUUM/ANALYZE for database maintenance
