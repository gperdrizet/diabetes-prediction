# Diabetes Prediction - Hill Climbing Ensemble

Kaggle Playground Series - Season 5, Episode 12: Diabetes Prediction Challenge

## Overview

This project implements a sophisticated ensemble learning system that uses **parallel hill climbing with simulated annealing** to build and optimize a diverse ensemble of machine learning models for diabetes prediction. The system combines traditional ML algorithms with a deep neural network meta-learner (Stage 2 DNN) to achieve superior predictive performance.

## Key Features

### Parallel hill climbing architecture
- **Batch training**: Trains 10 candidate models in parallel using ProcessPoolExecutor
- **Simulated annealing**: Adaptive acceptance strategy with temperature-based exploration
- **Hybrid scoring**: Uses DNN meta-learner for trained ensembles, falls back to mean aggregation for new candidates
- **Diversity optimization**: Actively seeks models that disagree to maximize ensemble strength

### Two-stage ensemble design
- **Stage 1**: Diverse pool of scikit-learn models (Random Forest, XGBoost, LightGBM, etc.)
- **Stage 2**: Transfer learning DNN that learns optimal model weighting and combinations
- **Progressive training**: Stage 2 DNN expands incrementally as ensemble grows

### Real-time monitoring
- **Live dashboard**: Streamlit-based visualization with 6 analysis pages
- **Performance tracking**: AUC scores, diversity metrics, acceptance rates
- **Memory and timing**: Comprehensive resource usage monitoring
- **SQLite logging**: WAL-mode database for concurrent read/write access

### Performance optimizations
- **Memory tracking**: psutil-based monitoring of peak memory usage
- **Timing metrics**: Detailed performance profiling of training stages
- **Efficient storage**: Compressed model bundles for deployment
- **Modular design**: Clean separation of concerns across 7+ modules

## Project structure

```
diabetes-prediction/
├── data/
│   ├── submission.csv                    # Kaggle submission file
│   └── ensemble_training.db              # Training metrics database
├── dashboard/
│   ├── app.py                            # Streamlit monitoring dashboard
│   └── .streamlit/
│       └── config.toml                   # Dashboard configuration
├── notebooks/
│   ├── 01-submission_template.ipynb      # Basic template
│   ├── 02-logistic_regression.ipynb      # Initial baseline model
│   ├── 02.1-ensemble_hill_climbing_refactored.ipynb  # Main training notebook
│   └── functions/
│       ├── ensemble_database.py          # SQLite database manager
│       ├── ensemble_initialization.py    # Data splits & preprocessing
│       ├── ensemble_parallel.py          # Parallel training workers
│       ├── ensemble_evaluation.py        # Hybrid scoring logic
│       ├── ensemble_stage2_training.py   # DNN meta-learner training
│       ├── ensemble_hill_climbing.py     # Core hill climbing logic
│       └── ensemble_stage2_model.py      # DNN model architecture
├── models/
│   └── run_YYYYMMDD_HHMMSS/             # Timestamped training runs
│       ├── ensemble_stage1_models/       # Individual model files
│       ├── ensemble_bundle.joblib        # Complete ensemble package
│       ├── ensemble_checkpoint.pkl       # Training checkpoint
│       └── ensemble_metadata.json        # Run configuration & stats
├── MEMORY_TRACKING.md                    # Memory monitoring documentation
├── TIMING_TRACKING.md                    # Timing metrics documentation
├── TIMING_SUMMARY.md                     # Performance analysis summary
└── requirements.txt                      # Python dependencies
```

## Getting started

### Prerequisites
```bash
Python 3.12+
TensorFlow 2.x
scikit-learn
pandas, numpy
xgboost, lightgbm
streamlit
psutil
```

### Installation
```bash
# Clone repository
git clone https://github.com/gperdrizet/diabetes-prediction.git
cd diabetes-prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the training

1. **Start the monitoring dashboard** (optional):
```bash
streamlit run ./dashboard/app.py
```

2. **Run the training notebook**:
Open `notebooks/02.1-ensemble_hill_climbing_refactored.ipynb` and execute cells sequentially.

The training process will:
- Load and split data into 3 validation sets
- Train a founder model to initialize the ensemble
- Run parallel hill climbing for up to 500 iterations
- Train Stage 2 DNN every 10 accepted models
- Save checkpoints and final ensemble bundle

### Configuration parameters

Key parameters in the notebook:
```python
RANDOM_STATE = 315              # Reproducibility
BATCH_SIZE = 10                 # Parallel candidates per batch
N_WORKERS = 10                  # Parallel worker processes
MAX_ITERATIONS = 500            # Maximum hill climbing iterations
PLATEAU_ITERATIONS = 100        # Early stopping threshold
STAGE2_BATCH_SIZE_MODELS = 10   # DNN retraining frequency
STAGE2_EPOCHS = 100             # DNN training epochs
```

## Dashboard features

The Streamlit dashboard provides 6 analysis pages:

1. **Performance**: AUC trends, acceptance rates, ensemble growth
2. **Diversity**: Model agreement patterns, diversity evolution
3. **Composition**: Classifier type distribution, hyperparameter analysis
4. **Stage 2 DNN**: Meta-learner training history and architecture
5. **Memory Usage**: Peak memory tracking for training stages
6. **Timing**: Performance profiling and time allocation

Access at `http://localhost:8501` after starting with streamlit.

## Hill climbing algorithm

The parallel hill climbing process:

1. **Generate batch**: Create 10 random model configurations
2. **Parallel train**: Train all candidates simultaneously
3. **Evaluate**: Score each candidate ensemble with hybrid method
4. **Accept/reject**: Use simulated annealing acceptance criteria
5. **Update temperature**: Decay temperature to reduce exploration over time
6. **Train DNN**: Retrain meta-learner every 10 accepted models
7. **Repeat**: Continue until max iterations or plateau reached

### Simulated annealing acceptance

```python
if candidate_score > current_score:
    accept = True  # Always accept improvements
else:
    delta = candidate_score - current_score
    probability = exp(delta / temperature)
    accept = random() < probability  # Probabilistic acceptance
```

## Stage 2 DNN meta-learner

The DNN architecture:
- **Input**: Predictions from all Stage 1 models (ensemble_size × 1)
- **Hidden layers**: Dense layers with dropout for regularization
- **Transfer learning**: Expands when ensemble grows, reuses learned weights
- **Output**: Final probability prediction (binary classification)

Benefits:
- Learns optimal model weighting automatically
- Captures complex interaction patterns
- Outperforms simple averaging strategies
- Adapts as ensemble composition changes

## Performance monitoring

### Memory tracking
- Peak memory usage during parallel training
- Stage 2 DNN training memory consumption
- Logged to database and displayed in dashboard

### Timing metrics
- Training time per model
- Batch completion times
- Stage 2 DNN training duration
- Overall iteration timing

## Model deployment

The final ensemble bundle contains:
```python
ensemble_bundle = {
    'ensemble_models': [...],      # List of fitted pipelines
    'stage2_model': ...,           # Trained Keras model
    'metadata': {...},             # Training statistics
    'base_preprocessor': ...,      # Feature preprocessing
    'feature_info': {...}          # Feature definitions
}
```

Load and use:
```python
import joblib

# Load bundle
bundle = joblib.load('models/run_*/ensemble_bundle.joblib')
ensemble_models = bundle['ensemble_models']
stage2_model = bundle['stage2_model']

# Make predictions
stage1_preds = [model.predict_proba(X)[:, 1] for model in ensemble_models]
final_preds = stage2_model.predict(np.column_stack(stage1_preds))
```

## Key algorithms and techniques

- **Simulated annealing**: Exploration-exploitation balance
- **Diversity metrics**: Pairwise model disagreement
- **Transfer learning**: Incremental DNN expansion
- **Parallel processing**: Multi-core model training
- **Early stopping**: Plateau detection
- **Checkpointing**: Resume from failures
- **WAL database**: Concurrent monitoring during training

## Results

Training typically produces:
- **Ensemble size**: 30-50 models
- **Validation AUC**: 0.86-0.88
- **Training time**: 2-6 hours (depending on hardware)
- **Acceptance rate**: 10-20%
- **Memory usage**: 2-4 GB peak

## Documentation

- `MEMORY_TRACKING.md`: Detailed memory monitoring implementation
- `TIMING_TRACKING.md`: Timing metrics system documentation
- `TIMING_SUMMARY.md`: Performance analysis and optimization guide

## Contributing

This is a Kaggle competition project. Feel free to fork and experiment with different:
- Model types and hyperparameters
- Hill climbing strategies
- Meta-learner architectures
- Diversity metrics

## License

See LICENSE file.

## Acknowledgments

- Kaggle Playground Series for the dataset
- scikit-learn and TensorFlow communities
- Streamlit for visualization framework
