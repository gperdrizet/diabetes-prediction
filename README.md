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
- **Adaptive sampling**: Classifier-specific row sample sizes (5-55% of data)
- **Adaptive n_jobs**: Intelligent CPU core allocation (1-5 cores per model)
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

### Model diversity

The ensemble generates diverse pipelines by randomly combining:

**13 Classifier types:**
1. **Logistic Regression**: Fast linear baseline with L2/no penalty
2. **Random Forest**: Ensemble of decision trees (10-100 estimators)
3. **Gradient Boosting**: Sequential boosting with HistGradientBoosting
4. **Linear SVC**: Fast O(n) linear SVM classifier
5. **SGD Classifier**: Stochastic gradient descent with multiple loss functions
6. **MLP**: Multi-layer perceptron neural network (1-3 hidden layers)
7. **k-Nearest Neighbors**: Instance-based learning (3-30 neighbors)
8. **Extra Trees**: Extremely randomized trees ensemble
9. **AdaBoost**: Adaptive boosting ensemble
10. **Naive Bayes**: Probabilistic classifier (Gaussian/Multinomial/Bernoulli variants)
11. **Gaussian Process**: Kernel-based probabilistic model (RBF/Matern/RationalQuadratic/DotProduct)
12. **Linear Discriminant Analysis (LDA)**: Gaussian with shared covariance
13. **Quadratic Discriminant Analysis (QDA)**: Gaussian with separate covariances

**17 Feature engineering transformers:**
1. **Ratio**: Feature pairwise ratios (5-30 features)
2. **Product**: Feature pairwise products
3. **Difference**: Feature pairwise differences
4. **Sum**: Feature pairwise sums
5. **Reciprocal**: 1/x transformations
6. **Square**: x² transformations
7. **Square Root**: √x transformations
8. **Log**: log(x+1) transformations
9. **Binning**: Quantile/uniform binning (3-10 bins)
10. **KDE Smoothing**: Kernel density estimation smoothing
11. **K-Means Clustering**: Cluster labels + optional distances (3-10 clusters)
12. **Nystroem**: Kernel approximation (RBF/poly/sigmoid/cosine, 30-300 components)
13. **RBF Sampler**: Random Fourier features for RBF kernel
14. **Skewed Chi²**: Approximates skewed chi-squared kernel
15. **Power Transform**: Yeo-Johnson normalization (handles negative values)
16. **Quantile Transform**: Transform to uniform or normal distribution (100/500/1000 quantiles)
17. **Standard Scaler**: Standardization with configurable centering/scaling

**5 Dimensionality reduction techniques** (randomly select one or none):
1. **PCA**: Principal Component Analysis (90%/95%/99% variance or MLE)
2. **Truncated SVD**: SVD without centering (5-50 components)
3. **Fast ICA**: Independent Component Analysis (5-50 components)
4. **NMF**: Non-negative Matrix Factorization (5-50 components)
5. **Factor Analysis**: Gaussian latent variable model (5-50 components)

Each pipeline randomly selects:
- 1 classifier with randomized hyperparameters
- 1-3 feature engineering transformers
- 0 or 1 dimensionality reduction technique (50% probability)
- Random column sampling (50-95% of features)
- Adaptive row sampling based on classifier complexity (2.5-27.5% of rows)

### Performance optimization strategies

**Adaptive row sampling by classifier complexity:**
- **Very slow** (GaussianProcess, kNN): 2.5-12.5% of data (early), 2.5-7.5% (late) → ~6-8x speedup
- **Moderately slow** (MLP, AdaBoost): 5-20% of data (early), 5-12.5% (late) → ~4-5x speedup
- **Moderate** (RandomForest, ExtraTrees, GradientBoosting): 7.5-22.5% (early), 7.5-15% (late) → ~3-4x speedup
- **Fast** (Logistic, LinearSVC, SGD, NaiveBayes, LDA, QDA): 10-27.5% (early), 10-17.5% (late) → ~2-3x speedup

**Note**: Row sampling was reduced by 50% across all categories to achieve 2x additional speedup while maintaining diversity through smaller, more varied samples.

**Adaptive CPU core allocation (n_jobs):**
- **GaussianProcess**: 3-5 cores (very slow O(n³), maximize parallelization)
- **kNN**: 2-4 cores (expensive distance calculations)
- **RandomForest/ExtraTrees**: 2-3 cores (independent tree building)
- **All other models**: 1 core (fast solvers or sequential algorithms)

With 24 available cores and 10 parallel jobs, this strategy:
- Gives more resources to bottleneck models
- Avoids coordination overhead for fast models
- Achieves ~1.5-2x additional speedup on slow model batches

**Combined optimizations:**
- Expected total training time: 15-45 minutes (down from 2-6 hours)
- Per model training: 0.5-4 seconds (down from 5-60 seconds)
- Per batch: 1.5-10 seconds (down from 10-120 seconds)
- Combined speedup: ~10-20x faster than original
- Maximum ensemble diversity maintained through randomization

### Hill climbing process

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

- **Simulated annealing**: Exploration-exploitation balance with adaptive temperature
- **Diversity metrics**: Pairwise model disagreement maximization
- **Transfer learning**: Incremental DNN expansion as ensemble grows
- **Parallel processing**: Multi-core batch training with adaptive n_jobs allocation
- **Adaptive sampling**: Classifier-aware row sampling (5-55% of data)
- **Feature engineering**: 15 transformer types with random combinations
- **Dimensionality reduction**: 5 techniques (PCA, TruncatedSVD, FastICA, NMF, FactorAnalysis)
- **Kernel approximation**: Nystroem, RBFSampler, SkewedChi2Sampler for non-linear features
- **Early stopping**: Plateau detection (100 iterations without improvement)
- **Checkpointing**: Resume from failures with full state preservation
- **WAL database**: Concurrent monitoring during training without locks

## Results

Training typically produces:
- **Ensemble size**: 30-50 diverse models
- **Classifier diversity**: All 13 classifier types represented
- **Feature engineering**: 1-3 transformers per model, ~50% use dimensionality reduction
- **Validation AUC**: 0.86-0.88
- **Training time**: 30-90 minutes (optimized from 2-6 hours)
- **Acceptance rate**: 10-20%
- **Memory usage**: 2-4 GB peak
- **CPU utilization**: Adaptive 1-5 cores per model based on complexity

## Documentation

- `MEMORY_TRACKING.md`: Detailed memory monitoring implementation
- `TIMING_TRACKING.md`: Timing metrics system documentation
- `TIMING_SUMMARY.md`: Performance analysis and optimization guide

## Contributing

This is a Kaggle competition project. Feel free to fork and experiment with different:
- Model types and hyperparameters (13 classifiers currently supported)
- Feature engineering transformers (15 types available)
- Dimensionality reduction techniques (5 methods implemented)
- Hill climbing strategies and acceptance criteria
- Meta-learner architectures and training strategies
- Diversity metrics and ensemble evaluation methods
- Performance optimizations (sampling strategies, CPU allocation)

The modular design makes it easy to:
- Add new classifier types in `ensemble_hill_climbing.py`
- Create custom transformers in `ensemble_transformers.py`
- Modify acceptance logic in simulated annealing
- Adjust adaptive sampling and n_jobs strategies

## License

See LICENSE file.

## Acknowledgments

- Kaggle Playground Series for the dataset
- scikit-learn and TensorFlow communities
- Streamlit for visualization framework
