# Performance Optimization Guide

## Current Performance Bottlenecks

### Critical Issues Found

1. **Slow Classifiers**
   - **SVC with probability=True**: Extremely slow, especially with large datasets (10k-50k samples)
   - **MLPClassifier with max_iter=1000**: Neural networks are slow trainers
   - **Large Ensemble Sizes**: Random Forest/Gradient Boosting using 30-300 estimators
   
2. **Data Sampling Overhead**
   - Each model trains on 10,000-50,000 randomly sampled rows
   - `train_test_split` creates copies of data for each model
   - With 10 parallel workers, this means 100k-500k rows in memory simultaneously

3. **No Random State Control**
   - All models use `# No random_state for diversity` 
   - Makes training times highly variable and non-reproducible
   - Some models may get "unlucky" initialization and take much longer

4. **Feature Engineering Pipeline**
   - 1-3 random transformers per model
   - PCA (50% chance) adds significant overhead
   - Custom transformers may be inefficient

## Performance Analysis by Classifier

**Slow Classifiers** (10+ seconds per model):
- **SVC**: O(n²) to O(n³) complexity, worst with probability=True
- **MLPClassifier**: Neural network training, very slow
- **KNN**: Slow prediction, though fast training

**Medium Speed** (1-10 seconds):
- **Random Forest**: Depends on n_estimators (30-300)
- **Gradient Boosting**: Depends on max_iter (30-300)
- **Extra Trees**: Similar to Random Forest
- **AdaBoost**: Depends on n_estimators (30-300)

**Fast Classifiers** (<1 second):
- **Logistic Regression**: Very fast, scales well
- **Ridge Classifier**: Very fast

## Optimization Strategies

### Quick Wins (Immediate Implementation)

1. **Reduce Ensemble Sizes**
   ```python
   # In generate_random_pipeline():
   # Change from:
   n_estimators = int(10 ** rng.uniform(1.5, 2.5))  # ~30 to 300
   # To:
   n_estimators = int(10 ** rng.uniform(1.0, 2.0))  # ~10 to 100
   ```

2. **Reduce MLP max_iter**
   ```python
   # Change from:
   max_iter=1000
   # To:
   max_iter=300  # Early stopping will handle convergence
   ```

3. **Remove SVC or Make it Faster**
   ```python
   # Option A: Remove SVC entirely from classifier_options
   classifier_options = [
       'logistic_regression',
       'random_forest',
       'gradient_boosting',
       # 'svc',  # REMOVE - too slow
       'mlp',
       'knn',
       'extra_trees',
       'adaboost'
   ]
   
   # Option B: Use LinearSVC instead (much faster)
   from sklearn.svm import LinearSVC
   classifier = LinearSVC(
       C=C,
       class_weight='balanced',
       max_iter=2000
   )
   ```

4. **Reduce Data Sample Sizes**
   ```python
   # In prepare_training_batch():
   # Change from:
   iteration_sample_size = rng.randint(10000, 50001)
   # To:
   iteration_sample_size = rng.randint(5000, 20001)  # 50-60% reduction
   ```

5. **Increase Batch Size**
   ```python
   # In notebook configuration:
   BATCH_SIZE = 20  # Instead of 10
   N_WORKERS = min(20, n_cpus)  # Use more workers
   ```

### Medium-Term Optimizations

6. **Add Early Stopping to Tree Models**
   ```python
   # For Random Forest, Extra Trees:
   min_samples_split = int(10 ** rng.uniform(1, 1.7))  # 10 to 50 (was 2-20)
   min_samples_leaf = int(10 ** rng.uniform(0.7, 1.3))  # 5 to 20 (was 1-10)
   ```

7. **Reduce Feature Engineering Complexity**
   ```python
   # Reduce PCA probability:
   use_pca = rng.random() < 0.3  # 30% instead of 50%
   
   # Reduce number of transformers:
   n_transformers = rng.randint(1, 3)  # 1-2 instead of 1-3
   ```

8. **Use Faster KDE Bandwidth**
   ```python
   # KDE with 'scott' or 'silverman' can be slow
   # Consider removing KDE or optimizing implementation
   ```

### Advanced Optimizations

9. **Implement Time Budget per Model**
   ```python
   # In train_single_candidate():
   import signal
   
   def timeout_handler(signum, frame):
       raise TimeoutError("Model training exceeded time limit")
   
   # Set 60 second timeout
   signal.signal(signal.SIGALRM, timeout_handler)
   signal.alarm(60)  # 60 seconds
   try:
       fitted_pipeline = pipeline.fit(X_train, y_train)
   finally:
       signal.alarm(0)  # Cancel alarm
   ```

10. **Use Sklearn's n_jobs Carefully**
    ```python
    # Current setting disables parallelism within models:
    n_jobs=1  # CORRECT for ProcessPoolExecutor
    
    # DO NOT change to n_jobs=-1 when using ProcessPoolExecutor
    # This would cause nested parallelism and severe slowdown
    ```

11. **Pre-sample Data Once**
    ```python
    # Instead of sampling in prepare_training_batch,
    # create fixed samples upfront and reuse them
    # Trade memory for speed
    ```

12. **Add Weighted Sampling for Faster Models**
    ```python
    # Bias toward faster classifiers:
    classifier_weights = {
        'logistic_regression': 0.20,
        'random_forest': 0.15,
        'gradient_boosting': 0.15,
        'svc': 0.05,  # Reduce probability
        'mlp': 0.10,  # Reduce probability
        'knn': 0.10,
        'extra_trees': 0.15,
        'adaboost': 0.10
    }
    classifier_type = rng.choice(
        list(classifier_weights.keys()),
        p=list(classifier_weights.values())
    )
    ```

## Recommended Configuration Changes

### Conservative (2-3x speedup)

```python
# In ensemble_hill_climbing.py, generate_random_pipeline():

# Tree models:
n_estimators = int(10 ** rng.uniform(1.0, 2.0))  # 10 to 100 (was 30-300)
max_iter = int(10 ** rng.uniform(1.0, 2.0))  # 10 to 100 (was 30-300)

# MLP:
max_iter=300  # (was 1000)

# Sample sizes in ensemble_parallel.py:
iteration_sample_size = rng.randint(5000, 20001)  # (was 10000-50001)

# Remove SVC:
classifier_options = [
    'logistic_regression',
    'random_forest',
    'gradient_boosting',
    # 'svc',  # REMOVED
    'mlp',
    'knn',
    'extra_trees',
    'adaboost'
]
```

### Aggressive (5-10x speedup)

```python
# All of conservative, plus:

# Even smaller ensembles:
n_estimators = int(10 ** rng.uniform(0.7, 1.7))  # 5 to 50

# Smaller sample sizes:
iteration_sample_size = rng.randint(3000, 10001)  # 3k-10k

# Remove slow classifiers:
classifier_options = [
    'logistic_regression',
    'random_forest',
    'gradient_boosting',
    'extra_trees'
]

# Reduce feature engineering:
n_transformers = rng.randint(0, 2)  # 0-1 transformers
use_pca = rng.random() < 0.2  # 20% chance
```

## Expected Training Times

### Current Configuration
- **Per model**: 5-60 seconds (highly variable)
- **Per batch (10 models)**: 10-120 seconds
- **500 iterations**: 2-6 hours (as you're experiencing)

### With Conservative Changes
- **Per model**: 2-15 seconds
- **Per batch (10 models)**: 5-30 seconds
- **500 iterations**: 45 minutes - 2 hours

### With Aggressive Changes
- **Per model**: 1-5 seconds
- **Per batch (10 models)**: 2-10 seconds
- **500 iterations**: 15-45 minutes

## Monitoring and Debugging

### Check Current Training Progress

```python
# In notebook, add timing info:
import sqlite3
conn = sqlite3.connect('/workspaces/diabetes-prediction/data/ensemble_training.db')
df = pd.read_sql("SELECT iteration, classifier_type, stage1_val_auc, training_time_sec FROM ensemble_log ORDER BY iteration DESC LIMIT 20", conn)
conn.close()
print(df)
```

### Identify Slow Models

```python
# Check which classifiers are slowest:
conn = sqlite3.connect('/workspaces/diabetes-prediction/data/ensemble_training.db')
df = pd.read_sql("""
    SELECT 
        classifier_type,
        COUNT(*) as count,
        AVG(training_time_sec) as avg_time,
        MAX(training_time_sec) as max_time
    FROM ensemble_log 
    WHERE training_time_sec IS NOT NULL
    GROUP BY classifier_type
    ORDER BY avg_time DESC
""", conn)
conn.close()
print(df)
```

## Implementation Priority

1. **Immediate** (do now):
   - Remove SVC from classifier options
   - Reduce MLP max_iter to 300
   - Reduce sample sizes to 5k-20k

2. **Next** (if still too slow):
   - Reduce ensemble sizes (n_estimators)
   - Reduce feature engineering complexity
   - Increase batch size to 20

3. **If desperate**:
   - Remove MLP entirely
   - Use only fast classifiers (Logistic, trees with small ensembles)
   - Reduce to 100-200 total iterations instead of 500
