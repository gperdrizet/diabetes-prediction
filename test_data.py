"""Test script for Phase 5: Data Management.

Tests data splitting and preprocessing utilities.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from ensemble.data.splits import DataSplits, create_three_way_split
from ensemble.data.preprocessing import (
    create_base_preprocessor,
    get_preprocessor_info,
    print_preprocessor_summary
)


def test_data_splits():
    """Test DataSplits class."""
    print("Testing DataSplits class...")
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    df['label'] = y
    
    # Create splits
    splits = DataSplits(
        data=df,
        label_column='label',
        random_state=42,
        train_pool_size=0.60,
        val_stage1_size=0.35,
        val_stage2_size=0.05
    )
    
    # Get all splits
    X_train, X_val_s1, X_val_s2, y_train, y_val_s1, y_val_s2 = splits.get_all_splits()
    
    # Check sizes
    total_size = len(X_train) + len(X_val_s1) + len(X_val_s2)
    assert total_size == 1000, f"Total size mismatch: {total_size}"
    
    train_pct = len(X_train) / total_size
    val_s1_pct = len(X_val_s1) / total_size
    val_s2_pct = len(X_val_s2) / total_size
    
    print(f"  ✓ Training pool: {len(X_train)} ({train_pct*100:.1f}%)")
    print(f"  ✓ Val Stage 1: {len(X_val_s1)} ({val_s1_pct*100:.1f}%)")
    print(f"  ✓ Val Stage 2: {len(X_val_s2)} ({val_s2_pct*100:.1f}%)")
    
    # Check that sizes are approximately correct (±1% due to rounding)
    assert abs(train_pct - 0.60) < 0.01, f"Training pool size wrong: {train_pct:.3f}"
    assert abs(val_s1_pct - 0.35) < 0.01, f"Val Stage 1 size wrong: {val_s1_pct:.3f}"
    assert abs(val_s2_pct - 0.05) < 0.01, f"Val Stage 2 size wrong: {val_s2_pct:.3f}"
    
    # Check stratification (class distribution preserved)
    full_pos_rate = y.mean()
    train_pos_rate = y_train.mean()
    val_s1_pos_rate = y_val_s1.mean()
    val_s2_pos_rate = y_val_s2.mean()
    
    print(f"  ✓ Full dataset positive rate: {full_pos_rate:.3f}")
    print(f"  ✓ Training pool positive rate: {train_pos_rate:.3f}")
    print(f"  ✓ Val S1 positive rate: {val_s1_pos_rate:.3f}")
    print(f"  ✓ Val S2 positive rate: {val_s2_pos_rate:.3f}")
    
    # Check that stratification is reasonable (±5%)
    assert abs(train_pos_rate - full_pos_rate) < 0.05, "Training stratification failed"
    assert abs(val_s1_pos_rate - full_pos_rate) < 0.05, "Val S1 stratification failed"
    assert abs(val_s2_pos_rate - full_pos_rate) < 0.05, "Val S2 stratification failed"
    
    # Test individual getters
    X_train2, y_train2 = splits.get_train_pool()
    assert len(X_train2) == len(X_train), "get_train_pool mismatch"
    
    X_val_s1_2, y_val_s1_2 = splits.get_val_stage1()
    assert len(X_val_s1_2) == len(X_val_s1), "get_val_stage1 mismatch"
    
    X_val_s2_2, y_val_s2_2 = splits.get_val_stage2()
    assert len(X_val_s2_2) == len(X_val_s2), "get_val_stage2 mismatch"
    
    print("  ✓ Individual getters work correctly")
    
    # Test summary
    summary = splits.summary()
    assert 'Data Splits Summary' in summary, "Summary missing title"
    assert 'Training pool' in summary, "Summary missing training pool"
    print("  ✓ Summary generation works")


def test_convenience_function():
    """Test create_three_way_split convenience function."""
    print("\nTesting create_three_way_split convenience function...")
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(8)])
    df['target'] = y
    
    # Create splits using convenience function
    X_train, X_val_s1, X_val_s2, y_train, y_val_s1, y_val_s2 = create_three_way_split(
        data=df,
        label_column='target',
        random_state=42
    )
    
    total = len(X_train) + len(X_val_s1) + len(X_val_s2)
    assert total == 500, f"Total size mismatch: {total}"
    
    print(f"  ✓ Created splits: {len(X_train)}, {len(X_val_s1)}, {len(X_val_s2)}")


def test_preprocessing():
    """Test preprocessing utilities."""
    print("\nTesting preprocessing utilities...")
    
    # Define feature types
    numerical_features = ['age', 'height', 'weight', 'bmi']
    ordinal_features = ['education', 'income']
    nominal_features = ['gender', 'smoking_status']
    
    # Define ordinal categories
    ordinal_categories = [
        ['elementary', 'high_school', 'college', 'graduate'],  # education
        ['low', 'medium', 'high', 'very_high']  # income
    ]
    
    # Create preprocessor
    preprocessor = create_base_preprocessor(
        numerical_features=numerical_features,
        ordinal_features=ordinal_features,
        nominal_features=nominal_features,
        ordinal_categories=ordinal_categories
    )
    
    print(f"  ✓ Created preprocessor with {len(preprocessor.transformers)} transformers")
    
    # Check transformer configuration
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert 'num' in transformer_names, "Missing numerical transformer"
    assert 'ord' in transformer_names, "Missing ordinal transformer"
    assert 'nom' in transformer_names, "Missing nominal transformer"
    print(f"  ✓ Transformers: {transformer_names}")
    
    # Test preprocessor info
    info = get_preprocessor_info(preprocessor)
    assert info['n_transformers'] == 3, "Wrong number of transformers"
    print(f"  ✓ Preprocessor info: {info['n_transformers']} transformers")
    
    # Test with actual data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'age': np.random.randint(20, 80, 100),
        'height': np.random.uniform(150, 200, 100),
        'weight': np.random.uniform(50, 120, 100),
        'bmi': np.random.uniform(18, 35, 100),
        'education': np.random.choice(['elementary', 'high_school', 'college', 'graduate'], 100),
        'income': np.random.choice(['low', 'medium', 'high', 'very_high'], 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'smoking_status': np.random.choice(['never', 'former', 'current'], 100)
    })
    
    # Fit and transform
    preprocessor.fit(test_data)
    transformed = preprocessor.transform(test_data)
    
    print(f"  ✓ Fitted preprocessor: {test_data.shape} → {transformed.shape}")
    assert transformed.shape[0] == 100, "Row count changed"
    
    # Print summary
    print("\n  Summary output:")
    print_preprocessor_summary(
        preprocessor=preprocessor,
        numerical_features=numerical_features,
        ordinal_features=ordinal_features,
        nominal_features=nominal_features
    )


def test_integration():
    """Test integration of splits and preprocessing."""
    print("\nTesting integration...")
    
    # Create synthetic data with mixed types
    np.random.seed(42)
    n_samples = 800
    
    data = pd.DataFrame({
        'num1': np.random.randn(n_samples),
        'num2': np.random.randn(n_samples),
        'num3': np.random.randn(n_samples),
        'ord1': np.random.choice(['low', 'medium', 'high'], n_samples),
        'nom1': np.random.choice(['A', 'B', 'C'], n_samples),
        'nom2': np.random.choice(['X', 'Y'], n_samples),
        'label': np.random.randint(0, 2, n_samples)
    })
    
    # Create splits
    splits = DataSplits(data=data, label_column='label', random_state=42)
    X_train, y_train = splits.get_train_pool()
    
    print(f"  ✓ Created splits: training pool has {len(X_train)} samples")
    
    # Create preprocessor
    preprocessor = create_base_preprocessor(
        numerical_features=['num1', 'num2', 'num3'],
        ordinal_features=['ord1'],
        nominal_features=['nom1', 'nom2'],
        ordinal_categories=[['low', 'medium', 'high']]
    )
    
    # Fit on training data
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    
    print(f"  ✓ Preprocessed training data: {X_train.shape} → {X_train_transformed.shape}")
    
    # Transform validation data
    X_val_s1, y_val_s1 = splits.get_val_stage1()
    X_val_s1_transformed = preprocessor.transform(X_val_s1)
    
    print(f"  ✓ Preprocessed validation data: {X_val_s1.shape} → {X_val_s1_transformed.shape}")
    
    # Check that output shapes match
    assert X_train_transformed.shape[1] == X_val_s1_transformed.shape[1], "Feature count mismatch"
    print(f"  ✓ Integration successful!")


if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 5: DATA MANAGEMENT VALIDATION")
    print("=" * 80)
    
    try:
        test_data_splits()
        test_convenience_function()
        test_preprocessing()
        test_integration()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nData management modules are working correctly.")
        print("Components validated:")
        print("  - DataSplits class (60/35/5 stratified split)")
        print("  - Stratification preservation")
        print("  - Convenience function")
        print("  - Base preprocessor creation")
        print("  - Numerical, ordinal, and nominal encoding")
        print("  - Integration with data splits")
        print("\nReady to proceed with Phase 6 (Tracking Simplification).")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
