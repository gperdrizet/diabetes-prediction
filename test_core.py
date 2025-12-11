"""Test script for Phase 2: Core Abstractions.

Validates that the core modules work correctly:
- DataSampler
- AcceptanceCriterion
- DiversityScorer
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ensemble.config import SamplingConfig
from ensemble.core import DataSampler, AcceptanceCriterion, DiversityScorer


def test_data_sampler():
    """Test DataSampler functionality."""
    print("Testing DataSampler...")
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(1000, 50),
        columns=[f'feature_{i}' for i in range(50)]
    )
    y = pd.Series(np.random.choice([0, 1], size=1000, p=[0.85, 0.15]))
    
    # Create sampler
    config = SamplingConfig(
        row_sample_range=(0.10, 0.40),
        column_sample_range=(0.30, 0.70)
    )
    sampler = DataSampler(config, random_state=42)
    
    # Test row sampling
    X_row, y_row = sampler.sample_rows(X, y)
    assert len(X_row) < len(X), "Row sampling should reduce size"
    assert len(X_row) == len(y_row), "X and y should have same length"
    
    # Check stratification preserved
    original_pos_ratio = y.mean()
    sampled_pos_ratio = y_row.mean()
    assert abs(sampled_pos_ratio - original_pos_ratio) < 0.05, "Stratification should preserve class ratio"
    
    print(f"  ✓ Row sampling: {len(X)} → {len(X_row)} rows")
    print(f"  ✓ Class ratio preserved: {original_pos_ratio:.3f} → {sampled_pos_ratio:.3f}")
    
    # Test column sampling
    columns = sampler.sample_columns(X)
    assert len(columns) < X.shape[1], "Column sampling should reduce features"
    assert all(col in X.columns for col in columns), "All selected columns should exist"
    
    print(f"  ✓ Column sampling: {X.shape[1]} → {len(columns)} columns")
    
    # Test combined sampling
    X_both, y_both = sampler.sample_both(X, y)
    assert X_both.shape[0] < X.shape[0], "Should have fewer rows"
    assert X_both.shape[1] < X.shape[1], "Should have fewer columns"
    assert len(y_both) == len(X_both), "X and y should match"
    
    print(f"  ✓ Combined sampling: {X.shape} → {X_both.shape}")
    
    # Test info method
    info = sampler.get_sample_info(X, y)
    assert 'expected_shape' in info
    assert info['original_shape'] == X.shape
    
    print(f"  ✓ Sample info: {info['expected_shape']}")


def test_acceptance_criterion():
    """Test AcceptanceCriterion functionality."""
    print("\nTesting AcceptanceCriterion...")
    
    criterion = AcceptanceCriterion(temperature=0.001, random_state=42)
    
    # Test improvement acceptance
    accept, reason = criterion.should_accept(0.650, 0.655)
    assert accept == True, "Should always accept improvements"
    assert "Improvement" in reason
    print(f"  ✓ Improvement: {reason}")
    
    # Test equal acceptance
    accept, reason = criterion.should_accept(0.650, 0.650)
    assert accept == True, "Should accept equal scores"
    assert "Equal" in reason
    print(f"  ✓ Equal: {reason}")
    
    # Test probabilistic acceptance (run multiple times)
    accepts = []
    for _ in range(100):
        criterion_test = AcceptanceCriterion(temperature=0.001, random_state=None)
        accept, reason = criterion_test.should_accept(0.650, 0.645)
        accepts.append(accept)
    
    accept_rate = sum(accepts) / len(accepts)
    print(f"  ✓ Probabilistic: {accept_rate:.1%} acceptance rate for -0.005 delta")
    
    # Test temperature decay
    initial_temp = criterion.temperature
    criterion.decay_temperature(0.998)
    assert criterion.temperature < initial_temp, "Decay should reduce temperature"
    print(f"  ✓ Temperature decay: {initial_temp:.6f} → {criterion.temperature:.6f}")
    
    # Test temperature increase
    before_increase = criterion.temperature
    criterion.increase_temperature(1.2)
    assert criterion.temperature > before_increase, "Increase should raise temperature"
    print(f"  ✓ Temperature increase: {before_increase:.6f} → {criterion.temperature:.6f}")
    
    # Test acceptance probability
    prob = criterion.get_acceptance_probability(0.650, 0.645)
    assert 0 <= prob <= 1, "Probability should be in [0, 1]"
    print(f"  ✓ Acceptance probability: {prob:.4f}")


def test_diversity_scorer():
    """Test DiversityScorer functionality."""
    print("\nTesting DiversityScorer...")
    
    scorer = DiversityScorer()
    
    # Create sample predictions
    np.random.seed(42)
    n_samples = 1000
    
    # High diversity: uncorrelated predictions
    pred1 = np.random.rand(n_samples)
    pred2 = np.random.rand(n_samples)
    pred3 = np.random.rand(n_samples)
    
    diversity_high = scorer.score([pred1, pred2, pred3])
    print(f"  ✓ High diversity (random): {diversity_high:.4f}")
    
    # Low diversity: highly correlated predictions
    pred4 = pred1 + np.random.randn(n_samples) * 0.01
    pred5 = pred1 + np.random.randn(n_samples) * 0.01
    
    diversity_low = scorer.score([pred1, pred4, pred5])
    print(f"  ✓ Low diversity (correlated): {diversity_low:.4f}")
    
    assert diversity_low > diversity_high, "Correlated predictions should have higher correlation"
    
    # Test detailed diversity
    stats = scorer.detailed_diversity([pred1, pred2, pred3])
    assert 'mean_correlation' in stats
    assert 'min_correlation' in stats
    assert 'max_correlation' in stats
    assert stats['n_models'] == 3
    print(f"  ✓ Detailed stats: mean={stats['mean_correlation']:.4f}, "
          f"range=[{stats['min_correlation']:.4f}, {stats['max_correlation']:.4f}]")
    
    # Test correlation matrix
    corr_matrix = scorer.correlation_matrix([pred1, pred2, pred3])
    assert corr_matrix.shape == (3, 3), "Correlation matrix should be n_models x n_models"
    assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1.0"
    print(f"  ✓ Correlation matrix shape: {corr_matrix.shape}")
    
    # Test diversity check
    is_diverse = scorer.is_diverse(diversity_high)
    print(f"  ✓ Diversity check: {is_diverse} (threshold 0.7)")
    
    # Test comparison
    comparison = scorer.compare_ensembles(
        [pred1, pred4, pred5],
        [pred1, pred2, pred3],
        "Low Diversity",
        "High Diversity"
    )
    assert "Ensemble Diversity Comparison" in comparison
    print(f"  ✓ Comparison generated ({len(comparison)} chars)")


def test_integration():
    """Test that all components work together."""
    print("\nTesting integration...")
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(1000, 50),
        columns=[f'feature_{i}' for i in range(50)]
    )
    y = pd.Series(np.random.choice([0, 1], size=1000))
    
    # Create components
    config = SamplingConfig()
    sampler = DataSampler(config, random_state=42)
    criterion = AcceptanceCriterion(temperature=0.001, random_state=42)
    scorer = DiversityScorer()
    
    # Simulate hill climbing iteration
    X_sample, y_sample = sampler.sample_both(X, y)
    print(f"  ✓ Sampled data: {X_sample.shape}")
    
    # Simulate scoring
    current_score = 0.650
    candidate_score = 0.655
    accept, reason = criterion.should_accept(current_score, candidate_score)
    print(f"  ✓ Acceptance decision: {accept} ({reason})")
    
    # Simulate diversity check
    predictions = [np.random.rand(100) for _ in range(3)]
    diversity = scorer.score(predictions)
    print(f"  ✓ Diversity score: {diversity:.4f}")
    
    print("  ✓ All components work together")


def main():
    """Run all tests."""
    print("="*70)
    print("PHASE 2: CORE ABSTRACTIONS VALIDATION")
    print("="*70)
    
    try:
        test_data_sampler()
        test_acceptance_criterion()
        test_diversity_scorer()
        test_integration()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nCore abstractions are working correctly.")
        print("Ready to proceed with Phase 3 (Stage 1 Isolation).")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
