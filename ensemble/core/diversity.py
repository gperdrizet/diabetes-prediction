"""Diversity scoring for ensemble models.

Measures how different models are from each other based on their predictions.
Higher diversity generally leads to better ensemble performance.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class DiversityScorer:
    """Calculates diversity metrics for ensemble models.
    
    Diversity is measured as the mean pairwise correlation between model
    predictions. Lower correlation indicates higher diversity.
    
    A diverse ensemble is more robust because:
    - Models make different errors
    - Averaging reduces variance
    - Less susceptible to overfitting
    
    Example:
        >>> scorer = DiversityScorer()
        >>> predictions = [model.predict_proba(X)[:, 1] for model in models]
        >>> diversity = scorer.score(predictions)
        >>> print(f"Diversity score: {diversity:.3f}")
    """
    
    def score(self, predictions: List[np.ndarray]) -> float:
        """Calculate mean pairwise correlation between predictions.
        
        Lower correlation = higher diversity = better
        
        Args:
            predictions: List of prediction arrays from different models.
                Each array should be 1D predictions for the same samples.
                
        Returns:
            Mean pairwise correlation (0 to 1). Lower is better.
            
        Raises:
            ValueError: If fewer than 2 prediction sets provided
            ValueError: If prediction arrays have different shapes
            
        Example:
            >>> preds = [
            ...     np.array([0.1, 0.2, 0.3]),
            ...     np.array([0.15, 0.25, 0.35]),
            ...     np.array([0.9, 0.8, 0.7])
            ... ]
            >>> diversity = scorer.score(preds)
        """
        if len(predictions) < 2:
            raise ValueError("Need at least 2 predictions to calculate diversity")
        
        # Validate shapes
        first_shape = predictions[0].shape
        for i, pred in enumerate(predictions[1:], 1):
            if pred.shape != first_shape:
                raise ValueError(
                    f"Prediction {i} has shape {pred.shape}, "
                    f"expected {first_shape}"
                )
        
        # Stack predictions into matrix (models x samples)
        pred_matrix = np.vstack(predictions)
        
        # Calculate correlation matrix between models
        corr_matrix = np.corrcoef(pred_matrix)
        
        # Extract upper triangle (excluding diagonal)
        # This gives us all unique pairwise correlations
        n = len(predictions)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_correlations = corr_matrix[upper_indices]
        
        # Return mean correlation (lower = more diverse)
        return float(np.mean(pairwise_correlations))
    
    def score_models(
        self, 
        models: List[BaseEstimator],
        X: pd.DataFrame
    ) -> float:
        """Calculate diversity directly from fitted models.
        
        Convenience method that generates predictions and scores them.
        
        Args:
            models: List of fitted sklearn models
            X: Feature DataFrame for prediction
            
        Returns:
            Mean pairwise correlation (lower = more diverse)
            
        Example:
            >>> diversity = scorer.score_models(ensemble_models, X_val)
        """
        predictions = []
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.decision_function(X)
            predictions.append(pred)
        
        return self.score(predictions)
    
    def detailed_diversity(
        self, 
        predictions: List[np.ndarray]
    ) -> dict:
        """Get detailed diversity statistics.
        
        Provides additional information beyond just mean correlation.
        
        Args:
            predictions: List of prediction arrays
            
        Returns:
            Dict with diversity statistics:
                - mean_correlation: Average pairwise correlation
                - min_correlation: Most diverse pair
                - max_correlation: Least diverse pair
                - std_correlation: Variability in correlations
                - n_pairs: Number of pairwise comparisons
                
        Example:
            >>> stats = scorer.detailed_diversity(predictions)
            >>> print(f"Diversity range: {stats['min_correlation']:.3f} to {stats['max_correlation']:.3f}")
        """
        if len(predictions) < 2:
            raise ValueError("Need at least 2 predictions to calculate diversity")
        
        # Stack and calculate correlation matrix
        pred_matrix = np.vstack(predictions)
        corr_matrix = np.corrcoef(pred_matrix)
        
        # Extract pairwise correlations
        n = len(predictions)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_correlations = corr_matrix[upper_indices]
        
        return {
            'mean_correlation': float(np.mean(pairwise_correlations)),
            'min_correlation': float(np.min(pairwise_correlations)),
            'max_correlation': float(np.max(pairwise_correlations)),
            'std_correlation': float(np.std(pairwise_correlations)),
            'n_pairs': len(pairwise_correlations),
            'n_models': n
        }
    
    def correlation_matrix(
        self, 
        predictions: List[np.ndarray]
    ) -> np.ndarray:
        """Get full correlation matrix between predictions.
        
        Useful for visualization and detailed analysis.
        
        Args:
            predictions: List of prediction arrays
            
        Returns:
            Correlation matrix (n_models x n_models)
            
        Example:
            >>> corr = scorer.correlation_matrix(predictions)
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            >>> plt.colorbar(label='Correlation')
            >>> plt.show()
        """
        pred_matrix = np.vstack(predictions)
        return np.corrcoef(pred_matrix)
    
    @staticmethod
    def is_diverse(mean_correlation: float, threshold: float = 0.7) -> bool:
        """Check if diversity is acceptable.
        
        Simple rule-of-thumb: mean correlation < 0.7 indicates good diversity.
        
        Args:
            mean_correlation: Mean pairwise correlation
            threshold: Maximum acceptable correlation (default 0.7)
            
        Returns:
            True if diverse enough, False otherwise
            
        Example:
            >>> diversity = scorer.score(predictions)
            >>> if scorer.is_diverse(diversity):
            ...     print("Good diversity!")
        """
        return mean_correlation < threshold
    
    def compare_ensembles(
        self,
        predictions_a: List[np.ndarray],
        predictions_b: List[np.ndarray],
        label_a: str = "Ensemble A",
        label_b: str = "Ensemble B"
    ) -> str:
        """Compare diversity of two ensembles.
        
        Args:
            predictions_a: Predictions from first ensemble
            predictions_b: Predictions from second ensemble
            label_a: Name for first ensemble
            label_b: Name for second ensemble
            
        Returns:
            Human-readable comparison summary
            
        Example:
            >>> comparison = scorer.compare_ensembles(
            ...     predictions_old, predictions_new,
            ...     "Before", "After"
            ... )
            >>> print(comparison)
        """
        stats_a = self.detailed_diversity(predictions_a)
        stats_b = self.detailed_diversity(predictions_b)
        
        lines = [
            "Ensemble Diversity Comparison",
            "=" * 50,
            f"{label_a}:",
            f"  Mean correlation: {stats_a['mean_correlation']:.4f}",
            f"  Range: [{stats_a['min_correlation']:.4f}, {stats_a['max_correlation']:.4f}]",
            f"  Models: {stats_a['n_models']}",
            "",
            f"{label_b}:",
            f"  Mean correlation: {stats_b['mean_correlation']:.4f}",
            f"  Range: [{stats_b['min_correlation']:.4f}, {stats_b['max_correlation']:.4f}]",
            f"  Models: {stats_b['n_models']}",
            "",
            "Difference:",
            f"  Δ Mean: {stats_b['mean_correlation'] - stats_a['mean_correlation']:+.4f}",
            f"  {'More diverse' if stats_b['mean_correlation'] < stats_a['mean_correlation'] else 'Less diverse'} (lower is better)"
        ]
        
        return "\n".join(lines)
