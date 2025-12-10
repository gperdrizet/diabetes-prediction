"""Simulated annealing acceptance criterion for hill climbing.

Implements the core acceptance logic that decides whether to add a
candidate model to the ensemble based on performance and temperature.
"""

from typing import Tuple
import numpy as np


class AcceptanceCriterion:
    """Simulated annealing acceptance decision logic.
    
    Decides whether to accept a candidate model into the ensemble based on:
    1. Performance comparison (always accept better models)
    2. Temperature-based probabilistic acceptance (accept worse models with probability)
    
    The temperature controls exploration vs exploitation:
    - High temperature: More likely to accept worse models (exploration)
    - Low temperature: Only accept better models (exploitation)
    
    Attributes:
        temperature: Current acceptance temperature (0 < temp < 1)
        random_state: Random seed for reproducibility
        rng: NumPy random number generator
    
    Example:
        >>> criterion = AcceptanceCriterion(temperature=0.001, random_state=42)
        >>> accept, reason = criterion.should_accept(
        ...     current_score=0.650,
        ...     candidate_score=0.652
        ... )
        >>> if accept:
        ...     print(f"Accepted: {reason}")
    """
    
    def __init__(self, temperature: float, random_state: int):
        """Initialize the acceptance criterion.
        
        Args:
            temperature: Acceptance temperature (0 < temp < 1)
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If temperature is not in valid range
        """
        if not 0 < temperature < 1:
            raise ValueError(f"Temperature must be in (0, 1), got {temperature}")
        
        self.temperature = temperature
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def should_accept(
        self, 
        current_score: float,
        candidate_score: float
    ) -> Tuple[bool, str]:
        """Decide whether to accept a candidate model.
        
        Uses simulated annealing logic:
        - If candidate is better: Accept with probability 1.0
        - If candidate is worse: Accept with probability exp(delta / temperature)
        
        Args:
            current_score: Current best ensemble score
            candidate_score: Candidate model score
            
        Returns:
            Tuple of (accept: bool, reason: str)
            - accept: True if should add model to ensemble
            - reason: Human-readable explanation of decision
            
        Example:
            >>> accept, reason = criterion.should_accept(0.650, 0.655)
            >>> print(reason)
            'Improvement: 0.655 > 0.650'
        """
        score_delta = candidate_score - current_score
        
        # Case 1: Improvement - always accept
        if score_delta > 0:
            return True, f"Improvement: {candidate_score:.4f} > {current_score:.4f}"
        
        # Case 2: No change - always accept
        if score_delta == 0:
            return True, f"Equal: {candidate_score:.4f} == {current_score:.4f}"
        
        # Case 3: Degradation - probabilistic acceptance
        acceptance_prob = np.exp(score_delta / self.temperature)
        random_value = self.rng.random()
        
        if random_value < acceptance_prob:
            return True, (
                f"Probabilistic: {candidate_score:.4f} < {current_score:.4f}, "
                f"but accepted (prob={acceptance_prob:.4f}, rand={random_value:.4f})"
            )
        else:
            return False, (
                f"Rejected: {candidate_score:.4f} < {current_score:.4f}, "
                f"prob={acceptance_prob:.4f} < rand={random_value:.4f}"
            )
    
    def set_temperature(self, temperature: float):
        """Update the acceptance temperature.
        
        Used to implement temperature decay or adaptive temperature adjustment.
        
        Args:
            temperature: New temperature value (0 < temp < 1)
            
        Raises:
            ValueError: If temperature is not in valid range
            
        Example:
            >>> criterion.set_temperature(0.0005)
            >>> criterion.temperature
            0.0005
        """
        if not 0 < temperature < 1:
            raise ValueError(f"Temperature must be in (0, 1), got {temperature}")
        
        self.temperature = temperature
    
    def decay_temperature(self, decay_factor: float):
        """Apply multiplicative decay to temperature.
        
        Gradually reduces temperature to shift from exploration to exploitation.
        
        Args:
            decay_factor: Multiplicative factor (0 < factor < 1)
            
        Example:
            >>> criterion = AcceptanceCriterion(0.001, 42)
            >>> criterion.decay_temperature(0.998)
            >>> criterion.temperature  # Now 0.001 * 0.998 = 0.000998
        """
        if not 0 < decay_factor <= 1:
            raise ValueError(f"Decay factor must be in (0, 1], got {decay_factor}")
        
        self.temperature *= decay_factor
    
    def increase_temperature(self, increase_factor: float):
        """Apply multiplicative increase to temperature.
        
        Used for adaptive temperature adjustment when plateau is detected.
        Temporarily increases exploration.
        
        Args:
            increase_factor: Multiplicative factor (>= 1.0)
            
        Example:
            >>> criterion = AcceptanceCriterion(0.001, 42)
            >>> criterion.increase_temperature(1.2)
            >>> criterion.temperature  # Now 0.001 * 1.2 = 0.0012
        """
        if increase_factor < 1.0:
            raise ValueError(f"Increase factor must be >= 1.0, got {increase_factor}")
        
        new_temp = self.temperature * increase_factor
        
        # Cap at reasonable maximum to prevent runaway exploration
        if new_temp >= 1.0:
            self.temperature = 0.999
        else:
            self.temperature = new_temp
    
    def get_acceptance_probability(
        self, 
        current_score: float, 
        candidate_score: float
    ) -> float:
        """Calculate acceptance probability without making decision.
        
        Useful for analysis and debugging.
        
        Args:
            current_score: Current best ensemble score
            candidate_score: Candidate model score
            
        Returns:
            Acceptance probability (0 to 1)
            
        Example:
            >>> prob = criterion.get_acceptance_probability(0.650, 0.645)
            >>> print(f"Would accept with {prob:.1%} probability")
        """
        score_delta = candidate_score - current_score
        
        if score_delta >= 0:
            return 1.0
        else:
            return np.exp(score_delta / self.temperature)
    
    def summary(self) -> str:
        """Generate human-readable summary of current state.
        
        Returns:
            Multi-line string describing acceptance criterion state
        """
        return (
            f"Acceptance Criterion:\n"
            f"  Temperature: {self.temperature:.6f}\n"
            f"  Random State: {self.random_state}\n"
            f"  Behavior:\n"
            f"    - Always accepts improvements\n"
            f"    - Accepts degradations with prob = exp(delta / {self.temperature:.6f})\n"
            f"    - Lower temperature = less exploration"
        )
