"""Core ensemble hill climbing abstractions.

This subpackage provides clean, testable implementations of the core
hill climbing concepts:
- Data sampling (rows and columns)
- Pipeline building (Stage 1 sklearn models)
- Acceptance criteria (simulated annealing)
- Diversity scoring (model correlation)
"""

from ensemble.core.sampler import DataSampler
from ensemble.core.acceptance import AcceptanceCriterion
from ensemble.core.diversity import DiversityScorer

__all__ = [
    'DataSampler',
    'AcceptanceCriterion',
    'DiversityScorer'
]
