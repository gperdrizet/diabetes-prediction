# Legacy Functions Archive

**DEPRECATED**: This directory contains archived legacy code that has been superseded by the refactored `ensemble/` package.

## Status

These functions are **no longer used** in the production ensemble training pipeline. They have been replaced by equivalent functionality in the refactored codebase:

- `ensemble_hill_climbing.py` → `ensemble/stage1/pipeline_builder.py` (PipelineBuilder class)
- `ensemble_config.py` → `ensemble/config.py` (EnsembleConfig dataclass)
- `ensemble_transformers.py` → `ensemble/stage1/transformers/` (modular transformers)
- `ensemble_database.py` → `ensemble/tracking/database.py` (EnsembleDatabase class)
- `ensemble_initialization.py` → `ensemble/data/` (data splitting and preprocessing)
- `ensemble_evaluation.py` → `ensemble/core/diversity.py` (DiversityScorer)
- `ensemble_parallel.py` → `ensemble/parallel/` (worker and scheduler modules)
- `ensemble_stage2_training.py` → `ensemble/stage2/` (Stage 2 DNN training)
- `ensemble_stage2_model.py` → `ensemble/stage2/model.py` (DNN model building)

## Migration

The refactored codebase provides:
- **Type safety**: Uses dataclasses with validation
- **Modularity**: Clean separation of concerns
- **Testability**: Each component can be tested independently
- **Maintainability**: Clear interfaces and documentation
- **No path manipulation**: Proper Python package structure

## Reference Notebook

The notebook `02.1-ensemble_hill_climbing.ipynb` uses these legacy functions and is kept for reference purposes only. For current ensemble training, use:

- **Production**: `03.1-ensemble_training_refactored.ipynb`

## Historical Context

This code was developed during the initial ensemble implementation phase and used a flat module structure with global configuration dictionaries. It has been successfully refactored into a proper Python package with:

- Object-oriented design patterns
- Configuration management via dataclasses
- Proper error handling and logging
- Parallel processing with timeout protection
- Database tracking and metrics collection

## Do Not Use

**Do not import or use these modules in new code.** They are preserved only for historical reference and to understand the evolution of the codebase.

---

*Archived: December 12, 2025*
*Superseded by: `ensemble/` package*
