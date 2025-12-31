#!/usr/bin/env python3
"""Prepare Kaggle dataset assets from ensemble checkpoint.

This script stages checkpoint files for manual upload to Kaggle as a dataset.
"""

import argparse
import shutil
import json
from pathlib import Path


def find_latest_run(models_dir):
    """Find the most recent run directory."""
    run_dirs = sorted([d for d in models_dir.glob('run_*') if d.is_dir()], reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {models_dir}")
    return run_dirs[0]


def find_checkpoint(run_dir, checkpoint_num):
    """Find specified checkpoint in run directory."""
    checkpoints_dir = run_dir / 'checkpoints'
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory in {run_dir}")
    
    if checkpoint_num == 'latest':
        checkpoint_dirs = sorted(checkpoints_dir.glob('checkpoint_*'))
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        return checkpoint_dirs[-1]
    else:
        checkpoint_dir = checkpoints_dir / f'checkpoint_{checkpoint_num}'
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
        return checkpoint_dir


def prepare_assets(source_checkpoint, output_dir):
    """Copy checkpoint files to staging directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Required files
    required_files = [
        'ensemble_stage1_models.joblib',
        'stage2_model.h5',
        'ensemble_classifier_transformers.py',
        'metadata.json'
    ]
    
    file_sizes = {}
    
    for filename in required_files:
        source = source_checkpoint / filename
        dest = output_dir / filename
        
        if not source.exists():
            raise FileNotFoundError(f"Required file not found: {source}")
        
        shutil.copy(source, dest)
        file_sizes[filename] = dest.stat().st_size
    
    return file_sizes


def create_readme(output_dir, metadata, file_sizes):
    """Create README for Kaggle dataset."""
    readme_content = f"""# Diabetes Prediction Ensemble Model - Checkpoint {metadata['checkpoint_num']:03d}

## Model Information

- **Ensemble Size**: {metadata['ensemble_size']} Stage 1 models
- **Stage 1 Validation AUC**: {metadata['stage1_val_auc']:.4f}
- **Stage 2 Validation AUC**: {metadata['stage2_val_auc']:.4f}
- **Retraining Count**: {metadata['retraining_count']}
- **Timestamp**: {metadata['timestamp']}

## Pseudo-Labeling

{f"**Enabled**: {metadata['pseudo_labeling']}" if metadata.get('pseudo_labeling', {}).get('enabled') else "**Status**: Disabled"}

## Files Included

"""
    
    for filename, size_bytes in file_sizes.items():
        size_mb = size_bytes / (1024**2)
        size_kb = size_bytes / 1024
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{size_kb:.1f} KB"
        readme_content += f"- `{filename}` ({size_str})\n"
    
    readme_content += """
## Usage in Kaggle Notebook

```python
import sys
import joblib
from pathlib import Path

# Configuration
KAGGLE = True
CHECKPOINT = '""" + f"{metadata['checkpoint_num']:03d}" + """'

# Set paths
checkpoint_dataset = f'diabetes-ensemble-checkpoint-{CHECKPOINT}'
module_path = Path(f'/kaggle/input/{checkpoint_dataset}')
sys.path.insert(0, str(module_path))

# Import ensemble classifier
from ensemble_classifier import EnsembleClassifier

# Load test data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Load model
model_path = module_path / 'ensemble_stage1_models.joblib'
stage2_model_path = module_path / 'stage2_model.h5'

model = joblib.load(model_path)
model.stage2_model_path = str(stage2_model_path)

# Make predictions (probabilities)
predictions_proba = model.predict_proba(test_df)
predictions = predictions_proba[:, 1]  # Positive class probability

# Create submission
submission_df = pd.DataFrame({
    'id': test_df['id'].astype(int),
    'diagnosed_diabetes': predictions
})
submission_df.to_csv('submission.csv', index=False)
```

## Dataset Upload Instructions

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload all 4 files from this directory
4. Name the dataset: `diabetes-ensemble-checkpoint-""" + f"{metadata['checkpoint_num']:03d}" + """`
5. Make the dataset public or private as needed
6. In your Kaggle notebook, add this dataset as a data source
7. Run the inference code above

## Notes

- This checkpoint contains """ + f"{metadata['ensemble_size']}" + """ Stage 1 models
- Stage 2 DNN is saved separately for compatibility
- All custom transformers are bundled in `ensemble_classifier_transformers.py`
- The model expects raw test data (no preprocessing needed)
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return readme_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ensemble checkpoint assets for Kaggle dataset upload'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint number (e.g., "003") or "latest"'
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=None,
        help='Training run directory (default: latest run in models/)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output staging directory (default: kaggle_assets/checkpoint_XXX/)'
    )
    
    args = parser.parse_args()
    
    # Find run directory
    if args.run_dir is None:
        models_dir = Path(__file__).parent.parent / 'models'
        run_dir = find_latest_run(models_dir)
        print(f"Using latest run: {run_dir.name}")
    else:
        run_dir = args.run_dir
    
    # Find checkpoint
    source_checkpoint = find_checkpoint(run_dir, args.checkpoint)
    
    # Load metadata
    metadata_path = source_checkpoint / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    checkpoint_num = metadata['checkpoint_num']
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent.parent / 'kaggle_assets' / f'checkpoint_{checkpoint_num:03d}'
    else:
        output_dir = args.output_dir
    
    print(f"\n{'='*70}")
    print(f"Preparing Kaggle Assets")
    print(f"{'='*70}")
    print(f"Source checkpoint: {source_checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint number: {checkpoint_num:03d}")
    print(f"Ensemble size: {metadata['ensemble_size']} models")
    print(f"Stage 1 AUC: {metadata['stage1_val_auc']:.4f}")
    print(f"Stage 2 AUC: {metadata['stage2_val_auc']:.4f}")
    print()
    
    # Copy files
    print("Copying checkpoint files...")
    file_sizes = prepare_assets(source_checkpoint, output_dir)
    
    # Create README
    print("Creating README...")
    readme_path = create_readme(output_dir, metadata, file_sizes)
    
    # Calculate total size
    total_size_mb = sum(file_sizes.values()) / (1024**2)
    
    print(f"\n{'='*70}")
    print("Assets staged successfully!")
    print(f"{'='*70}")
    print(f"\nFiles staged in: {output_dir}")
    print(f"\nFile list:")
    for filename, size_bytes in file_sizes.items():
        size_mb = size_bytes / (1024**2)
        size_kb = size_bytes / 1024
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{size_kb:.1f} KB"
        print(f"  ✓ {filename:45s} {size_str:>12s}")
    print(f"  ✓ {readme_path.name:45s} {readme_path.stat().st_size / 1024:>11.1f} KB")
    
    print(f"\nTotal size: {total_size_mb:.2f} MB")
    
    print(f"\n{'='*70}")
    print("Next Steps:")
    print(f"{'='*70}")
    print(f"1. Go to https://www.kaggle.com/datasets")
    print(f"2. Click 'New Dataset'")
    print(f"3. Upload all files from: {output_dir}")
    print(f"4. Name the dataset: diabetes-ensemble-checkpoint-{checkpoint_num:03d}")
    print(f"5. In your Kaggle notebook, add this dataset as a data source")
    print(f"6. Set CHECKPOINT = '{checkpoint_num:03d}' in the inference notebook")
    print(f"7. Run inference to generate submission.csv")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
