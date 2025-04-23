# References Directory

This directory contains reference scripts and code examples that have been incorporated into the main codebase in a more structured way.

## Contents

### Runners

The `runners/` directory contains the original pipeline scripts that have been refactored into the unified pipeline:

- `run_baseline.py` - Original baseline pipeline runner
- `run_baseline_train.py` - Script for training baseline models on pre-extracted features
- `run_baseline_train_balanced.py` - Balanced training script with class weighting
- `run_baseline_modified.py` - Modified version of the baseline pipeline 

### Download

The `download/` directory contains the original download scripts that have been consolidated into the `src/utils/download.py` module:

- `download_full_dataset.py` - Script to download the complete OASIS dataset
- `download_and_process_data.py` - Script to download and process MRI data
- `process_downloaded_data.py` - Script to process downloaded MRI data and extract features 