# Balanced Sampling for Alzheimer's Detection

This module provides improved sampling techniques to handle class imbalance in the Alzheimer's detection dataset, with a particular focus on the under-represented CDR score 2 class.

## Overview

The OASIS Alzheimer's dataset has significant class imbalance, particularly for patients with CDR score 2 (representing moderate dementia). This imbalance can lead to biased models that perform poorly on minority classes. This pipeline implements several advanced sampling techniques to address this issue.

## Sampling Techniques

The following sampling techniques are available:

1. **Basic Resampling**:
   - `oversample`: Randomly duplicate examples from minority classes
   - `undersample`: Randomly remove examples from majority classes
   - `combine`: Combination of over and under sampling

2. **Advanced Techniques** (requires `imbalanced-learn` package):
   - `smote`: Synthetic Minority Over-sampling Technique - creates synthetic samples
   - `smote_tomek`: Combines SMOTE with Tomek links undersampling
   - `adasyn`: Adaptive Synthetic Sampling - creates more synthetic samples for harder-to-learn examples

3. **Special CDR Score 2 Focus**:
   - The `--focus_cdr2` flag enables special treatment for CDR score 2 class

## Usage

Run the balanced pipeline using the command:

```bash
python -m src.pipelines.run_balanced_pipeline --sampling_method smote --focus_cdr2
```

### Command-line Options

```
--metadata_path      Path to the metadata CSV file
--data_dir           Directory containing MRI files
--output_dir         Directory to save results
--log_file           Log file path
--model_types        Model types to train (logistic_regression, random_forest, svm, knn, kmeans)
--feature_types      Feature types to extract (statistical, textural)
--sampling_method    Sampling method (none, oversample, undersample, combine, smote, smote_tomek, adasyn)
--focus_cdr2         Enable special focus on CDR score 2 class
--class_weight       Class weight strategy (balanced, none, custom)
--test_size          Proportion of data for testing
--val_size           Proportion of data for validation
--n_features         Number of features to select
```

## Example Commands

1. Run with SMOTE and special focus on CDR score 2:
```bash
python -m src.pipelines.run_balanced_pipeline --sampling_method smote --focus_cdr2
```

2. Use ADASYN with custom class weights:
```bash
python -m src.pipelines.run_balanced_pipeline --sampling_method adasyn --class_weight custom
```

3. Basic oversampling with statistical features only:
```bash
python -m src.pipelines.run_balanced_pipeline --sampling_method oversample --feature_types statistical
```

## Class Weights vs. Sampling

This pipeline provides two complementary approaches to handling class imbalance:

1. **Sampling techniques**: Change the distribution of the training data by creating a more balanced dataset
2. **Class weights**: Adjust the importance of different classes during model training

Using both approaches together (e.g., SMOTE with class_weight='balanced') can sometimes lead to over-correction. The `--class_weight none` option can be used with sampling techniques to avoid this.

## Requirements

Install the required packages:

```bash
pip install imbalanced-learn
```

Or update your environment using the project's requirements.txt:

```bash
pip install -r requirements.txt
``` 