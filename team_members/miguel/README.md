# Miguel's Workspace

This directory contains reference files, code explorations, and development notes for Miguel's work on the project.

## Contents

- `scratch/`: Experimental scripts and code snippets
- `notes/`: Project notes and documentation
- `references/`: Useful reference materials and resources

Feel free to use this space for any exploratory work or experiments without affecting the main codebase.

# Miguel's Contribution: Baseline Models Pipeline

This directory contains the source code developed by Miguel for the Alzheimer's Detection Project, focusing on establishing baseline model performance.

## Overview

The code provides a framework for extracting features from MRI data and training/evaluating standard machine learning models.

## Directory Structure (`src`)

```
src/
├── features/       # Feature extraction (statistical, textural, etc.)
│   ├── __init__.py
│   ├── core.py
│   ├── statistical.py
│   └── ...
├── models/         # Model implementations
│   ├── __init__.py
│   ├── base_model.py # Abstract base class for models
│   └── baseline/     # Baseline model implementations (LR, RF, KNN, etc.)
│       ├── __init__.py
│       ├── logistic_regression.py
│       └── ...
├── pipelines/      # End-to-end pipelines
│   ├── __init__.py
│   ├── baseline_pipeline.py # Core logic for baseline runs
│   └── run_baseline.py      # CLI script to execute the baseline pipeline
└── utils/          # Utility functions (dataloader, preprocessing, helpers)
    ├── __init__.py
    ├── dataloader.py
    ├── preprocessing.py
    └── ...
```

## Baseline Models Pipeline Script

The `src/pipelines/run_baseline.py` script (relative to this directory) provides a command-line interface to run baseline machine learning models.

**Purpose:**
These baseline models (Logistic Regression, Random Forest, KNN, K-Means) serve as a benchmark. They are trained on statistical and/or textural features extracted from MRI data. The methodology focuses on providing a consistent framework and results for comparison against more complex Deep Learning approaches.

**Features:**
*   Modular design using a `BaseModel` interface (`src/models/base_model.py`).
*   Configurable feature extraction (`statistical`, `textural`).
*   Configurable model selection.
*   Subject-aware data splitting (default) to prevent data leakage.
*   Handles 3-class classification (non-demented=0, very mild=1, mild/moderate=2) by default.
*   Option to run binary classification (non-demented=0, demented>=0.5 -> 1) using `--combine_cdr` flag.
*   Generates evaluation metrics, confusion matrices, ROC/PR curves, and feature importance plots.

**Usage:**

Navigate to the **project root directory** (the one containing `team_members`).

*   **Run default 3-class classification with LR and RF using statistical features:**
    ```bash
    python -m team_members.miguel.src.pipelines.run_baseline --model_types logistic_regression random_forest --feature_types statistical
    ```

*   **Run all baseline classifiers (excluding SVM) for 3-class problem:**
    ```bash
    python -m team_members.miguel.src.pipelines.run_baseline --model_types logistic_regression random_forest knn kmeans --feature_types statistical
    ```

*   **Run binary classification (Non-demented vs. Demented) using Random Forest:**
    ```bash
    python -m team_members.miguel.src.pipelines.run_baseline --model_types random_forest --feature_types statistical --combine_cdr
    ```

*   **Specify different parameters (e.g., test size, feature selection):**
    ```bash
    python -m team_members.miguel.src.pipelines.run_baseline --model_types knn --feature_types statistical --test_size 0.25 --n_features 30
    ```

See `python -m team_members.miguel.src.pipelines.run_baseline --help` for all available options.

**Outputs:**
Results are saved in a timestamped subdirectory within the main project's `outputs/` directory, e.g., `outputs/baseline_run_YYYYMMDD_HHMMSS/`.
This includes:
*   `evaluation_metrics.json`: Detailed performance metrics for each model.
*   Confusion matrix plots (`*_confusion_matrix.png`).
*   ROC curve plots (`*_roc_curve.png`).
*   Precision-Recall curve plots (`*_precision_recall_curve.png`).
*   Feature importance plots (`*_feature_importance.png`).
*   Overall selected feature importance (`feature_importance.csv`).
*   Saved model artifacts (e.g., `*.pkl` files for trained models and scalers).

## Setup

Ensure you have Python installed. Install dependencies from the main project `requirements.txt`:

```bash
# From the project root directory
pip install -r requirements.txt
```
