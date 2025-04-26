# Alzheimer's Detection Project

This project aims to detect Alzheimer's disease using MRI data.

## Project Structure

```
project_root/
├── data/
│   ├── raw/          # Raw MRI scans (e.g., NIfTI, ANALYZE)
│   └── oasis-cross-sectional.csv # Metadata file
├── notebooks/        # Jupyter notebooks for exploration, visualization
├── outputs/          # Directory for pipeline outputs (models, results, plots)
│   └── baseline_run_TIMESTAMP/
│       ├── evaluation_metrics.json
│       ├── *.png                   # Plots (Confusion Matrix, Feature Importance, etc.)
│       ├── feature_importance.csv
│       └── models/                 # Saved model artifacts
│           └── *.pkl
├── src/
│   ├── features/       # Feature extraction (statistical, textural, etc.)
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── statistical.py
│   │   └── ...
│   ├── models/         # Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py # Abstract base class for models
│   │   └── baseline/     # Baseline model implementations (LR, RF, KNN, etc.)
│   │       ├── __init__.py
│   │       ├── logistic_regression.py
│   │       └── ...
│   ├── pipelines/      # End-to-end pipelines
│   │   ├── __init__.py
│   │   ├── baseline_pipeline.py # Core logic for baseline runs
│   │   └── run_baseline.py      # CLI script to execute the baseline pipeline
│   └── utils/          # Utility functions (dataloader, preprocessing, helpers)
│       ├── __init__.py
│       ├── dataloader.py
│       ├── preprocessing.py
│       └── ...
├── tests/            # Unit and integration tests
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## Baseline Models Pipeline

The `src/pipelines/run_baseline.py` script provides a command-line interface to run baseline machine learning models for Alzheimer's detection based on extracted features.

**Purpose:**
These baseline models (Logistic Regression, Random Forest, KNN, K-Means) serve as a benchmark. They are trained on statistical and/or textural features extracted from MRI data. The methodology focuses on providing a consistent framework and results for comparison against more complex Deep Learning approaches.

**Features:**
*   Modular design using a `BaseModel` interface (`src/models/base_model.py`).
*   Configurable feature extraction (`statistical`, `textural`).
*   Configurable model selection.
*   Subject-aware data splitting (default) to prevent data leakage.
*   Handles 3-class classification (non-demented=0, very mild=1, mild/moderate=2) by default.
*   Option to run binary classification (non-demented=0, demented>=0.5 -> 1) using `--combine_cdr` flag.
*   Generates evaluation metrics, confusion matrices, and feature importance plots.

**Usage:**

Navigate to the project root directory.

*   **Run default 3-class classification with LR and RF using statistical features:**
    ```bash
    python -m src.pipelines.run_baseline --model_types logistic_regression random_forest --feature_types statistical
    ```

*   **Run all baseline classifiers (excluding SVM) for 3-class problem:**
    ```bash
    python -m src.pipelines.run_baseline --model_types logistic_regression random_forest knn kmeans --feature_types statistical 
    ```

*   **Run binary classification (Non-demented vs. Demented) using Random Forest:**
    ```bash
    python -m src.pipelines.run_baseline --model_types random_forest --feature_types statistical --combine_cdr
    ```

*   **Specify different parameters (e.g., test size, feature selection):**
    ```bash
    python -m src.pipelines.run_baseline --model_types knn --feature_types statistical --test_size 0.25 --n_features 30
    ```

See `python -m src.pipelines.run_baseline --help` for all available options.

**Outputs:**
Results are saved in a timestamped subdirectory within `outputs/`, e.g., `outputs/baseline_run_YYYYMMDD_HHMMSS/`.
This includes:
*   `evaluation_metrics.json`: Detailed performance metrics for each model.
*   Confusion matrix plots (`*_confusion_matrix.png`).
*   Feature importance plots (`*_feature_importance.png`).
*   Overall selected feature importance (`feature_importance.csv`).
*   Saved model artifacts (e.g., `*.pkl` files for trained models and scalers).

## Next Steps

*   Develop and integrate Deep Learning models.
*   Compare DL model performance against these established baselines.
*   Further explore feature engineering and data augmentation techniques.
