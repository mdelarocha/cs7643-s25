# MRI Analysis for Alzheimer's Detection

This project implements machine learning models for Alzheimer's disease detection using MRI scans from the OASIS dataset.

## Directory Structure

```
.
├── data/                # Data directory
│   ├── raw/             # Raw MRI scans
│   └── processed/       # Preprocessed data
├── outputs/             # Model outputs and results
├── src/                 # Source code
│   ├── features/        # Feature extraction modules
│   ├── models/          # Model implementation
│   │   ├── baseline/    # Traditional ML models
│   │   └── deep_learning/ # Deep learning models
│   ├── pipelines/       # End-to-end pipelines
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks for exploration
├── team_members/        # Team member workspaces
├── tests/               # Unit tests
└── run_pipeline.py      # Main entry point
```

## Key Components

- **Data Loading**: Utilities for loading OASIS dataset metadata and MRI volumes
- **Preprocessing**: Comprehensive MRI preprocessing functions
- **Feature Extraction**: Statistical and textural feature extraction from MRI scans
- **Baseline Models**: Traditional machine learning models (Logistic Regression, SVM, Random Forest, etc.)
- **Visualization**: Tools for visualizing MRI data and model results
- **Pipelines**: End-to-end pipelines from raw data to predictions

## Usage

### Running the Full Pipeline

```bash
python run_pipeline.py
```

### Running the Baseline Pipeline

```bash
python -m src.pipelines.run_baseline --metadata_path data/oasis-cross-sectional.csv --data_dir data/raw --output_dir outputs
```

### Running with Specific Options

```bash
python -m src.pipelines.run_baseline --model_types logistic_regression random_forest --feature_types statistical
```

## Dependencies

The project requires the following dependencies:
- NumPy, Pandas, SciKit-Learn
- TensorFlow, PyTorch (for deep learning models)
- Nibabel (for NIfTI file handling)
- Matplotlib, Seaborn, Plotly (for visualization)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the OASIS (Open Access Series of Imaging Studies) dataset, which provides a cross-sectional collection of MRI scans of healthy individuals and those with Alzheimer's disease. The dataset includes:

- T1-weighted MRI scans
- Clinical data including demographics, Mini-Mental State Examination (MMSE) scores, and Clinical Dementia Rating (CDR) scores

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.
