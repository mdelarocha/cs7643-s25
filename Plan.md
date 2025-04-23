# Implementation Plan: Baseline ML Models for Alzheimer's Detection

## PHASE 1: PROJECT CONTEXT

### Repository Overview
The project already has a good foundation with several key utilities implemented:

- **Data Loading**: `dataloader.py` with functions for loading OASIS metadata and MRI volumes
- **Preprocessing**: `preprocessing.py` with comprehensive MRI processing functions including:
  - Volume loading and normalization
  - Slice extraction
  - Dimension standardization
  - 2D/3D processing capabilities
- **Data Splitting**: `train_test_split.py` with subject-aware splitting functionality
- **Visualization**: `visualization.py` with functions to plot brain slices from different angles
- **GCS Integration**: `gcs_utils.py` for cloud storage access to the dataset

### Dependencies
The project already includes necessary libraries:
- Data processing: NumPy, Pandas, SciKit-Learn
- Deep learning: TensorFlow, PyTorch, TorchVision
- Neuroimaging: Nibabel for NIfTI file handling
- Visualization: Matplotlib, Seaborn, Plotly

### Existing Test Framework
Tests are in place for:
- Data standardization
- OASIS dataset processing
- Download functionality
- Utility functions

## PHASE 2: IMPLEMENTATION TASKS

### Task 1: Feature Extraction Module
- Create `src/features/statistical.py`:
  - Implement intensity-based feature extraction:
    - Mean, median, standard deviation, min/max
    - Histogram statistics (skewness, kurtosis)
    - Region-based statistics (e.g., brain quadrants)
  - Implement ROI-based feature extraction:
    - Hippocampus volumetric measures
    - Ventricle size measurements
    - Gray/white matter ratio

- Create `src/features/textural.py`:
  - Implement texture feature extraction:
    - Gray-level co-occurrence matrix (GLCM) features
    - Local binary patterns (LBP)
    - Gabor filter responses
    - Edge density measurements

- Create `src/features/dimensionality_reduction.py`:
  - Implement PCA for feature compression
  - Implement feature selection methods:
    - Filter methods (correlation, variance thresholds)
    - Wrapper methods (recursive feature elimination)
    - Embedded methods (L1 regularization)

### Task 2: Baseline Model Implementation
- Create `src/models/baseline/logistic_regression.py`:
  - Implement training and prediction functions
  - Hyperparameter tuning via grid search
  - Model evaluation and interpretation

- Create `src/models/baseline/knn.py`:
  - Implement K-Nearest Neighbors classifier
  - Distance metric selection
  - Hyperparameter optimization for K

- Create `src/models/baseline/kmeans.py`:
  - Implement K-Means clustering
  - Integration with classifier for prediction
  - Cluster visualization and interpretation

- Create `src/models/baseline/svm.py`:
  - Implement Support Vector Machine classifier
  - Kernel selection and parameter tuning
  - Scaling and preprocessing integration

- Create `src/models/baseline/random_forest.py`:
  - Implement Random Forest classifier
  - Feature importance extraction
  - Hyperparameter optimization

### Task 3: Model Evaluation Framework
- Extend `src/utils/evaluation.py`:
  - Implement confusion matrix visualization
  - Add ROC curve plotting
  - Implement cross-validation utilities
  - Create comprehensive performance metrics:
    - Accuracy, precision, recall, F1-score
    - Subject-level vs. scan-level performance
    - Class imbalance handling (weighted metrics)

### Task 4: Pipeline Integration
- Create `src/pipelines/baseline_pipeline.py`:
  - Implement end-to-end pipeline from raw data to predictions
  - Configuration options for feature selection
  - Model selection and ensemble methods
  - Results logging and visualization

### Task 5: LLM Integration
- Create `src/models/llm/feature_converter.py`:
  - Implement feature-to-text conversion utilities
  - Create templates for LLM prompting
  - Design output parsing functions

- Create `src/models/llm/inference.py`:
  - Implement interface for local LLM deployment
  - Add support for multiple model backends:
    - Mistral 7B
    - LLaMA 2
    - FLAN-T5
  - Implement quantization options for memory efficiency

- Create `src/models/llm/vision_models.py`:
  - Add support for multimodal vision-language models
  - Implement direct MRI scan processing with CLIP or similar
  - Create zero-shot classification interface

### Task 6: Notebooks and Documentation
- Create `notebooks/02_feature_extraction.ipynb`:
  - Demonstrate feature extraction process
  - Visualize feature distributions
  - Analyze feature importance

- Create `notebooks/03_baseline_models.ipynb`:
  - Implement and evaluate all baseline models
  - Compare performance across models
  - Visualize decision boundaries and predictions

- Create `notebooks/04_llm_evaluation.ipynb`:
  - Compare traditional ML with LLM approaches
  - Analyze complementary strengths
  - Explore ensemble methods combining both approaches

## Sample Function Signatures

```python
# src/features/statistical.py
def extract_statistical_features(volume):
    """Extract statistical features from an MRI volume"""
    features = {}
    features['mean_intensity'] = np.mean(volume)
    features['std_intensity'] = np.std(volume)
    # Additional statistical features...
    return features

# src/models/baseline/logistic_regression.py
def train_logistic_regression(X_train, y_train, C=1.0, penalty='l2', class_weight=None):
    """Train a logistic regression model with specified parameters"""
    model = LogisticRegression(C=C, penalty=penalty, class_weight=class_weight, max_iter=1000)
    model.fit(X_train, y_train)
    return model

# src/pipelines/baseline_pipeline.py
def run_baseline_pipeline(metadata_path, data_dir, model_type='logistic_regression', feature_types=['statistical'], output_dir=None):
    """Run an end-to-end baseline model pipeline"""
    # Load data
    metadata = load_oasis_metadata(metadata_path)
    
    # Split data
    train_df, val_df, test_df = split_data_by_subject(metadata)
    
    # Extract features
    X_train, y_train = extract_features_from_metadata(train_df, data_dir, feature_types)
    X_test, y_test = extract_features_from_metadata(test_df, data_dir, feature_types)
    
    # Train model
    model = train_model(X_train, y_train, model_type)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save results
    if output_dir:
        save_results(model, metrics, output_dir)
    
    return model, metrics
```
```