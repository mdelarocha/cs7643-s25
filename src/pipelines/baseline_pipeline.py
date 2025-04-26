"""
Comprehensive baseline pipeline for Alzheimer's detection.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
import json
from typing import List, Dict, Optional, Tuple, Union, Any, Type

from src.utils.dataloader import load_oasis_metadata, create_dataset_from_metadata
from src.utils.train_test_split import split_data_by_subject, split_data_by_range, create_stratified_split
from src.features.statistical import extract_statistical_features_batch, dict_list_to_array
from src.features.textural import extract_textural_features_batch
from src.features.core import extract_features
from src.features.dimensionality_reduction import select_top_k_features_combined
from src.utils.preprocessing import preprocess_features
from src.utils.helpers import convert_numpy_types
from src.utils.io import save_pipeline_artifacts, load_pipeline_artifacts

from src.models.base_model import BaseModel
from src.models.baseline.logistic_regression import LogisticRegressionModel
from src.models.baseline.random_forest import RandomForestModel
from src.models.baseline.svm import SVMModel
from src.models.baseline.knn import KNNModel
from src.models.baseline.kmeans import KMeansModel

logger = logging.getLogger(__name__)

# --- Model Registry --- 
# Maps string names to model classes for easier instantiation
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    'logistic_regression': LogisticRegressionModel,
    'random_forest': RandomForestModel,
    'svm': SVMModel,
    'knn': KNNModel,
    'kmeans': KMeansModel
}

def select_best_features(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                         feature_names: List[str], n_features: int = 50, methods: Optional[List[str]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[pd.DataFrame]]:
    """
    Select the best features using multiple selection methods. Returns original arrays if selection fails.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Test features.
        feature_names (list): List of feature names.
        n_features (int): Number of features to select.
        methods (list, optional): List of feature selection methods. Defaults ['f_classif', 'random_forest'].
        
    Returns:
        tuple: (X_train_selected, X_test_selected, selected_feature_names, importance_df) - Selected features and names.
               Returns original X_train, X_test, feature_names, None if selection fails.
    """
    if X_train is None or y_train is None or X_test is None or not feature_names:
        logger.error("Cannot select features with None inputs")
        return X_train, X_test, feature_names, None
    
    if n_features <= 0 or n_features >= X_train.shape[1]:
        logger.info(f"Skipping feature selection: n_features ({n_features}) is invalid or >= total features ({X_train.shape[1]})")
        return X_train, X_test, feature_names, None
    
    if methods is None:
        methods = ['f_classif', 'random_forest']
    
    try:
        logger.info(f"Running feature selection pipeline with methods: {methods}, selecting top {n_features} features.")
        results = feature_selection_pipeline(X_train, y_train, feature_names, methods, n_features)
        
        if not results or 'methods' not in results or not results['methods']:
            logger.warning("Feature selection pipeline did not return valid results. Using all features.")
            return X_train, X_test, feature_names, None
        
        importance_df = get_combined_feature_importance(results, top_k=n_features)
        
        if importance_df is None or importance_df.empty:
            logger.warning("Could not obtain combined feature importance. Using all features.")
            return X_train, X_test, feature_names, None
        
        top_features = importance_df.index.tolist()[:n_features]
        
        if not top_features:
            logger.warning("No top features identified from importance data. Using all features.")
            return X_train, X_test, feature_names, importance_df
        
        X_train_selected, selected_indices = select_features_from_combined(X_train, feature_names, top_features)
        X_test_selected, _ = select_features_from_combined(X_test, feature_names, top_features)
        
        if X_train_selected is None or X_test_selected is None:
            logger.warning("Feature selection failed during application (select_features_from_combined). Using all features.")
            return X_train, X_test, feature_names, importance_df
        
        # Ensure selected features have correct shape (2D)
        if X_train_selected.ndim == 1: X_train_selected = X_train_selected.reshape(-1, 1)
        if X_test_selected.ndim == 1: X_test_selected = X_test_selected.reshape(-1, 1)
        
        # Check number of selected features matches expected
        if X_train_selected.shape[1] != len(top_features) or X_test_selected.shape[1] != len(top_features):
            logger.warning(f"Selected feature count mismatch. Expected {len(top_features)}, got Train:{X_train_selected.shape[1]}, Test:{X_test_selected.shape[1]}. Using all features.")
            return X_train, X_test, feature_names, importance_df
        
        selected_feature_names = top_features
        
        logger.info(f"Successfully selected {len(selected_feature_names)} features.")
        return X_train_selected, X_test_selected, selected_feature_names, importance_df
    
    except Exception as e:
        logger.exception(f"Error selecting best features: {e}")
        return X_train, X_test, feature_names, None

def train_models(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                 model_types: Optional[List[str]] = None, model_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, BaseModel]]:
    """
    Train multiple models on the same data using the BaseModel interface.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        model_types: List of model type names (keys in MODEL_REGISTRY).
        model_params: Optional dictionary mapping model_type to its specific parameters.
        
    Returns:
        Dictionary of trained models {model_type: model_instance}. Returns None if fatal error.
    """
    if X_train is None or y_train is None:
        logger.error("Cannot train models with None training data")
        return None
    
    if model_types is None:
        model_types = list(MODEL_REGISTRY.keys()) # Default to all registered models
    
    trained_models = {}
    
    for model_type in model_types:
        if model_type not in MODEL_REGISTRY:
            logger.warning(f"Model type '{model_type}' not found in registry. Skipping.")
            continue

        logger.info(f"Training {model_type} model...")
        ModelClass = MODEL_REGISTRY[model_type]
        
        # Get specific parameters for this model type, if provided
        params = model_params.get(model_type, {}) if model_params else {}

        try:
            model_instance = ModelClass(**params)
            # Pass validation data if available
            model_instance.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            trained_models[model_type] = model_instance
            logger.info(f"Successfully trained {model_type}.")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}", exc_info=True) # Log traceback

    if not trained_models:
         logger.error("No models were successfully trained.")
         return None
         
    logger.info(f"Attempted training for {len(model_types)} model types. Successfully trained: {list(trained_models.keys())}")
    # The returned dictionary now contains the *instances* of the BaseModel subclasses
    return trained_models # Return dict of model instances directly

def evaluate_models(trained_models: Dict[str, BaseModel], X_test: np.ndarray, y_test: np.ndarray,
                    feature_names: Optional[List[str]] = None, output_dir: Optional[str] = None) -> Optional[Dict[str, Dict]]:
    """
    Evaluate all trained models (BaseModel instances) on test data.
    
    Args:
        trained_models: Dictionary containing trained model instances {model_type: model_instance}.
        X_test: Test features.
        y_test: Test labels.
        feature_names: Names of features for plotting importance (if applicable).
        output_dir: Directory to save evaluation results and plots.
        
    Returns:
        Dictionary of evaluation results {model_type: metrics_dict}. Returns None if fatal error.
    """
    if not trained_models:
        logger.error("No valid models provided for evaluation.")
        return None
    if X_test is None or y_test is None:
        logger.error("Cannot evaluate models with None test data.")
        return None
    
    results = {}

    try:
        for model_type, model_instance in trained_models.items():
            logger.info(f"Evaluating {model_type} model...")
            metrics = None

            try:
                # Use the evaluate method of the BaseModel instance
                # The model instance itself handles any necessary preprocessing like scaling
                metrics = model_instance.evaluate(X_test, y_test)

                # 2. Call plotting method if it exists and output_dir is provided
                if output_dir and hasattr(model_instance, 'plot_feature_importance') and callable(getattr(model_instance, 'plot_feature_importance')):
                    if feature_names:
                        try:
                            plot_path = os.path.join(output_dir, f"{model_type}_feature_importance.png")
                            model_instance.plot_feature_importance(feature_names, plot_path)
                        except Exception as plot_err:
                            logger.error(f"Failed to plot feature importance for {model_type}: {plot_err}")
                    else:
                        logger.warning(f"Cannot plot feature importance for {model_type}: feature_names not provided.")

                # 3. Call confusion matrix plotting method if it exists (should exist for classifiers)
                if output_dir and hasattr(model_instance, 'plot_confusion_matrix') and callable(getattr(model_instance, 'plot_confusion_matrix')):
                    # Define class labels based on your dataset (modify as needed)
                    # Example assumes binary classification 0: Non-demented, 1: Demented (if combine_cdr=True)
                    # Or 0: Non, 1: Very Mild, 2: Mild (if combine_cdr=False)
                    # This part might need adjustment based on the actual labels used.
                    # Let's try fetching from the model if possible, otherwise default
                    class_labels = None 
                    if hasattr(model_instance.model, 'classes_'):
                        try: 
                            # Attempt to map numeric classes to meaningful names
                            class_mapping = {0: 'nondemented', 1: 'very mild dementia', 2: 'mild dementia'} # Adjust if combine_cdr changes this
                            class_labels = [class_mapping.get(c, str(c)) for c in model_instance.model.classes_]
                        except Exception as label_err:
                            logger.warning(f"Could not automatically map class labels for {model_type}: {label_err}")

                    try:
                        cm_plot_path = os.path.join(output_dir, f"{model_type}_confusion_matrix.png")
                        model_instance.plot_confusion_matrix(X_test, y_test, cm_plot_path, class_labels=class_labels)
                    except Exception as plot_err:
                        logger.error(f"Failed to plot confusion matrix for {model_type}: {plot_err}")

                # 4. Add other potential plots here (e.g., plot_clusters for KMeans)
                if output_dir and model_type == 'kmeans' and hasattr(model_instance, 'plot_clusters') and callable(getattr(model_instance, 'plot_clusters')):
                    try:
                        cluster_plot_path = os.path.join(output_dir, f"{model_type}_clusters.png")
                        # Pass y_test to plot_clusters if it can use true labels for coloring
                        model_instance.plot_clusters(X_test, y_test, cluster_plot_path)
                    except Exception as plot_err:
                        logger.error(f"Failed to plot clusters for {model_type}: {plot_err}")

                # Store and log metrics
                if metrics:
                    results[model_type] = metrics
                    logger.info(f"--- {model_type} Evaluation Metrics ---")
                    for name, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            logger.info(f"  {name}: \\n{value}")
                        else:
                            try:
                                # Use .4f for floats, otherwise just print the value
                                logger.info(f"  {name}: {value:.4f}" if isinstance(value, (float, np.floating)) else f"  {name}: {value}")
                            except TypeError: # Handle cases where value is not easily formatted (e.g., complex objects)
                                logger.info(f"  {name}: {value}")
                    logger.info("-" * (len(model_type) + 28)) # Separator line
                else:
                    logger.warning(f"Evaluation failed or returned no metrics for {model_type}.")

            except Exception as eval_exc:
                logger.error(f"Error evaluating model {model_type}: {eval_exc}", exc_info=True)
                results[model_type] = {"error": str(eval_exc)}

        # Save Overall Metrics
        if output_dir and results:
            # Filter out error entries before saving
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                metrics_path_json = os.path.join(output_dir, "evaluation_metrics.json")
                try:
                    with open(metrics_path_json, 'w') as f:
                        json.dump(valid_results, f, indent=4, default=convert_numpy_types)
                    logger.info(f"Saved combined evaluation metrics to {metrics_path_json}")
                except Exception as json_exc:
                    logger.error(f"Error saving evaluation metrics to JSON: {json_exc}")
            else:
                logger.warning("No valid evaluation results to save.")
        
        return results # Return the dictionary including any error entries
    
    except Exception as e:
        logger.exception(f"Critical error during model evaluation process: {e}")
        return None

def run_baseline_pipeline(
    metadata_path: str,
    data_dir: str,
    output_dir: str,
    model_types: List[str],
    feature_types: List[str],
    split_strategy: str = 'subject',
    test_size: float = 0.2,
    val_size: float = 0.1,
    train_range: Optional[Tuple[int, int]] = None,
    n_features: Optional[int] = 50,
    combine_cdr: bool = True,
    random_state: int = 42,
    model_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Dict]]]:
    """
    Runs the consolidated end-to-end baseline model pipeline.
    
    Args:
        metadata_path: Path to metadata CSV.
        data_dir: Directory containing MRI files.
        output_dir: Base directory to save results (timestamped subdir created).
        model_types: List of model types to train.
        feature_types: List of feature types to extract.
        split_strategy: Method for splitting data ('subject', 'stratified', 'range'). Defaults to 'subject'.
        test_size: Proportion for test set (used if strategy is not 'range').
        val_size: Proportion for validation set (used if strategy is not 'range').
        train_range: Tuple (start_idx, end_idx) for 'range' split strategy.
        n_features: Number of features to select (<=0 or None disables selection).
        combine_cdr: If True, performs binary classification (0 vs >=0.5) using 'CDR_Combined' column.
                     If False (default behavior implicitly if flag not set in run_baseline.py),
                     performs 3-class classification (0, 1, 2) using 'CDR' column from dataloader.
        random_state: Seed for random operations.
        model_params: Optional dictionary mapping model_type name to its specific configuration dict.
        
    Returns:
        tuple: (trained_models, evaluation_results)
               trained_models: Dictionary {model_type: trained_model_instance}.
               evaluation_results: Dictionary {model_type: metrics_dict}.
    """
    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"baseline_run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    logger.info(f"Starting baseline pipeline run. Output directory: {run_output_dir}")
    logger.info(f"Run parameters: models={model_types}, features={feature_types}, split={split_strategy}, "
                f"test_size={test_size}, val_size={val_size}, train_range={train_range}, "
                f"n_features={n_features}, combine_cdr={combine_cdr}, split_strategy={split_strategy}")

    trained_data = None
    results = None

    try:
        logger.info(f"Loading metadata from: {metadata_path}")
        metadata_df = load_oasis_metadata(metadata_path, combine_cdr=combine_cdr)
        if metadata_df is None or metadata_df.empty:
            raise ValueError("Failed to load metadata or metadata is empty.")
        logger.info(f"Metadata loaded successfully. Shape: {metadata_df.shape}")

        label_column = 'CDR_Combined' if combine_cdr else 'CDR'
        if label_column not in metadata_df.columns:
            raise ValueError(f"Required label column '{label_column}' not found in metadata after loading.")

        logger.info(f"Splitting data using strategy: {split_strategy}")
        train_df, val_df, test_df = None, None, None

        if split_strategy == 'range':
            if train_range is None or len(train_range) != 2:
                raise ValueError("train_range (start, end) must be provided for 'range' split strategy.")
            train_start_idx, train_end_idx = train_range
            train_df, val_df, test_df = split_data_by_range(
                metadata_df, train_start_idx, train_end_idx, test_size, val_size
            )
            logger.info(f"Range split: Train indices {train_start_idx}-{train_end_idx}, Test size {test_size}, Val size {val_size}")

        elif split_strategy == 'stratified':
            logger.info(f"Attempting stratified split by '{label_column}'. Test size: {test_size}, Val size: {val_size}")
            min_samples_per_class = metadata_df[label_column].value_counts().min()
            required_samples = 2
            if val_size > 0 and (1-test_size-val_size) > 0 : required_samples = 3
            elif val_size > 0 or test_size > 0: required_samples = 2

            if min_samples_per_class < required_samples:
                logger.warning(f"Minimum samples per class ({min_samples_per_class}) is less than required ({required_samples}) for stratification on '{label_column}'. Falling back to 'subject' split.")
                split_strategy = 'subject'
            else:
                try:
                    train_df, val_df, test_df = create_stratified_split(
                        metadata_df, label_column, test_size, val_size, random_state=random_state
                    )
                    logger.info("Stratified split successful.")
                except Exception as strat_err:
                    logger.warning(f"Stratified split failed ({strat_err}). Falling back to 'subject' split.")
                    split_strategy = 'subject'

        if split_strategy == 'subject':
            logger.info(f"Splitting data by subject (default or fallback). Test size: {test_size}, Val size: {val_size}")
            train_df, val_df, test_df = split_data_by_subject(
                metadata_df, test_size, val_size, random_state=random_state
            )

        if train_df is None or train_df.empty or test_df is None or test_df.empty:
            raise ValueError("Data splitting resulted in empty train or test set.")
        logger.info(f"Data split sizes: Train={len(train_df)}, Validation={len(val_df) if val_df is not None else 0}, Test={len(test_df)}")

        logger.info("Creating datasets (loading MRI volumes)...")
        X_train_raw, y_train = create_dataset_from_metadata(train_df, data_dir, label_col=label_column)
        X_val_raw, y_val = create_dataset_from_metadata(val_df, data_dir, label_col=label_column) if val_df is not None else (None, None)
        X_test_raw, y_test = create_dataset_from_metadata(test_df, data_dir, label_col=label_column)

        if X_train_raw is None or y_train is None or X_test_raw is None or y_test is None:
            raise ValueError("Failed to create datasets (load MRI volumes). Check paths and metadata linkage.")
        logger.info("MRI volumes loaded.")

        logger.info(f"Extracting features: {feature_types}")
        X_train_feat, feature_names = extract_features(X_train_raw, feature_types)
        X_val_feat, _ = extract_features(X_val_raw, feature_types) if X_val_raw else (None, None)
        X_test_feat, _ = extract_features(X_test_raw, feature_types)

        if X_train_feat is None or feature_names is None or X_test_feat is None:
            raise ValueError("Feature extraction failed for train or test set.")
        logger.info(f"Feature extraction complete. Shape: {X_train_feat.shape}")

        logger.info("Preprocessing features (handling NaN/Inf)...")
        X_train_proc = preprocess_features(X_train_feat, feature_names)
        X_val_proc = preprocess_features(X_val_feat, feature_names) if X_val_feat is not None else None
        X_test_proc = preprocess_features(X_test_feat, feature_names)

        if X_train_proc is None or X_test_proc is None:
            raise ValueError("Feature preprocessing (NaN/Inf handling) failed.")
        logger.info("Feature preprocessing complete.")

        X_train_final, X_test_final = X_train_proc, X_test_proc
        X_val_final = X_val_proc
        current_feature_names = feature_names
        importance_df = None

        if n_features is not None and n_features > 0 and n_features < X_train_proc.shape[1]:
            logger.info(f"Selecting top {n_features} features...")
            X_train_final, X_test_final, X_val_final, current_feature_names, importance_df = select_top_k_features_combined(
                X_train_proc, y_train, X_test_proc, X_val_proc, feature_names, n_features=n_features
            )

            if importance_df is not None and not importance_df.empty:
                importance_path = os.path.join(run_output_dir, "feature_importance.csv")
                try:
                    importance_df.to_csv(importance_path)
                    logger.info(f"Saved feature importance scores to {importance_path}")
                except Exception as imp_save_err:
                    logger.error(f"Failed to save feature importance: {imp_save_err}")
        else:
            logger.info("Skipping feature selection (n_features invalid or not specified). Using all preprocessed features.")
            # Ensure X_train_final, etc., are still assigned the processed versions
            X_train_final, X_test_final = X_train_proc, X_test_proc
            X_val_final = X_val_proc
            current_feature_names = feature_names

        logger.info(f"Training models: {model_types}...")
        # Pass model parameters dictionary
        trained_data = train_models(X_train_final, y_train, X_val_final, y_val, model_types, model_params)
        if trained_data is None or not trained_data:
            raise ValueError("Model training failed or returned no models.")
        logger.info(f"Models trained: {list(trained_data.keys())}")

        logger.info("Evaluating models on the test set...")
        # Pass the final selected/processed test features and the correct feature names
        results = evaluate_models(trained_data, X_test_final, y_test, current_feature_names, run_output_dir)
        if results is None:
            # Evaluate_models logs errors internally, main pipeline should still proceed if possible
            logger.warning("Model evaluation process finished, but might have encountered errors (check logs). Results might be partial or None.")
        logger.info("Model evaluation complete.")

        logger.info("Saving trained models and scalers...")
        save_pipeline_artifacts(trained_data, run_output_dir)
        logger.info("Pipeline artifacts saved.")

        logger.info("Baseline pipeline run completed successfully.")

    except Exception as e:
        logger.exception(f"Baseline pipeline run failed: {e}")
        trained_data = None
        results = None

    finally:
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        logger.info(f"Pipeline finished in {duration}.")
        return trained_data, results 