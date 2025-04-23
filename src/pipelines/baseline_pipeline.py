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

from src.utils.dataloader import load_oasis_metadata, create_dataset_from_metadata
from src.utils.train_test_split import split_data_by_subject
from src.features.statistical import extract_statistical_features, extract_statistical_features_batch, dict_list_to_array
from src.features.textural import extract_textural_features, extract_textural_features_batch
from src.features.dimensionality_reduction import (
    feature_selection_pipeline, 
    pca_reduction, 
    get_combined_feature_importance,
    select_features_from_combined
)

from src.models.baseline.logistic_regression import (
    train_logistic_regression, 
    evaluate_model as evaluate_logistic_regression,
    standardize_features as standardize_features_lr,
    plot_feature_importance as plot_feature_importance_lr
)
from src.models.baseline.random_forest import (
    train_random_forest,
    evaluate_model as evaluate_random_forest,
    plot_feature_importance as plot_feature_importance_rf
)
from src.models.baseline.svm import (
    train_svm,
    evaluate_model as evaluate_svm,
    standardize_features as standardize_features_svm
)
from src.models.baseline.knn import (
    train_knn,
    evaluate_model as evaluate_knn,
    standardize_features as standardize_features_knn,
    find_optimal_k
)
from src.models.baseline.kmeans import (
    train_kmeans,
    assign_cluster_labels,
    evaluate_clustering_as_classifier,
    plot_clusters
)

logger = logging.getLogger(__name__)

def preprocess_features(X, feature_names=None):
    """
    Preprocess feature matrix to handle NaN and infinite values.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        feature_names (list, optional): Names of features for logging.
        
    Returns:
        numpy.ndarray: Preprocessed feature matrix.
    """
    if X is None or X.size == 0:
        logger.error("Cannot preprocess empty feature matrix")
        return None
    
    try:
        # Check for NaN or infinite values
        nan_mask = np.isnan(X)
        inf_mask = np.isinf(X)
        problem_mask = nan_mask | inf_mask
        
        if np.any(problem_mask):
            # Count problematic features and samples
            nan_counts_per_feature = np.sum(nan_mask, axis=0)
            inf_counts_per_feature = np.sum(inf_mask, axis=0)
            
            # Log information about problematic features
            total_problems = np.sum(problem_mask)
            logger.warning(f"Found {total_problems} problematic values ({np.sum(nan_mask)} NaN, {np.sum(inf_mask)} infinite)")
            
            # Log specific features with problems if feature names are provided
            if feature_names is not None:
                for i, (nan_count, inf_count) in enumerate(zip(nan_counts_per_feature, inf_counts_per_feature)):
                    if nan_count > 0 or inf_count > 0:
                        feature_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                        logger.warning(f"Feature '{feature_name}': {nan_count} NaN, {inf_count} infinite values")
            
            # Replace NaN and infinite values with feature means (computed from non-NaN values)
            X_clean = X.copy()
            
            for col in range(X.shape[1]):
                col_data = X[:, col]
                problem_indices = np.where(np.isnan(col_data) | np.isinf(col_data))[0]
                
                if len(problem_indices) < len(col_data):  # Only if some values are valid
                    # Compute mean from valid values
                    valid_indices = np.where(~(np.isnan(col_data) | np.isinf(col_data)))[0]
                    col_mean = np.mean(col_data[valid_indices])
                    
                    # Replace problematic values with mean
                    X_clean[problem_indices, col] = col_mean
                else:
                    # If all values are problematic, use zero
                    X_clean[:, col] = 0
                    logger.warning(f"Feature at index {col} has all NaN/infinite values; replacing with zeros")
            
            logger.info("Successfully replaced NaN and infinite values in feature matrix")
            return X_clean
        else:
            logger.info("No NaN or infinite values found in feature matrix")
            return X
    
    except Exception as e:
        logger.error(f"Error preprocessing features: {str(e)}")
        return X

def extract_features(volumes, feature_types=None):
    """
    Extract features from MRI volumes.
    
    Args:
        volumes (list): List of MRI volumes.
        feature_types (list, optional): List of feature types to extract.
        
    Returns:
        tuple: (X, feature_names) - Feature matrix and feature names.
    """
    if volumes is None or len(volumes) == 0:
        logger.error("No volumes provided for feature extraction")
        return None, None
    
    if feature_types is None:
        feature_types = ['statistical', 'textural']
    
    try:
        all_features = []
        all_feature_names = []
        
        # Extract features by type
        if 'statistical' in feature_types:
            logger.info("Extracting statistical features...")
            statistical_features = extract_statistical_features_batch(volumes)
            if statistical_features and len(statistical_features) > 0:
                X_stat, feat_names_stat = dict_list_to_array(statistical_features)
                if X_stat is not None and X_stat.size > 0:
                    all_features.append(X_stat)
                    all_feature_names.extend(feat_names_stat)
        
        if 'textural' in feature_types:
            logger.info("Extracting textural features...")
            textural_features = extract_textural_features_batch(volumes)
            if textural_features and len(textural_features) > 0:
                X_text, feat_names_text = dict_list_to_array(textural_features)
                if X_text is not None and X_text.size > 0:
                    all_features.append(X_text)
                    all_feature_names.extend(feat_names_text)
        
        # Combine features
        if len(all_features) > 0:
            X = np.hstack(all_features)
            logger.info(f"Extracted {X.shape[1]} features in total")
            return X, all_feature_names
        else:
            logger.error("No features were extracted")
            return None, None
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None, None

def select_best_features(X_train, y_train, X_test, feature_names, n_features=50, methods=None):
    """
    Select the best features using multiple selection methods.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Test features.
        feature_names (list): List of feature names.
        n_features (int, optional): Number of features to select.
        methods (list, optional): List of feature selection methods.
        
    Returns:
        tuple: (X_train_selected, X_test_selected, selected_feature_names, importance_df) - Selected features and names.
    """
    if X_train is None or y_train is None or X_test is None or not feature_names:
        logger.error("Cannot select features with None inputs")
        return None, None, None, None
    
    if methods is None:
        methods = ['f_classif', 'mutual_info', 'random_forest', 'logistic_l1']
    
    try:
        # Run feature selection pipeline
        results = feature_selection_pipeline(X_train, y_train, feature_names, methods, n_features)
        
        # If no results were obtained, return the original features
        if not results or 'methods' not in results or len(results['methods']) == 0:
            logger.warning("No feature selection results were obtained. Using all features.")
            return X_train, X_test, feature_names, None
        
        # Get combined feature importance
        importance_df = get_combined_feature_importance(results, top_k=n_features)
        
        # If importance_df is empty or None, return original features
        if importance_df is None or len(importance_df) == 0:
            logger.warning("No feature importance data was obtained. Using all features.")
            return X_train, X_test, feature_names, None
        
        # Get top features
        top_features = importance_df.index.tolist()[:min(n_features, len(importance_df))]
        
        # If no top features were found, return original features
        if not top_features:
            logger.warning("No top features were identified. Using all features.")
            return X_train, X_test, feature_names, importance_df
        
        # Select features
        X_train_selected, selected_indices = select_features_from_combined(X_train, feature_names, top_features)
        X_test_selected, _ = select_features_from_combined(X_test, feature_names, top_features)
        
        # If selection failed, return original features
        if X_train_selected is None or X_test_selected is None or not selected_indices:
            logger.warning("Feature selection failed. Using all features.")
            return X_train, X_test, feature_names, importance_df
        
        # Get names of selected features
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_feature_names)} best features")
        
        return X_train_selected, X_test_selected, selected_feature_names, importance_df
    
    except Exception as e:
        logger.error(f"Error selecting best features: {str(e)}")
        return X_train, X_test, feature_names, None

def train_models(X_train, y_train, X_val=None, y_val=None, model_types=None):
    """
    Train multiple baseline models.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray, optional): Validation features.
        y_val (numpy.ndarray, optional): Validation labels.
        model_types (list, optional): List of model types to train.
        
    Returns:
        dict: Dictionary of trained models.
    """
    if X_train is None or y_train is None:
        logger.error("Cannot train models with None inputs")
        return {}
    
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest', 'svm', 'knn', 'kmeans']
    
    models = {}
    
    try:
        # Train logistic regression
        if 'logistic_regression' in model_types:
            logger.info("Training logistic regression model...")
            X_train_scaled, X_val_scaled, _, _ = standardize_features_lr(X_train, X_val)
            model = train_logistic_regression(X_train_scaled, y_train, C=1.0, penalty='l2')
            if model is not None:
                models['logistic_regression'] = {
                    'model': model,
                    'scaler': 'standard',
                    'preprocessed': True
                }
        
        # Train random forest
        if 'random_forest' in model_types:
            logger.info("Training random forest model...")
            model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=None)
            if model is not None:
                models['random_forest'] = {
                    'model': model,
                    'scaler': None,
                    'preprocessed': False
                }
        
        # Train SVM
        if 'svm' in model_types:
            logger.info("Training SVM model...")
            X_train_scaled, X_val_scaled, _, _ = standardize_features_svm(X_train, X_val)
            model = train_svm(X_train_scaled, y_train, C=1.0, kernel='rbf')
            if model is not None:
                models['svm'] = {
                    'model': model,
                    'scaler': 'standard',
                    'preprocessed': True
                }
        
        # Train KNN
        if 'knn' in model_types:
            logger.info("Training KNN model...")
            X_train_scaled, X_val_scaled, _, _ = standardize_features_knn(X_train, X_val)
            
            # Find optimal k if validation set is provided
            if X_val is not None and y_val is not None:
                optimal_k, _, _ = find_optimal_k(X_train_scaled, y_train, X_val_scaled, y_val)
                if optimal_k is not None:
                    n_neighbors = optimal_k
                else:
                    n_neighbors = 5
            else:
                n_neighbors = 5
            
            model = train_knn(X_train_scaled, y_train, n_neighbors=n_neighbors)
            if model is not None:
                models['knn'] = {
                    'model': model,
                    'scaler': 'standard',
                    'preprocessed': True
                }
        
        # Train KMeans
        if 'kmeans' in model_types:
            logger.info("Training KMeans model...")
            X_train_scaled, X_val_scaled, _, _ = standardize_features_knn(X_train, X_val)
            
            # Determine number of clusters based on number of classes
            n_clusters = len(np.unique(y_train))
            
            model = train_kmeans(X_train_scaled, n_clusters=n_clusters)
            if model is not None:
                # Assign cluster labels using training data
                cluster_labels = model.predict(X_train_scaled)
                cluster_to_class_map, _ = assign_cluster_labels(cluster_labels, y_train)
                
                models['kmeans'] = {
                    'model': model,
                    'scaler': 'standard',
                    'preprocessed': True,
                    'cluster_to_class_map': cluster_to_class_map
                }
        
        logger.info(f"Trained {len(models)} models")
        return models
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return models

def evaluate_models(models, X_test, y_test, feature_names=None, output_dir=None):
    """
    Evaluate multiple trained models.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        feature_names (list, optional): Names of features.
        output_dir (str, optional): Directory to save evaluation results.
        
    Returns:
        dict: Dictionary of evaluation metrics for each model.
    """
    if not models or X_test is None or y_test is None:
        logger.error("Cannot evaluate models with None inputs")
        return {}
    
    results = {}
    
    try:
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each model
        for model_name, model_info in models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            model = model_info.get('model')
            if model is None:
                continue
            
            # Preprocess test data if needed
            X_test_processed = X_test
            if model_info.get('preprocessed', False):
                if model_info.get('scaler') == 'standard':
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.fit(X_test)
                    X_test_processed = scaler.transform(X_test)
            
            # Evaluate model based on type
            if model_name == 'logistic_regression':
                metrics = evaluate_logistic_regression(model, X_test_processed, y_test, feature_names)
                
                # Plot feature importance if feature names are provided
                if feature_names is not None and output_dir:
                    fig = plot_feature_importance_lr(model, feature_names)
                    if fig:
                        plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"))
                        plt.close(fig)
            
            elif model_name == 'random_forest':
                metrics = evaluate_random_forest(model, X_test_processed, y_test, feature_names)
                
                # Plot feature importance if feature names are provided
                if feature_names is not None and output_dir:
                    fig = plot_feature_importance_rf(model, feature_names)
                    if fig:
                        plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"))
                        plt.close(fig)
            
            elif model_name == 'svm':
                metrics = evaluate_svm(model, X_test_processed, y_test)
            
            elif model_name == 'knn':
                metrics = evaluate_knn(model, X_test_processed, y_test)
            
            elif model_name == 'kmeans':
                cluster_to_class_map = model_info.get('cluster_to_class_map')
                if cluster_to_class_map:
                    metrics = evaluate_clustering_as_classifier(model, X_test_processed, y_test, cluster_to_class_map)
                    
                    # Plot clusters if output directory is provided
                    if output_dir:
                        fig = plot_clusters(X_test_processed, model, true_labels=y_test)
                        if fig:
                            plt.savefig(os.path.join(output_dir, f"{model_name}_clusters.png"))
                            plt.close(fig)
                else:
                    metrics = {}
            
            else:
                metrics = {}
            
            # Save results
            results[model_name] = metrics
            
            # Save metrics to file if output directory is provided
            if output_dir and metrics:
                metrics_file = os.path.join(output_dir, f"{model_name}_metrics.json")
                with open(metrics_file, 'w') as f:
                    # Convert numpy types to Python types for JSON serialization
                    metrics_json = {}
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            metrics_json[key] = value.tolist()
                        elif isinstance(value, np.integer):
                            metrics_json[key] = int(value)
                        elif isinstance(value, np.floating):
                            metrics_json[key] = float(value)
                        else:
                            metrics_json[key] = value
                    
                    json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Evaluated {len(results)} models")
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating models: {str(e)}")
        return results

def save_models(models, output_dir):
    """
    Save trained models to files.
    
    Args:
        models (dict): Dictionary of trained models.
        output_dir (str): Directory to save models.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not models or not output_dir:
        logger.error("Cannot save models with None inputs")
        return False
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each model
        for model_name, model_info in models.items():
            model = model_info.get('model')
            if model is None:
                continue
            
            # Create model file path
            model_file = os.path.join(output_dir, f"{model_name}.pkl")
            
            # Save model
            with open(model_file, 'wb') as f:
                pickle.dump(model_info, f)
            
            logger.info(f"Saved {model_name} model to {model_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        return False

def load_models(input_dir):
    """
    Load trained models from files.
    
    Args:
        input_dir (str): Directory containing saved models.
        
    Returns:
        dict: Dictionary of loaded models.
    """
    if not input_dir:
        logger.error("Cannot load models with None input directory")
        return {}
    
    models = {}
    
    try:
        # Check if directory exists
        if not os.path.isdir(input_dir):
            logger.error(f"Directory {input_dir} does not exist")
            return {}
        
        # List model files
        model_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
        
        # Load each model
        for model_file in model_files:
            model_path = os.path.join(input_dir, model_file)
            model_name = os.path.splitext(model_file)[0]
            
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)
            
            models[model_name] = model_info
            logger.info(f"Loaded {model_name} model from {model_path}")
        
        return models
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return models

def run_baseline_pipeline(metadata_path, data_dir, model_types=None, feature_types=None, 
                         output_dir=None, test_size=0.2, val_size=0.1, n_features=50):
    """
    Run an end-to-end baseline model pipeline.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        data_dir (str): Directory containing MRI files.
        model_types (list, optional): List of model types to train.
        feature_types (list, optional): List of feature types to extract.
        output_dir (str, optional): Directory to save results.
        test_size (float, optional): Proportion of data for testing.
        val_size (float, optional): Proportion of data for validation.
        n_features (int, optional): Number of features to select.
        
    Returns:
        tuple: (models, results) - Trained models and evaluation results.
    """
    # Set default values
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest', 'svm', 'knn', 'kmeans']
    
    if feature_types is None:
        feature_types = ['statistical', 'textural']
    
    # Create output directory with timestamp
    if output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"baseline_pipeline_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load metadata
        logger.info("Loading metadata...")
        metadata = load_oasis_metadata(metadata_path)
        if metadata is None:
            return None, None
        
        # Split data
        logger.info("Splitting data...")
        train_df, val_df, test_df = split_data_by_subject(metadata, test_size=test_size, val_size=val_size)
        
        # Create datasets
        logger.info("Creating dataset from metadata...")
        X_train_raw, y_train = create_dataset_from_metadata(train_df, data_dir)
        X_val_raw, y_val = create_dataset_from_metadata(val_df, data_dir) if val_df is not None else (None, None)
        X_test_raw, y_test = create_dataset_from_metadata(test_df, data_dir)
        
        # Extract features
        logger.info("Extracting features...")
        X_train, feature_names = extract_features(X_train_raw, feature_types)
        X_val, _ = extract_features(X_val_raw, feature_types) if X_val_raw is not None else (None, None)
        X_test, _ = extract_features(X_test_raw, feature_types)
        
        if X_train is None or X_test is None:
            return None, None
            
        # Preprocess features to handle NaN values
        logger.info("Preprocessing features to handle NaN values...")
        X_train = preprocess_features(X_train, feature_names)
        X_val = preprocess_features(X_val) if X_val is not None else None
        X_test = preprocess_features(X_test)
        
        if X_train is None or X_test is None:
            logger.error("Feature preprocessing failed")
            return None, None
        
        # Select best features
        logger.info("Selecting best features...")
        X_train_selected, X_test_selected, selected_feature_names, importance_df = select_best_features(
            X_train, y_train, X_test, feature_names, n_features=n_features
        )
        
        if X_val is not None:
            # Apply same feature selection to validation set
            _, X_val_selected = select_features_from_combined(X_val, feature_names, selected_feature_names)
        else:
            X_val_selected = None
        
        # Save feature importance if output directory is provided
        if output_dir and importance_df is not None:
            importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"))
        
        # Train models
        logger.info("Training models...")
        models = train_models(X_train_selected, y_train, X_val_selected, y_val, model_types)
        
        # Evaluate models
        logger.info("Evaluating models...")
        results = evaluate_models(models, X_test_selected, y_test, selected_feature_names, output_dir)
        
        # Save models
        if output_dir:
            logger.info("Saving models...")
            save_models(models, os.path.join(output_dir, "models"))
        
        logger.info("Pipeline completed successfully")
        return models, results
    
    except Exception as e:
        logger.error(f"Error running baseline pipeline: {str(e)}")
        return None, None

# Create an additional wrapper function that provides a simpler interface
def run_simple_baseline(metadata_path, data_dir, output_dir, model_types=None, feature_types=None):
    """
    A simplified version of the baseline pipeline with reasonable defaults.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        data_dir (str): Directory containing MRI files.
        output_dir (str): Directory to save results.
        model_types (list, optional): List of model types to train.
        feature_types (list, optional): List of feature types to extract.
        
    Returns:
        tuple: (models, results) - Trained models and evaluation results.
    """
    # Set default values
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest']
    
    if feature_types is None:
        feature_types = ['statistical']
    
    # Run the pipeline with reasonable defaults
    return run_baseline_pipeline(
        metadata_path=metadata_path,
        data_dir=data_dir,
        model_types=model_types,
        feature_types=feature_types,
        output_dir=output_dir,
        test_size=0.2,
        val_size=0.1,
        n_features=20
    ) 