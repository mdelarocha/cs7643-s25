#!/usr/bin/env python
"""
Unified runner script for Alzheimer's detection baseline pipeline with improved
sampling techniques for class imbalance, particularly for CDR score 2.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipelines.baseline_pipeline import run_baseline_pipeline
from src.utils.dataloader import load_oasis_metadata, create_dataset_from_metadata
from src.utils.train_test_split import split_data_by_subject
from src.pipelines.balanced_sampling import (
    get_class_distribution, calculate_class_weights, 
    get_sampling_function, create_custom_sampling_strategy,
    apply_smote, apply_smote_tomek, apply_adasyn
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_logging(log_file=None):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"balanced_pipeline_{timestamp}.log")
    elif not os.path.isabs(log_file):
        # If a relative path is provided, make it relative to the logs directory
        log_file = os.path.join(logs_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Alzheimer's detection baseline pipeline with improved sampling for class imbalance"
    )
    
    parser.add_argument('--metadata_path', type=str, default="data/oasis-cross-sectional.csv",
                        help="Path to the metadata CSV file")
    parser.add_argument('--data_dir', type=str, default="data/raw",
                        help="Directory containing MRI files")
    parser.add_argument('--output_dir', type=str, default="outputs/balanced",
                        help="Directory to save results")
    parser.add_argument('--log_file', type=str, default=None,
                        help="Log file path (default: balanced_pipeline_TIMESTAMP.log)")
    
    # Pipeline options
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['logistic_regression', 'random_forest', 'svm', 'knn'],
                        choices=['logistic_regression', 'random_forest', 'svm', 'knn', 'kmeans'],
                        help="Model types to train")
    parser.add_argument('--feature_types', type=str, nargs='+',
                        default=['statistical', 'textural'],
                        choices=['statistical', 'textural'],
                        help="Feature types to extract")
    
    # Sampling options
    parser.add_argument('--sampling_method', type=str, default='smote',
                        choices=['none', 'oversample', 'undersample', 'combine', 
                                'smote', 'smote_tomek', 'adasyn'],
                        help="Sampling method to handle class imbalance")
    parser.add_argument('--focus_cdr2', action='store_true',
                        help="Apply special focus on CDR score 2 class (the most imbalanced)")
    parser.add_argument('--class_weight', type=str, default='balanced',
                        choices=['balanced', 'none', 'custom'],
                        help="Class weight strategy for model training")
    
    # Train/Test split options
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="Proportion of data for testing")
    parser.add_argument('--val_size', type=float, default=0.1,
                        help="Proportion of data for validation")
    parser.add_argument('--n_features', type=int, default=20,
                        help="Number of features to select")
    
    # Add flag for combining classes
    parser.add_argument('--combine_classes', action='store_true',
                        help="Combine CDR scores 1.0 and 2.0 into a single class (>=1)")
    
    return parser.parse_args()

def run_balanced_baseline_pipeline(metadata_path, data_dir, model_types=None, feature_types=None, 
                                  output_dir=None, test_size=0.2, val_size=0.1, n_features=50,
                                  sampling_method='smote', focus_cdr2=False, class_weight='balanced',
                                  combine_classes=False):
    """
    Run an end-to-end balanced baseline model pipeline with improved sampling.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        data_dir (str): Directory containing MRI files.
        model_types (list, optional): List of model types to train.
        feature_types (list, optional): List of feature types to extract.
        output_dir (str, optional): Directory to save results.
        test_size (float, optional): Proportion of data for testing.
        val_size (float, optional): Proportion of data for validation.
        n_features (int, optional): Number of features to select.
        sampling_method (str, optional): Method to use for handling class imbalance.
        focus_cdr2 (bool, optional): Whether to focus on balancing CDR score 2.
        class_weight (str, optional): Class weight strategy for model training.
        combine_classes (bool, optional): Whether to combine CDR 1.0 and 2.0 into class 2.
        
    Returns:
        tuple: (models, results) - Trained models and evaluation results.
    """
    # Set default values
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest', 'svm', 'knn']
    
    if feature_types is None:
        feature_types = ['statistical', 'textural']
    
    # Create output directory with timestamp
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"balanced_pipeline_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load metadata
        logger.info("Loading metadata...")
        metadata = load_oasis_metadata(metadata_path)
        if metadata is None:
            return None, None
        
        # Split data
        logger.info("Splitting data by subject...")
        train_df, val_df, test_df = split_data_by_subject(metadata, test_size=test_size, val_size=val_size)
        
        # Create datasets
        logger.info("Creating dataset from metadata...")
        # Load with original classes first
        X_train_raw, y_train_orig = create_dataset_from_metadata(train_df, data_dir, combine_cdr=False)
        X_val_raw, y_val_orig = create_dataset_from_metadata(val_df, data_dir, combine_cdr=False) if val_df is not None else (None, None)
        X_test_raw, y_test_orig = create_dataset_from_metadata(test_df, data_dir, combine_cdr=False)

        # Combine classes if requested
        if combine_classes:
            # The dataloader already maps {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
            # We need to apply our combination logic to these integer labels.
            logger.info("Combining classes based on integer labels: 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 2")
            # Define the mapping for the integer labels from the dataloader
            mapping = {0: 0, 1: 1, 2: 2, 3: 2}  # Map original 3 (from CDR 2.0) to combined class 2
            # Create a vectorized function for mapping
            mapping_func = np.vectorize(lambda x: mapping.get(x, x)) # Default to original value if not in map

            # Apply mapping to NumPy arrays (y_train_orig etc. now hold integer labels)
            y_train = mapping_func(y_train_orig).astype(int)
            y_val = mapping_func(y_val_orig).astype(int) if y_val_orig is not None else None
            y_test = mapping_func(y_test_orig).astype(int)
            
            # Debug log for unique values AFTER our combination
            logger.debug(f"Unique y_train values after combining: {np.unique(y_train)}")
            logger.debug(f"Unique y_val values after combining: {np.unique(y_val) if y_val is not None else 'None'}")
            logger.debug(f"Unique y_test values after combining: {np.unique(y_test)}")

            # Check if 'focus_cdr2' is enabled and disable it with a warning
            if focus_cdr2:
                logger.warning("Disabling 'focus_cdr2' because 'combine_classes' is enabled. Class 2 now represents CDR >= 1.0.")
                focus_cdr2 = False # Ensure focus_cdr2 is False if combine_classes is True

            # Add logging to check labels after mapping
            unique_labels, counts = np.unique(y_train, return_counts=True)
            logger.info(f"Labels in y_train after mapping: {dict(zip(unique_labels, counts))}")

        else:
            y_train = y_train_orig
            y_val = y_val_orig
            y_test = y_test_orig
        
        # Extract features using the original pipeline
        from src.pipelines.baseline_pipeline import extract_features, preprocess_features, select_best_features, train_models, evaluate_models, save_models
        
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
            from src.features.dimensionality_reduction import select_features_from_combined
            X_val_selected, _ = select_features_from_combined(X_val, feature_names, selected_feature_names)
        else:
            X_val_selected = None
        
        # Save feature importance if output directory is provided
        if output_dir and importance_df is not None:
            importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"))
        
        # Display original class distribution
        logger.info("Original class distribution:")
        get_class_distribution(y_train)
        
        # Apply sampling method to handle class imbalance
        logger.info(f"Applying {sampling_method} sampling technique...")
        
        # If focus on CDR 2, create a custom sampling strategy
        if focus_cdr2 and "2" in np.unique(y_train).astype(str):
            logger.info("Applying special focus to CDR score 2 class")
            
            # Convert to integer labels if needed
            if not isinstance(y_train[0], (int, np.integer)):
                # Check if we're dealing with CDR scores (0, 0.5, 1, 2)
                unique_vals = np.unique(y_train)
                if 0.5 in unique_vals or 1.0 in unique_vals or 2.0 in unique_vals:
                    logger.info("Converting CDR scores to integer labels")
                    
                    # Map CDR scores: 0 -> 0, 0.5 -> 1, 1.0 -> 2, 2.0 -> 3
                    y_train_int = np.array([0 if y == 0 else 
                                           (1 if y == 0.5 else 
                                           (2 if y == 1.0 else 3)) 
                                           for y in y_train])
                    
                    # Create mapping from integer to original labels for later
                    int_to_orig = {0: 0, 1: 0.5, 2: 1.0, 3: 2.0}
                    orig_to_int = {v: k for k, v in int_to_orig.items()}
                    
                    # Use integer labels for sampling
                    int_labels = True
                    y_train_orig = y_train.copy()
                    y_train = y_train_int
                else:
                    int_labels = False
            else:
                int_labels = False
            
            # Create a custom strategy focusing on CDR score 2
            # Find index of CDR score 2 after conversion
            cdr2_idx = 3 if int_labels else 2  # 3 in converted integers, 2 if already integers
            
            # Set target ratios with higher focus on CDR score 2
            target_ratios = {label: 0.5 for label in np.unique(y_train)}
            
            # If CDR 2 exists in the dataset
            if cdr2_idx in np.unique(y_train):
                # Set higher ratio for CDR 2 - make it at least 60% of majority class
                target_ratios[cdr2_idx] = 0.6
                
                # Create a custom sampling strategy
                sampling_strategy = create_custom_sampling_strategy(y_train, target_ratios)
            else:
                sampling_strategy = 'auto'
        else:
            sampling_strategy = 'auto'
            int_labels = False
        
        # Get the appropriate sampling function
        sampling_func = get_sampling_function(sampling_method)
        
        # Apply sampling
        X_train_sampled, y_train_sampled = sampling_func(
            X_train_selected, y_train, 
            sampling_strategy=sampling_strategy,
            random_state=42
        )
        
        # If we converted to integer labels, convert back
        if int_labels:
            logger.info("Converting integer labels back to original CDR scores")
            y_train_sampled = np.array([int_to_orig[y] for y in y_train_sampled])
        
        # Display new class distribution
        logger.info("After sampling, new class distribution:")
        get_class_distribution(y_train_sampled)
        
        # Set class weights for model training
        if class_weight == 'balanced':
            model_class_weights = 'balanced'
        elif class_weight == 'custom':
            model_class_weights = calculate_class_weights(y_train_sampled)
        else:
            model_class_weights = None
        
        # Train models with the sampled data
        logger.info(f"Training models with {sampling_method} sampled data...")
        models = train_models(X_train_sampled, y_train_sampled, X_val_selected, y_val, model_types)
        
        # Evaluate models
        logger.info("Evaluating models...")
        results = evaluate_models(models, X_test_selected, y_test, selected_feature_names, output_dir)
        
        # Save models
        if output_dir:
            logger.info("Saving models...")
            save_models(models, os.path.join(output_dir, "models"))
        
        # Save sampling information
        if output_dir:
            # Convert numpy types in distribution dicts to standard types for JSON
            def convert_dist_keys(dist_dict):
                if dist_dict is None:
                    return None
                return {str(k): v for k, v in dist_dict.items()}

            sampling_info = {
                'method': sampling_method,
                'focus_cdr2': focus_cdr2,
                'class_weight': class_weight,
                # Convert keys before storing
                'original_distribution': convert_dist_keys(get_class_distribution(y_train)),
                'sampled_distribution': convert_dist_keys(get_class_distribution(y_train_sampled))
            }
            
            import json
            # Ensure default=str is robust for other potential numpy types within values
            with open(os.path.join(output_dir, "sampling_info.json"), 'w') as f:
                json.dump(sampling_info, f, indent=4, default=str)
        
        logger.info("Balanced pipeline completed successfully")
        return models, results
    
    except Exception as e:
        logger.error(f"Error running balanced baseline pipeline: {str(e)}")
        return None, None

def main():
    """Run the balanced baseline pipeline with provided options."""
    args = parse_args()
    logger = setup_logging(args.log_file)
    
    logger.info("Starting balanced baseline pipeline")
    logger.info(f"Metadata: {args.metadata_path}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Sampling method: {args.sampling_method}")
    logger.info(f"Focus on CDR score 2: {args.focus_cdr2}")
    logger.info(f"Combine Classes (>=1): {args.combine_classes}")
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run balanced pipeline
    try:
        models, results = run_balanced_baseline_pipeline(
            metadata_path=args.metadata_path,
            data_dir=args.data_dir,
            model_types=args.model_types,
            feature_types=args.feature_types,
            output_dir=args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            n_features=args.n_features,
            sampling_method=args.sampling_method,
            focus_cdr2=args.focus_cdr2,
            class_weight=args.class_weight,
            combine_classes=args.combine_classes
        )
        
        if models is not None and results is not None:
            logger.info("Pipeline completed successfully")
            return 0
        else:
            logger.error("Pipeline failed")
            return 1
    
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 