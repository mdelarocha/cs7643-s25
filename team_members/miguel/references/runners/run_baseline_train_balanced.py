#!/usr/bin/env python
"""
Script to train balanced baseline models using pre-extracted features with class imbalance handling.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from datetime import datetime
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

from src.models.baseline.logistic_regression import (
    train_logistic_regression, 
    evaluate_model as evaluate_logistic_regression,
    standardize_features as standardize_features_lr,
    plot_feature_importance as plot_feature_importance_lr,
    plot_roc_curve as plot_roc_lr,
    plot_confusion_matrix as plot_cm_lr
)
from src.models.baseline.random_forest import (
    train_random_forest,
    evaluate_model as evaluate_random_forest,
    plot_feature_importance as plot_feature_importance_rf,
    plot_roc_curve as plot_roc_rf,
    plot_confusion_matrix as plot_cm_rf
)
from src.models.baseline.svm import (
    train_svm,
    evaluate_model as evaluate_svm_original,
    standardize_features as standardize_features_svm,
    plot_roc_curve as plot_roc_svm,
    plot_confusion_matrix as plot_cm_svm
)

# Create a wrapper for SVM evaluation that matches the other evaluate_model functions
def evaluate_svm(model, X_test, y_test, feature_names=None):
    """Wrapper for SVM evaluation that accepts feature_names parameter to match other evaluate functions"""
    return evaluate_svm_original(model, X_test, y_test)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('baseline_train_balanced.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
FEATURES_DIR = "outputs/features"
OUTPUT_DIR = "outputs/models_balanced"

def compute_class_weights(y):
    """
    Compute balanced class weights manually to handle class imbalance.
    
    Args:
        y (numpy.ndarray): Target labels
        
    Returns:
        dict: Class weights dictionary
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    
    class_weights = {}
    for i, c in enumerate(classes):
        class_weights[c] = n_samples / (n_classes * counts[i])
    
    logger.info(f"Computed class weights: {class_weights}")
    return class_weights

def plot_precision_recall_curve(model, X_test, y_test, output_path=None):
    """
    Plot Precision-Recall curve for a binary classification model.
    
    Args:
        model: Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        output_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: PR curve figure
    """
    try:
        y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        average_precision = average_precision_score(y_test, y_score)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='b',
                label=f'Average Precision = {average_precision:.2f}')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='best')
        ax.grid(True)
        
        if output_path:
            plt.savefig(output_path)
            
        return fig
    except Exception as e:
        logger.error(f"Error plotting precision-recall curve: {str(e)}")
        return None

def train_baseline_models():
    """
    Train baseline models using pre-extracted features with class imbalance handling.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load training features
    train_features_path = os.path.join(FEATURES_DIR, "train_statistical_features.pkl")
    val_features_path = os.path.join(FEATURES_DIR, "val_statistical_features.pkl")
    test_features_path = os.path.join(FEATURES_DIR, "test_statistical_features.pkl")
    
    # Check for top features from feature selection
    top_features_path = os.path.join(FEATURES_DIR, "top_features.csv")
    use_top_features = os.path.exists(top_features_path)
    top_features = None
    
    if use_top_features:
        try:
            top_features_df = pd.read_csv(top_features_path)
            top_features = top_features_df.index.tolist()
            logger.info(f"Using top {len(top_features)} features from feature selection")
        except Exception as e:
            logger.error(f"Error loading top features: {str(e)}")
            use_top_features = False
    
    try:
        with open(train_features_path, 'rb') as f:
            train_data = pickle.load(f)
        
        # Load validation set if available
        val_data = None
        if os.path.exists(val_features_path):
            with open(val_features_path, 'rb') as f:
                val_data = pickle.load(f)
                logger.info(f"Loaded validation features: {val_data['features'].shape}")
        
        with open(test_features_path, 'rb') as f:
            test_data = pickle.load(f)
        
        X_train = train_data['features']
        y_train = train_data['labels']
        feature_names = train_data['feature_names']
        
        X_val = val_data['features'] if val_data is not None else None
        y_val = val_data['labels'] if val_data is not None else None
        
        X_test = test_data['features']
        y_test = test_data['labels']
        
        logger.info(f"Loaded features - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # If using selected top features, filter the feature matrices
        if use_top_features and top_features:
            # Get indices of top features
            top_indices = [feature_names.index(feature) for feature in top_features if feature in feature_names]
            
            if top_indices:
                X_train = X_train[:, top_indices]
                X_test = X_test[:, top_indices]
                
                if X_val is not None:
                    X_val = X_val[:, top_indices]
                
                # Update feature names
                feature_names = [feature_names[i] for i in top_indices]
                logger.info(f"Using {len(top_indices)} selected features")
            else:
                logger.warning("No top features found in the current feature set, using all features")
        
        logger.info(f"Original class distribution - Train: {np.unique(y_train, return_counts=True)}")
        logger.info(f"Original class distribution - Test: {np.unique(y_test, return_counts=True)}")
        
        # Analyze whether to use binary classification (0 vs >0) or multi-class
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        
        use_binary = False
        
        # If we only have two classes in test, use binary classification
        if len(test_classes) <= 2 or (len(train_classes) > 2 and 0.5 in train_classes):
            use_binary = True
            logger.info("Converting to binary classification (0 = normal, >0 = impaired)")
            
            # Convert to binary classification: 0 -> 0 (Normal), >0 -> 1 (Impaired)
            y_train_binary = np.array([0 if y == 0 else 1 for y in y_train])
            y_test_binary = np.array([0 if y == 0 else 1 for y in y_test])
            
            if y_val is not None:
                y_val_binary = np.array([0 if y == 0 else 1 for y in y_val])
            
            logger.info(f"Binary class distribution - Train: {np.unique(y_train_binary, return_counts=True)}")
            logger.info(f"Binary class distribution - Test: {np.unique(y_test_binary, return_counts=True)}")
            
            # Use binary labels
            y_train = y_train_binary
            y_test = y_test_binary
            if y_val is not None:
                y_val = y_val_binary
        elif 0.5 in train_classes or 0.5 in test_classes:
            # Convert to integer class labels for multi-class classification
            logger.info("Using multi-class classification with integer labels")
            
            # Map CDR scores: 0 -> 0, 0.5 -> 1, 1.0 -> 2
            y_train_multi = np.array([0 if y == 0 else (1 if y == 0.5 else 2) for y in y_train])
            y_test_multi = np.array([0 if y == 0 else (1 if y == 0.5 else 2) for y in y_test])
            
            if y_val is not None:
                y_val_multi = np.array([0 if y == 0 else (1 if y == 0.5 else 2) for y in y_val])
            
            logger.info(f"Multi-class distribution - Train: {np.unique(y_train_multi, return_counts=True)}")
            logger.info(f"Multi-class distribution - Test: {np.unique(y_test_multi, return_counts=True)}")
            
            # Use multi-class labels
            y_train = y_train_multi
            y_test = y_test_multi
            if y_val is not None:
                y_val = y_val_multi
        
        # Compute class weights to handle imbalance
        class_weights = 'balanced'
        
        # Define models to train with class imbalance handling
        models = {
            'logistic_regression': {
                'train_func': train_logistic_regression,
                'eval_func': evaluate_logistic_regression,
                'standardize_func': standardize_features_lr,
                'plot_importance_func': plot_feature_importance_lr,
                'plot_roc_func': plot_roc_lr,
                'plot_cm_func': plot_cm_lr,
                'needs_scaling': True,
                'params': {'C': 1.0, 'penalty': 'l2', 'class_weight': class_weights}
            },
            'random_forest': {
                'train_func': train_random_forest,
                'eval_func': evaluate_random_forest,
                'standardize_func': None,
                'plot_importance_func': plot_feature_importance_rf,
                'plot_roc_func': plot_roc_rf,
                'plot_cm_func': plot_cm_rf,
                'needs_scaling': False,
                'params': {'n_estimators': 100, 'max_depth': None, 'class_weight': class_weights}
            },
            'svm': {
                'train_func': train_svm,
                'eval_func': evaluate_svm,
                'standardize_func': standardize_features_svm,
                'plot_importance_func': None,
                'plot_roc_func': plot_roc_svm,
                'plot_cm_func': plot_cm_svm,
                'needs_scaling': True,
                'params': {'C': 1.0, 'kernel': 'rbf', 'class_weight': class_weights}
            }
        }
        
        # Store results for comparison
        all_results = {}
        
        # Train and evaluate each model
        for model_name, model_info in models.items():
            logger.info(f"Training {model_name} model with class balancing...")
            
            # Apply standardization if needed
            X_train_proc = X_train
            X_test_proc = X_test
            
            if model_info['needs_scaling'] and model_info['standardize_func']:
                X_train_proc, X_test_proc, _, _ = model_info['standardize_func'](X_train, X_test)
            
            # Train model
            model = model_info['train_func'](X_train_proc, y_train, **model_info['params'])
            
            if model is None:
                logger.error(f"Failed to train {model_name} model")
                continue
            
            # Create model directory
            model_dir = os.path.join(OUTPUT_DIR, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            with open(os.path.join(model_dir, "model.pkl"), 'wb') as f:
                pickle.dump(model, f)
            
            # Evaluate model
            logger.info(f"Evaluating {model_name} model...")
            metrics = model_info['eval_func'](model, X_test_proc, y_test, feature_names)
            
            # Add F1 score for each class (weighted more important for imbalanced)
            y_pred = model.predict(X_test_proc)
            metrics['f1_score'] = {}
            for avg in ['micro', 'macro', 'weighted']:
                metrics['f1_score'][avg] = float(f1_score(y_test, y_pred, average=avg))
            
            logger.info(f"F1 Score (weighted): {metrics['f1_score']['weighted']:.4f}")
            
            # Save metrics
            with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
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
            
            # Store results for comparison
            all_results[model_name] = {
                'accuracy': metrics['accuracy'] if 'accuracy' in metrics else 0,
                'f1_weighted': metrics['f1_score']['weighted'] if 'f1_score' in metrics else 0,
                'roc_auc': metrics['roc_auc'] if 'roc_auc' in metrics else 0
            }
            
            # Plot feature importance if available
            if model_info['plot_importance_func']:
                logger.info(f"Plotting feature importance for {model_name}...")
                fig = model_info['plot_importance_func'](model, feature_names)
                if fig:
                    plt.savefig(os.path.join(model_dir, "feature_importance.png"))
                    plt.close(fig)
            
            # Plot ROC curve if available
            if model_info['plot_roc_func']:
                logger.info(f"Plotting ROC curve for {model_name}...")
                fig = model_info['plot_roc_func'](model, X_test_proc, y_test)
                if fig:
                    plt.savefig(os.path.join(model_dir, "roc_curve.png"))
                    plt.close(fig)
            
            # Plot confusion matrix if available
            if model_info['plot_cm_func']:
                logger.info(f"Plotting confusion matrix for {model_name}...")
                fig = model_info['plot_cm_func'](model, X_test_proc, y_test)
                if fig:
                    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
                    plt.close(fig)
                
            # Plot precision-recall curve
            logger.info(f"Plotting precision-recall curve for {model_name}...")
            fig = plot_precision_recall_curve(model, X_test_proc, y_test)
            if fig:
                plt.savefig(os.path.join(model_dir, "precision_recall_curve.png"))
                plt.close(fig)
        
        # Save comparison of all models
        comparison_df = pd.DataFrame(all_results).T
        comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
        logger.info(f"Model comparison:\n{comparison_df}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting balanced baseline model training")
    
    # Train models
    success = train_baseline_models()
    
    if success:
        logger.info("Balanced baseline model training completed successfully")
    else:
        logger.error("Balanced baseline model training failed") 