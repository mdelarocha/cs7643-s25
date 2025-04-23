#!/usr/bin/env python
"""
Script to train baseline models using pre-extracted features.
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
    evaluate_model as evaluate_svm,
    standardize_features as standardize_features_svm,
    plot_roc_curve as plot_roc_svm,
    plot_confusion_matrix as plot_cm_svm
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('baseline_train.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
FEATURES_DIR = "outputs/features"
OUTPUT_DIR = "outputs/models"

def train_baseline_models():
    """
    Train baseline models using pre-extracted features.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load training features
    train_features_path = os.path.join(FEATURES_DIR, "train_stat_features.pkl")
    test_features_path = os.path.join(FEATURES_DIR, "test_stat_features.pkl")
    
    try:
        with open(train_features_path, 'rb') as f:
            train_data = pickle.load(f)
        
        with open(test_features_path, 'rb') as f:
            test_data = pickle.load(f)
        
        X_train = train_data['features']
        y_train = train_data['labels']
        feature_names = train_data['feature_names']
        
        X_test = test_data['features']
        y_test = test_data['labels']
        
        logger.info(f"Loaded features - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Define models to train
        models = {
            'logistic_regression': {
                'train_func': train_logistic_regression,
                'eval_func': evaluate_logistic_regression,
                'standardize_func': standardize_features_lr,
                'plot_importance_func': plot_feature_importance_lr,
                'plot_roc_func': plot_roc_lr,
                'plot_cm_func': plot_cm_lr,
                'needs_scaling': True,
                'params': {'C': 1.0, 'penalty': 'l2'}
            },
            'random_forest': {
                'train_func': train_random_forest,
                'eval_func': evaluate_random_forest,
                'standardize_func': None,
                'plot_importance_func': plot_feature_importance_rf,
                'plot_roc_func': plot_roc_rf,
                'plot_cm_func': plot_cm_rf,
                'needs_scaling': False,
                'params': {'n_estimators': 100, 'max_depth': None}
            },
            'svm': {
                'train_func': train_svm,
                'eval_func': evaluate_svm,
                'standardize_func': standardize_features_svm,
                'plot_roc_func': plot_roc_svm,
                'plot_cm_func': plot_cm_svm,
                'needs_scaling': True,
                'params': {'C': 1.0, 'kernel': 'rbf'}
            }
        }
        
        # Train and evaluate each model
        for model_name, model_info in models.items():
            logger.info(f"Training {model_name} model...")
            
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
        
        return True
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting baseline model training")
    
    # Train models
    success = train_baseline_models()
    
    if success:
        logger.info("Baseline model training completed successfully")
    else:
        logger.error("Baseline model training failed") 