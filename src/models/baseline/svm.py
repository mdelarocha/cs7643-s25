"""
SVM baseline model for Alzheimer's detection.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def train_svm(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', class_weight=None, random_state=42):
    """
    Train an SVM classifier with specified parameters.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        C (float, optional): Regularization parameter.
        kernel (str, optional): Kernel type ('linear', 'poly', 'rbf', 'sigmoid').
        gamma (str or float, optional): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        class_weight (dict or str, optional): Class weights.
        random_state (int, optional): Random seed.
        
    Returns:
        SVC: Trained model.
    """
    if X_train is None or y_train is None:
        logger.error("Cannot train model with None inputs")
        return None
    
    try:
        # Initialize and train model
        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # Log model information
        train_accuracy = model.score(X_train, y_train)
        logger.info(f"Trained SVM model (C={C}, kernel={kernel}, gamma={gamma})")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training SVM model: {str(e)}")
        return None

def hyperparameter_tuning(X_train, y_train, X_val=None, y_val=None, cv=5, scoring='accuracy', random_state=42):
    """
    Perform hyperparameter tuning for SVM.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray, optional): Validation features.
        y_val (numpy.ndarray, optional): Validation labels.
        cv (int, optional): Number of cross-validation folds.
        scoring (str, optional): Scoring metric.
        random_state (int, optional): Random seed.
        
    Returns:
        tuple: (best_model, grid_results) - Best model and grid search results.
    """
    if X_train is None or y_train is None:
        logger.error("Cannot tune hyperparameters with None inputs")
        return None, None
    
    try:
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'class_weight': [None, 'balanced']
        }
        
        # Initialize grid search
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log results
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation {scoring}: {grid_search.best_score_:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_accuracy = best_model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return best_model, grid_search.cv_results_
    
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return None, None

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained SVM model.
    
    Args:
        model (SVC): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    if model is None or X_test is None or y_test is None:
        logger.error("Cannot evaluate model with None inputs")
        return {}
    
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # ROC AUC for binary classification
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Log results
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return {}

def extract_support_vectors(model, X_train, feature_names=None):
    """
    Extract support vectors from a trained SVM model.
    
    Args:
        model (SVC): Trained model.
        X_train (numpy.ndarray): Training features.
        feature_names (list, optional): Names of features.
        
    Returns:
        pandas.DataFrame: Support vectors with feature information.
    """
    if model is None or X_train is None:
        logger.error("Cannot extract support vectors from None inputs")
        return None
    
    try:
        # Get support vectors
        support_vectors = model.support_vectors_
        
        # Get indices of support vectors
        support_indices = model.support_
        
        # Log information
        logger.info(f"Extracted {len(support_indices)} support vectors")
        
        # Create DataFrame with feature values
        if feature_names is not None and len(feature_names) == X_train.shape[1]:
            support_vector_df = pd.DataFrame(support_vectors, columns=feature_names)
        else:
            support_vector_df = pd.DataFrame(support_vectors)
        
        return support_vector_df
    
    except Exception as e:
        logger.error(f"Error extracting support vectors: {str(e)}")
        return None

def plot_roc_curve(model, X_test, y_test, output_path=None):
    """
    Plot ROC curve for a binary SVM model.
    
    Args:
        model (SVC): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        matplotlib.figure.Figure: The ROC curve figure.
    """
    if len(model.classes_) != 2:
        logger.error("ROC curve can only be plotted for binary classification")
        return None
    
    try:
        from sklearn.metrics import roc_curve, auc
        
        # Get predicted probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        
        # Save the plot if output path is provided
        if output_path:
            plt.savefig(output_path)
            logger.info(f"ROC curve saved to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")
        return None

def plot_confusion_matrix(model, X_test, y_test, output_path=None):
    """
    Plot confusion matrix for an SVM model.
    
    Args:
        model (SVC): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        matplotlib.figure.Figure: The confusion matrix figure.
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(model.classes_)
        ax.set_yticklabels(model.classes_)
        
        # Save the plot if output path is provided
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Confusion matrix saved to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        return None

def standardize_features(X_train, X_test=None, X_val=None):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray, optional): Test features.
        X_val (numpy.ndarray, optional): Validation features.
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, X_val_scaled, scaler) - Scaled features and scaler.
    """
    if X_train is None:
        logger.error("Cannot standardize None features")
        return None, None, None, None
    
    try:
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit on training data and transform
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Transform test and validation data if provided
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
        X_val_scaled = scaler.transform(X_val) if X_val is not None else None
        
        logger.info("Features standardized")
        
        return X_train_scaled, X_test_scaled, X_val_scaled, scaler
    
    except Exception as e:
        logger.error(f"Error standardizing features: {str(e)}")
        return X_train, X_test, X_val, None

def get_feature_weights_for_linear_svm(model, feature_names=None):
    """
    Get feature weights for a linear SVM model (only applicable for linear kernel).
    
    Args:
        model (SVC): Trained SVM model with a linear kernel.
        feature_names (list, optional): Names of features.
        
    Returns:
        pandas.DataFrame: Feature weights.
    """
    if model is None:
        logger.error("Cannot get feature weights from None model")
        return None
    
    if model.kernel != 'linear':
        logger.error("Feature weights can only be extracted from a linear SVM model")
        return None
    
    try:
        # Get coefficients (weights)
        weights = model.coef_[0]
        
        # Create DataFrame
        if feature_names is not None and len(feature_names) == len(weights):
            weights_df = pd.DataFrame({
                'feature': feature_names,
                'weight': weights
            })
        else:
            weights_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(weights))],
                'weight': weights
            })
        
        # Sort by absolute weight value
        weights_df['abs_weight'] = weights_df['weight'].abs()
        weights_df = weights_df.sort_values('abs_weight', ascending=False)
        
        logger.info(f"Extracted {len(weights)} feature weights from linear SVM")
        
        return weights_df
    
    except Exception as e:
        logger.error(f"Error getting feature weights: {str(e)}")
        return None

def plot_feature_weights(model, feature_names, top_n=20, output_path=None):
    """
    Plot feature weights for a linear SVM model.
    
    Args:
        model (SVC): Trained SVM model with a linear kernel.
        feature_names (list): Names of features.
        top_n (int, optional): Number of top features to show.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        matplotlib.figure.Figure: The feature weights figure.
    """
    if model is None or model.kernel != 'linear':
        logger.error("Feature weights can only be plotted for a linear SVM model")
        return None
    
    try:
        # Get feature weights
        weights_df = get_feature_weights_for_linear_svm(model, feature_names)
        
        if weights_df is None:
            return None
        
        # Take top N features
        top_weights = weights_df.head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        bars = ax.barh(top_weights['feature'], top_weights['weight'])
        
        # Color bars based on weight sign
        for i, bar in enumerate(bars):
            if top_weights.iloc[i]['weight'] < 0:
                bar.set_color('r')
            else:
                bar.set_color('g')
        
        ax.set_xlabel('Weight')
        ax.set_title(f'Top {top_n} Features by Weight (Linear SVM)')
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Save the plot if output path is provided
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path)
            logger.info(f"Feature weights plot saved to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting feature weights: {str(e)}")
        return None 