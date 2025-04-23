"""
K-Nearest Neighbors baseline model for Alzheimer's detection.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def train_knn(X_train, y_train, n_neighbors=5, weights='uniform', metric='minkowski', p=2):
    """
    Train a K-Nearest Neighbors classifier with specified parameters.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        n_neighbors (int, optional): Number of neighbors.
        weights (str, optional): Weight function ('uniform' or 'distance').
        metric (str, optional): Distance metric.
        p (int, optional): Power parameter for Minkowski metric.
        
    Returns:
        KNeighborsClassifier: Trained model.
    """
    if X_train is None or y_train is None:
        logger.error("Cannot train model with None inputs")
        return None
    
    try:
        # Initialize and train model
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Log model information
        train_accuracy = model.score(X_train, y_train)
        logger.info(f"Trained KNN model (n_neighbors={n_neighbors}, weights={weights}, metric={metric})")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training KNN model: {str(e)}")
        return None

def hyperparameter_tuning(X_train, y_train, X_val=None, y_val=None, cv=5, scoring='accuracy'):
    """
    Perform hyperparameter tuning for K-Nearest Neighbors.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray, optional): Validation features.
        y_val (numpy.ndarray, optional): Validation labels.
        cv (int, optional): Number of cross-validation folds.
        scoring (str, optional): Scoring metric.
        
    Returns:
        tuple: (best_model, grid_results) - Best model and grid search results.
    """
    if X_train is None or y_train is None:
        logger.error("Cannot tune hyperparameters with None inputs")
        return None, None
    
    try:
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]  # p=1 for manhattan, p=2 for euclidean
        }
        
        # Initialize grid search
        grid_search = GridSearchCV(
            KNeighborsClassifier(n_jobs=-1),
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
    Evaluate a trained K-Nearest Neighbors model.
    
    Args:
        model (KNeighborsClassifier): Trained model.
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

def plot_roc_curve(model, X_test, y_test, output_path=None):
    """
    Plot ROC curve for a binary K-Nearest Neighbors model.
    
    Args:
        model (KNeighborsClassifier): Trained model.
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
    Plot confusion matrix for a K-Nearest Neighbors model.
    
    Args:
        model (KNeighborsClassifier): Trained model.
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

def find_optimal_k(X_train, y_train, X_val, y_val, k_range=None, metric='accuracy'):
    """
    Find the optimal number of neighbors (k) for KNN.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.
        k_range (list, optional): Range of k values to try.
        metric (str, optional): Evaluation metric.
        
    Returns:
        tuple: (optimal_k, metrics_dict, fig) - Optimal k value, metrics for each k, and figure.
    """
    if X_train is None or y_train is None or X_val is None or y_val is None:
        logger.error("Cannot find optimal k with None inputs")
        return None, None, None
    
    if k_range is None:
        k_range = list(range(1, min(31, len(X_train))))
    
    try:
        # Initialize metrics
        metrics = {'k': k_range, 'accuracy': [], 'roc_auc': []}
        
        # Evaluate each k
        for k in k_range:
            # Train model
            model = train_knn(X_train, y_train, n_neighbors=k)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_val, y_pred)
            metrics['accuracy'].append(accuracy)
            
            # Calculate ROC AUC for binary classification
            if len(np.unique(y_train)) == 2:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_pred_proba)
                metrics['roc_auc'].append(roc_auc)
        
        # Find optimal k
        if metric == 'roc_auc' and 'roc_auc' in metrics:
            optimal_k = k_range[np.argmax(metrics['roc_auc'])]
            best_score = np.max(metrics['roc_auc'])
            logger.info(f"Optimal k: {optimal_k} with ROC AUC: {best_score:.4f}")
        else:
            optimal_k = k_range[np.argmax(metrics['accuracy'])]
            best_score = np.max(metrics['accuracy'])
            logger.info(f"Optimal k: {optimal_k} with accuracy: {best_score:.4f}")
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(k_range, metrics['accuracy'], marker='o', linestyle='-', label='Accuracy')
        
        if 'roc_auc' in metrics:
            ax.plot(k_range, metrics['roc_auc'], marker='s', linestyle='-', label='ROC AUC')
        
        ax.set_xlabel('Number of Neighbors (k)')
        ax.set_ylabel('Score')
        ax.set_title('KNN Performance vs. Number of Neighbors')
        ax.legend()
        ax.grid(True)
        
        # Highlight optimal k
        ax.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        ax.text(optimal_k, 0.5, f'Optimal k = {optimal_k}', rotation=90, verticalalignment='center')
        
        return optimal_k, metrics, fig
    
    except Exception as e:
        logger.error(f"Error finding optimal k: {str(e)}")
        return None, None, None

def plot_decision_regions(X, y, model, feature_idx1=0, feature_idx2=1, resolution=0.02, output_path=None):
    """
    Plot decision regions for a KNN model (for 2D visualization).
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels.
        model (KNeighborsClassifier): Trained model.
        feature_idx1 (int, optional): Index of first feature to plot.
        feature_idx2 (int, optional): Index of second feature to plot.
        resolution (float, optional): Resolution of the mesh grid.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        matplotlib.figure.Figure: The decision regions figure.
    """
    if model is None or X is None or y is None:
        logger.error("Cannot plot decision regions with None inputs")
        return None
    
    try:
        # Select two features for visualization
        X_subset = X[:, [feature_idx1, feature_idx2]]
        
        # Define mesh grid
        x1_min, x1_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
        x2_min, x2_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
        
        # Create dummy features for prediction
        dummy_features = np.zeros((xx1.ravel().shape[0], X.shape[1]))
        dummy_features[:, feature_idx1] = xx1.ravel()
        dummy_features[:, feature_idx2] = xx2.ravel()
        
        # Make predictions
        Z = model.predict(dummy_features)
        Z = Z.reshape(xx1.shape)
        
        # Plot decision regions
        fig, ax = plt.subplots(figsize=(12, 10))
        contour = ax.contourf(xx1, xx2, Z, alpha=0.3, cmap='viridis')
        
        # Plot data points
        scatter = ax.scatter(X_subset[:, 0], X_subset[:, 1], c=y, edgecolors='k', cmap='viridis')
        
        # Set labels and title
        ax.set_xlabel(f'Feature {feature_idx1}')
        ax.set_ylabel(f'Feature {feature_idx2}')
        ax.set_title('Decision Regions for KNN Model')
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        
        # Save the plot if output path is provided
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path)
            logger.info(f"Decision regions plot saved to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting decision regions: {str(e)}")
        return None 