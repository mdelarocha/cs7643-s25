"""
K-Means clustering baseline model for Alzheimer's detection.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

logger = logging.getLogger(__name__)

def train_kmeans(X_train, n_clusters=2, random_state=42, n_init=10):
    """
    Train a K-Means clustering model with specified parameters.
    
    Args:
        X_train (numpy.ndarray): Training features.
        n_clusters (int, optional): Number of clusters.
        random_state (int, optional): Random seed.
        n_init (int, optional): Number of times the k-means algorithm will run with different centroid seeds.
        
    Returns:
        KMeans: Trained model.
    """
    if X_train is None:
        logger.error("Cannot train model with None inputs")
        return None
    
    try:
        # Initialize and train model
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init
        )
        
        model.fit(X_train)
        
        # Log model information
        logger.info(f"Trained KMeans model (n_clusters={n_clusters})")
        logger.info(f"Inertia: {model.inertia_:.4f}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training KMeans model: {str(e)}")
        return None

def find_optimal_clusters(X, max_clusters=10, random_state=42):
    """
    Find the optimal number of clusters using the elbow method and silhouette score.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        max_clusters (int, optional): Maximum number of clusters to try.
        random_state (int, optional): Random seed.
        
    Returns:
        tuple: (optimal_k, metrics_dict, fig) - Optimal number of clusters, metrics for each k, and figure.
    """
    if X is None:
        logger.error("Cannot find optimal clusters with None inputs")
        return None, None, None
    
    try:
        # Initialize metrics
        metrics = {
            'k': range(2, max_clusters + 1),
            'inertia': [],
            'silhouette': []
        }
        
        # Evaluate each k
        for k in metrics['k']:
            # Train model
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            model.fit(X)
            
            # Calculate inertia (within-cluster sum of squares)
            metrics['inertia'].append(model.inertia_)
            
            # Calculate silhouette score (measure of how similar an object is to its own cluster)
            if len(X) > k:  # Silhouette score requires more samples than clusters
                silhouette = silhouette_score(X, model.labels_)
                metrics['silhouette'].append(silhouette)
            else:
                metrics['silhouette'].append(0)
        
        # Find optimal k using silhouette score
        optimal_k = metrics['k'][np.argmax(metrics['silhouette'])]
        
        # Log results
        logger.info(f"Optimal number of clusters: {optimal_k} with silhouette score: {max(metrics['silhouette']):.4f}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot inertia (elbow method)
        ax1.plot(metrics['k'], metrics['inertia'], marker='o')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        
        # Plot silhouette score
        ax2.plot(metrics['k'], metrics['silhouette'], marker='o')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Method for Optimal k')
        ax2.grid(True)
        
        # Highlight optimal k
        ax1.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        return optimal_k, metrics, fig
    
    except Exception as e:
        logger.error(f"Error finding optimal clusters: {str(e)}")
        return None, None, None

def assign_cluster_labels(cluster_labels, true_labels):
    """
    Assign class labels to clusters by majority voting.
    
    Args:
        cluster_labels (numpy.ndarray): Cluster assignments.
        true_labels (numpy.ndarray): True class labels.
        
    Returns:
        tuple: (cluster_to_class_map, accuracy) - Mapping from clusters to classes and accuracy.
    """
    if cluster_labels is None or true_labels is None:
        logger.error("Cannot assign cluster labels with None inputs")
        return {}, 0
    
    try:
        # Initialize mapping
        unique_clusters = np.unique(cluster_labels)
        cluster_to_class = {}
        
        # For each cluster, find the most common true label
        for cluster in unique_clusters:
            mask = (cluster_labels == cluster)
            cluster_true_labels = true_labels[mask]
            
            if len(cluster_true_labels) > 0:
                most_common_label = mode(cluster_true_labels).mode[0]
                cluster_to_class[cluster] = most_common_label
            else:
                cluster_to_class[cluster] = -1  # Default for empty clusters
        
        # Calculate accuracy using the mapping
        predicted_labels = np.array([cluster_to_class[cluster] for cluster in cluster_labels])
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        logger.info(f"Assigned cluster labels with accuracy: {accuracy:.4f}")
        logger.info(f"Cluster to class mapping: {cluster_to_class}")
        
        return cluster_to_class, accuracy
    
    except Exception as e:
        logger.error(f"Error assigning cluster labels: {str(e)}")
        return {}, 0

def predict_with_kmeans(model, X, cluster_to_class_map=None):
    """
    Make predictions with a K-Means model, optionally mapping clusters to classes.
    
    Args:
        model (KMeans): Trained model.
        X (numpy.ndarray): Features to predict.
        cluster_to_class_map (dict, optional): Mapping from cluster IDs to class labels.
        
    Returns:
        numpy.ndarray: Predicted cluster assignments or class labels.
    """
    if model is None or X is None:
        logger.error("Cannot predict with None inputs")
        return None
    
    try:
        # Get cluster assignments
        cluster_assignments = model.predict(X)
        
        # If mapping is provided, map clusters to classes
        if cluster_to_class_map is not None:
            class_predictions = np.array([cluster_to_class_map.get(cluster, -1) for cluster in cluster_assignments])
            return class_predictions
        else:
            return cluster_assignments
    
    except Exception as e:
        logger.error(f"Error predicting with KMeans: {str(e)}")
        return None

def evaluate_clustering_as_classifier(model, X_test, y_test, cluster_to_class_map):
    """
    Evaluate a K-Means model as a classifier using a mapping from clusters to classes.
    
    Args:
        model (KMeans): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        cluster_to_class_map (dict): Mapping from cluster IDs to class labels.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    if model is None or X_test is None or y_test is None or not cluster_to_class_map:
        logger.error("Cannot evaluate model with None inputs")
        return {}
    
    try:
        # Make predictions
        y_pred = predict_with_kmeans(model, X_test, cluster_to_class_map)
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Log results
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating clustering as classifier: {str(e)}")
        return {}

def plot_clusters(X, model, X_reduced=None, feature_idx1=0, feature_idx2=1, true_labels=None, output_path=None):
    """
    Plot clusters assigned by K-Means model.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        model (KMeans): Trained model.
        X_reduced (numpy.ndarray, optional): Dimensionality-reduced features (e.g., PCA).
        feature_idx1 (int, optional): Index of first feature to plot if X_reduced not provided.
        feature_idx2 (int, optional): Index of second feature to plot if X_reduced not provided.
        true_labels (numpy.ndarray, optional): True class labels for comparison.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        matplotlib.figure.Figure: The cluster plot figure.
    """
    if model is None or X is None:
        logger.error("Cannot plot clusters with None inputs")
        return None
    
    try:
        # Get cluster assignments
        cluster_labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        
        # Prepare data for plotting
        if X_reduced is not None:
            # Use pre-reduced data (e.g., PCA)
            plot_data = X_reduced[:, :2]  # Use first two components
        else:
            # Use selected features
            plot_data = X[:, [feature_idx1, feature_idx2]]
        
        # Create figure
        n_plots = 1 if true_labels is None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        
        if n_plots == 1:
            ax1 = axes
        else:
            ax1, ax2 = axes
        
        # Plot clusters
        scatter = ax1.scatter(plot_data[:, 0], plot_data[:, 1], c=cluster_labels, cmap='viridis', 
                          edgecolor='k', s=50, alpha=0.8)
        
        # Plot centroids
        centroids = model.cluster_centers_
        if X_reduced is not None:
            # If data is reduced, we need to project centroids to the same space
            # For simplicity, we'll skip this and just not plot centroids for reduced data
            pass
        else:
            # Plot centroids in original feature space
            ax1.scatter(centroids[:, feature_idx1], centroids[:, feature_idx2], 
                     marker='X', s=200, color='red', label='Centroids')
        
        ax1.set_title('K-Means Clustering Results')
        ax1.set_xlabel('Feature 1' if X_reduced is None else 'Component 1')
        ax1.set_ylabel('Feature 2' if X_reduced is None else 'Component 2')
        ax1.legend()
        
        # If true labels provided, plot for comparison
        if true_labels is not None:
            scatter2 = ax2.scatter(plot_data[:, 0], plot_data[:, 1], c=true_labels, cmap='viridis', 
                              edgecolor='k', s=50, alpha=0.8)
            ax2.set_title('True Labels')
            ax2.set_xlabel('Feature 1' if X_reduced is None else 'Component 1')
            ax2.set_ylabel('Feature 2' if X_reduced is None else 'Component 2')
            
            # Add legend for true labels
            legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes")
            ax2.add_artist(legend2)
        
        plt.tight_layout()
        
        # Save the plot if output path is provided
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Cluster plot saved to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting clusters: {str(e)}")
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

def plot_silhouette_analysis(X, model, output_path=None):
    """
    Plot silhouette analysis for K-Means clustering results.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        model (KMeans): Trained model.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        matplotlib.figure.Figure: The silhouette analysis figure.
    """
    if X is None or model is None:
        logger.error("Cannot plot silhouette with None inputs")
        return None
    
    try:
        from sklearn.metrics import silhouette_samples
        
        # Compute cluster labels and silhouette scores
        cluster_labels = model.predict(X)
        n_clusters = len(set(cluster_labels))
        
        # Skip if only one cluster or more clusters than samples
        if n_clusters <= 1 or n_clusters >= len(X):
            logger.error(f"Cannot plot silhouette with {n_clusters} clusters for {len(X)} samples")
            return None
        
        # Compute the silhouette scores for each sample
        silhouette_vals = silhouette_samples(X, cluster_labels)
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        y_lower = 10
        
        # Plot silhouette scores for each cluster
        for i in range(n_clusters):
            # Get silhouette values for cluster i
            ith_cluster_silhouette_values = silhouette_vals[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            # Size of the cluster
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            # Generate color for the cluster
            color = plt.cm.viridis(float(i) / n_clusters)
            
            # Fill the silhouette
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, 
                           ith_cluster_silhouette_values,
                           facecolor=color, alpha=0.7)
            
            # Label the cluster
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        # Add average silhouette score line
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        ax.set_title(f"Silhouette Analysis (Avg. Score: {silhouette_avg:.3f})")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.set_yticks([])  # Clear y ticks
        ax.set_xlim([-0.1, 1])
        
        # Save the plot if output path is provided
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path)
            logger.info(f"Silhouette plot saved to {output_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting silhouette analysis: {str(e)}")
        return None 