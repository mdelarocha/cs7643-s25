"""
K-Means clustering baseline model for Alzheimer's detection.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any, List

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, silhouette_score, 
    davies_bouldin_score, adjusted_rand_score, homogeneity_score, 
    completeness_score, v_measure_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

from ..base_model import BaseModel

logger = logging.getLogger(__name__)

# --- Helper function for scaling (Optional for K-Means, but often beneficial) ---
def _standardize_features(X_train: np.ndarray, X_test: Optional[np.ndarray] = None) \
                         -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Standardize features using StandardScaler. Fits on train, transforms train and test."""
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    logger.info(f"Scaled training data (KMeans). Mean: {np.mean(X_train_std):.3f}, Std: {np.std(X_train_std):.3f}")
    X_test_std = scaler.transform(X_test) if X_test is not None else None
    if X_test_std is not None: logger.debug("Scaled test data (KMeans).")
    return X_train_std, X_test_std, scaler

# --- Helper to map cluster labels to true labels using Hungarian algorithm ---
def _map_clusters_to_labels(y_clusters: np.ndarray, y_true: np.ndarray) -> Dict[int, int]:
    """Maps cluster IDs to true label IDs using the Hungarian algorithm."""
    y_clusters = np.asarray(y_clusters)
    y_true = np.asarray(y_true)
    assert y_clusters.size == y_true.size
    
    unique_clusters = np.unique(y_clusters)
    unique_labels = np.unique(y_true)
    n_clusters = len(unique_clusters)
    n_labels = len(unique_labels)
    
    # Create cost matrix (negative intersection count)
    cost_matrix = np.zeros((n_clusters, n_labels), dtype=int)
    for i, cluster_id in enumerate(unique_clusters):
        for j, label_id in enumerate(unique_labels):
            # Count how many times this cluster overlaps with this true label
            mask = (y_clusters == cluster_id) & (y_true == label_id)
            cost_matrix[i, j] = -np.sum(mask)
    
    # Find the optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create the mapping dictionary
    mapping = {unique_clusters[r]: unique_labels[c] for r, c in zip(row_ind, col_ind)}
    logger.debug(f"Cluster to label mapping: {mapping}")
    return mapping

# --- KMeans Model Class ---
class KMeansModel(BaseModel):
    """KMeans clustering model implementing the BaseModel interface."""

    @property
    def needs_scaling(self) -> bool:
        # K-Means is distance-based, so scaling is usually recommended
        return True 

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'KMeansModel':
        """Fit the KMeans model. Uses y_train only to determine n_clusters (or uses param).
           Optionally standardizes data before fitting."""
        logger.info(f"Fitting KMeans model...")
        
        # 1. Determine n_clusters
        if 'n_clusters' in self.model_params:
            n_clusters = self.model_params['n_clusters']
            logger.info(f"Using n_clusters={n_clusters} from model_params.")
        elif y_train is not None:
            n_clusters = len(np.unique(y_train))
            logger.info(f"Determined n_clusters={n_clusters} from unique labels in y_train.")
        else:
            default_k = 3 # Arbitrary default if no info given
            logger.warning(f"Cannot determine n_clusters from y_train or params. Using default k={default_k}")
            n_clusters = default_k
            
        self.model_params['n_clusters'] = n_clusters # Store the determined k
        
        # 2. Standardize features (optional but recommended)
        X_train_proc = X_train
        self.scaler = None # Initialize scaler as None
        if self.model_params.get('standardize', True): # Default to standardize=True
             logger.info("Standardizing features for KMeans.")
             try:
                  X_train_proc, _, self.scaler = _standardize_features(X_train)
             except Exception as e:
                  logger.exception("Error during standardization for KMeans. Fitting on unscaled data.")
                  X_train_proc = X_train # Use original data if scaling fails
             
        # 3. Train KMeans model
        logger.info(f"Training KMeans with k={n_clusters}...")
        try:
            init_method = self.model_params.get('init', 'k-means++')
            n_init = self.model_params.get('n_init', 10) # Default in scikit-learn
            max_iter = self.model_params.get('max_iter', 300)
            
            self.model = KMeans(
                n_clusters=n_clusters, 
                init=init_method, 
                n_init=n_init, 
                max_iter=max_iter, 
                random_state=42
            )
            self.model.fit(X_train_proc) # Fit on processed (potentially scaled) data
            logger.info(f"KMeans model training complete. Inertia: {self.model.inertia_:.2f}")

        except Exception as e:
            logger.exception(f"Error training KMeans model: {e}")
            raise
             
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments. Applies scaling if scaler was fitted."""
        if self.model is None: raise RuntimeError("Model not trained.")
        
        X_proc = X
        if self.scaler is not None:
            logger.debug("Applying scaling before KMeans prediction.")
            try:
                X_proc = self.scaler.transform(X)
            except Exception as e:
                 logger.exception("Error applying scaler during prediction. Predicting on raw data.")
                 X_proc = X # Use raw data if scaling fails
        
        try:
            predictions = self.model.predict(X_proc)
            return predictions
        except Exception as e:
             logger.exception(f"Error during prediction: {e}")
             raise

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """KMeans does not predict probabilities in the classification sense. Returns None."""
        logger.warning("predict_proba is not applicable for KMeans.")
        return None

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate KMeans clustering using internal and external metrics (if true labels provided)."""
        if self.model is None: raise RuntimeError("Model not trained.")
        
        X_test_proc = X_test
        if self.scaler is not None:
            logger.debug("Applying scaling before KMeans evaluation.")
            try:
                X_test_proc = self.scaler.transform(X_test)
            except Exception as e:
                 logger.exception("Error applying scaler during evaluation. Evaluating on raw data.")
                 X_test_proc = X_test # Use raw data if scaling fails

        logger.info(f"Evaluating KMeans model (k={self.model.n_clusters})...")
        metrics = {}
        try:
            y_pred_clusters = self.model.predict(X_test_proc)
            
            # --- Internal Evaluation Metrics (don't require true labels) ---
            try:
                 # Silhouette Score: Higher is better (-1 to 1)
                 metrics['silhouette_score'] = silhouette_score(X_test_proc, y_pred_clusters)
                 logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            except ValueError as sil_err: # Can fail if only 1 cluster predicted
                 logger.warning(f"Could not calculate Silhouette Score: {sil_err}")
                 metrics['silhouette_score'] = None
            
            try:
                 # Davies-Bouldin Score: Lower is better (>= 0)
                 metrics['davies_bouldin_score'] = davies_bouldin_score(X_test_proc, y_pred_clusters)
                 logger.info(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
            except ValueError as db_err:
                 logger.warning(f"Could not calculate Davies-Bouldin Score: {db_err}")
                 metrics['davies_bouldin_score'] = None

            # Inertia (Sum of squared distances to closest centroid)
            metrics['inertia'] = self.model.inertia_
            logger.info(f"  Inertia (Train Set): {metrics['inertia']:.2f}") # Note: Inertia is from training

            # --- External Evaluation Metrics (require true labels y_test) ---
            if y_test is not None:
                 logger.info("Calculating external evaluation metrics using true labels...")
                 try:
                      # Adjusted Rand Index (ARI): Measures similarity between true and cluster labels. Max=1.
                      metrics['adjusted_rand_index'] = adjusted_rand_score(y_test, y_pred_clusters)
                      logger.info(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
                 except Exception as ari_err:
                      logger.warning(f"Could not calculate Adjusted Rand Index: {ari_err}")
                      
                 try:
                      # Homogeneity, Completeness, V-measure: Range [0, 1], higher is better.
                      metrics['homogeneity_score'] = homogeneity_score(y_test, y_pred_clusters)
                      metrics['completeness_score'] = completeness_score(y_test, y_pred_clusters)
                      metrics['v_measure_score'] = v_measure_score(y_test, y_pred_clusters)
                      logger.info(f"  Homogeneity: {metrics['homogeneity_score']:.4f}, Completeness: {metrics['completeness_score']:.4f}, V-Measure: {metrics['v_measure_score']:.4f}")
                 except Exception as hcv_err:
                     logger.warning(f"Could not calculate Homogeneity/Completeness/V-Measure: {hcv_err}")
                     
                 # Evaluate as Classifier (Map clusters to labels & calc accuracy etc.)
                 try:
                     cluster_label_map = _map_clusters_to_labels(y_pred_clusters, y_test)
                     y_pred_mapped = np.vectorize(cluster_label_map.get)(y_pred_clusters)
                     
                     accuracy = accuracy_score(y_test, y_pred_mapped)
                     f1 = f1_score(y_test, y_pred_mapped, average='weighted', zero_division=0)
                     cm = confusion_matrix(y_test, y_pred_mapped)
                     report = classification_report(y_test, y_pred_mapped, zero_division=0, output_dict=True)
                     
                     metrics['mapped_accuracy'] = accuracy
                     metrics['mapped_f1_score'] = f1
                     metrics['mapped_confusion_matrix'] = cm
                     metrics['mapped_classification_report'] = report
                     logger.info(f"  Mapped Accuracy: {accuracy:.4f}, Mapped F1: {f1:.4f}")
                     
                 except Exception as map_err:
                      logger.warning(f"Could not evaluate KMeans as classifier via mapping: {map_err}")
                      
            return metrics
        except Exception as e:
            logger.exception(f"Error during KMeans evaluation: {e}")
            return {"error": str(e)}
            
    # KMeans does not have intrinsic feature importance
    def plot_feature_importance(self, feature_names: List[str], output_path: str):
        """Feature importance is not applicable for KMeans."""
        super().plot_feature_importance(feature_names, output_path)

    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray, output_path: str):
        """ROC curve is not applicable for KMeans clustering."""
        logger.warning(f"ROC curve plotting is not applicable for {self.__class__.__name__}. Skipping.")
        pass # Do nothing

    def plot_precision_recall_curve(self, X_test: np.ndarray, y_test: np.ndarray, output_path: str):
        """Precision-Recall curve is not applicable for KMeans clustering."""
        logger.warning(f"Precision-Recall curve plotting is not applicable for {self.__class__.__name__}. Skipping.")
        pass # Do nothing
        
    # Implement cluster plotting
    def plot_clusters(self, X: np.ndarray, y: Optional[np.ndarray], output_path: str):
         """Plot K-Means clusters (using first 2 features or PCA). Applies scaling if needed."""
         if self.model is None: raise RuntimeError("Model not trained.")
         
         X_proc = X
         if self.scaler is not None:
             try:
                 X_proc = self.scaler.transform(X)
             except Exception as e:
                  logger.exception("Error applying scaler for plotting. Plotting raw data.")
                  X_proc = X

         try:
             # Reduce dimensionality for plotting if necessary (e.g., > 2 features)
             if X_proc.shape[1] > 2:
                 logger.info("Performing PCA for 2D visualization of clusters.")
                 from sklearn.decomposition import PCA # Local import
                 pca = PCA(n_components=2, random_state=42)
                 X_plot = pca.fit_transform(X_proc)
                 xlabel, ylabel = "PCA Component 1", "PCA Component 2"
             elif X_proc.shape[1] == 2:
                 X_plot = X_proc
                 # Assume first two features are meaningful? Get names if available.
                 xlabel, ylabel = "Feature 1", "Feature 2" # Placeholder labels
             else:
                 logger.error(f"Cannot plot clusters for data with shape {X.shape}")
                 return
                 
             # Get cluster assignments and centroids
             cluster_assignments = self.predict(X) # Predict on original X, handles scaling inside
             centroids = self.model.cluster_centers_
             
             # If PCA was used, transform centroids
             if X_proc.shape[1] > 2:
                 centroids_plot = pca.transform(centroids)
             else:
                 centroids_plot = centroids
             
             plt.figure(figsize=(10, 8))
             # Plot data points, colored by predicted cluster
             sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=cluster_assignments, 
                             palette='viridis', alpha=0.7, legend='full')
             
             # Plot centroids
             plt.scatter(centroids_plot[:, 0], centroids_plot[:, 1], marker='X', s=200, 
                         c='red', edgecolors='k', label='Centroids')
             
             plt.title(f'K-Means Clustering (k={self.model.n_clusters})')
             plt.xlabel(xlabel)
             plt.ylabel(ylabel)
             plt.legend()
             plt.grid(True, linestyle='--', alpha=0.6)
             plt.tight_layout()
             
             plt.savefig(output_path)
             plt.close()
             logger.info(f"Saved cluster plot to {output_path}")
             
         except Exception as e:
              logger.exception(f"Error plotting KMeans clusters: {e}")
 
# --- Remove old standalone functions ---
# train_kmeans
# assign_cluster_labels
# evaluate_clustering
# evaluate_clustering_as_classifier
# plot_clusters
# plot_elbow_method
# plot_silhouette_scores 