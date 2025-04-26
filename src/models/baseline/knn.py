"""
K-Nearest Neighbors baseline model for Alzheimer's detection.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, log_loss
)

from ..base_model import BaseModel # Import base class

logger = logging.getLogger(__name__)

# --- Helper function for scaling ---
def _standardize_features(X_train: np.ndarray, X_val: Optional[np.ndarray] = None, X_test: Optional[np.ndarray] = None) \
                         -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    logger.info(f"Scaled training data (KNN). Mean: {np.mean(X_train_std):.3f}, Std: {np.std(X_train_std):.3f}")
    X_val_std = scaler.transform(X_val) if X_val is not None else None
    if X_val_std is not None: logger.debug("Scaled validation data (KNN).")
    X_test_std = scaler.transform(X_test) if X_test is not None else None
    if X_test_std is not None: logger.debug("Scaled test data (KNN).")
    return X_train_std, X_val_std, X_test_std, scaler

# --- Helper function to find optimal K ---
def find_optimal_k(X_train_std: np.ndarray, y_train: np.ndarray, 
                   X_val_std: np.ndarray, y_val: np.ndarray, 
                   k_range: Tuple[int, int] = (1, 31), step: int = 2) -> Optional[int]:
    """Find the optimal K value using validation set accuracy."""
    logger.info(f"Finding optimal K for KNN in range {k_range}...")
    k_values = range(k_range[0], k_range[1], step)
    val_accuracies = []

    try:
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_std, y_train)
            val_preds = knn.predict(X_val_std)
            accuracy = accuracy_score(y_val, val_preds)
            val_accuracies.append(accuracy)
            logger.debug(f"  K={k}, Validation Accuracy={accuracy:.4f}")

        if not val_accuracies:
             logger.warning("Could not calculate validation accuracies for any K.")
             return None
             
        best_k_index = np.argmax(val_accuracies)
        optimal_k = k_values[best_k_index]
        logger.info(f"Optimal K found: {optimal_k} (Validation Accuracy: {val_accuracies[best_k_index]:.4f})")
        return optimal_k
    except Exception as e:
        logger.exception(f"Error finding optimal K: {e}")
        return None

# --- KNN Model Class ---
class KNNModel(BaseModel):
    """K-Nearest Neighbors (KNN) model implementing the BaseModel interface."""

    @property
    def needs_scaling(self) -> bool:
        return True

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'KNNModel':
        """Fit the KNN model, performing standardization and finding optimal K if validation data is provided."""
        logger.info(f"Fitting KNN model...")
        
        # 1. Standardize features
        try:
            X_train_std, X_val_std, _, self.scaler = _standardize_features(X_train, X_val)
        except Exception as e:
             logger.exception(f"Error during standardization for KNN: {e}")
             raise

        # 2. Determine K value
        k = self.model_params.get('n_neighbors', 5) # Default K
        if X_val_std is not None and y_val is not None:
             # Find optimal k if validation data is present and no specific k is forced
             if 'n_neighbors' not in self.model_params:
                 optimal_k = find_optimal_k(X_train_std, y_train, X_val_std, y_val)
                 if optimal_k:
                     k = optimal_k
                 else:
                      logger.warning("Failed to find optimal K, using default k={k}")
             else:
                  logger.info(f"Using pre-defined k={k} from model_params.")
        else:
             logger.info(f"No validation data provided or k specified. Using default k={k}.")

        # 3. Train final KNN model
        logger.info(f"Training final KNN model with k={k}.")
        try:
            # Get other params like weights, metric from model_params or use defaults
            weights = self.model_params.get('weights', 'uniform')
            metric = self.model_params.get('metric', 'minkowski')
            
            self.model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, n_jobs=-1)
            self.model.fit(X_train_std, y_train)
            logger.info("KNN model training complete.")

        except Exception as e:
            logger.exception(f"Error training final KNN model with k={k}: {e}")
            raise
             
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions. Applies scaling."""
        if self.model is None: raise RuntimeError("Model not trained.")
        if self.scaler is None: raise RuntimeError("Scaler not fitted.")
        
        try:
            X_std = self.scaler.transform(X)
            predictions = self.model.predict(X_std)
            return predictions
        except Exception as e:
             logger.exception(f"Error during prediction: {e}")
             raise

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities. Applies scaling."""
        if self.model is None: raise RuntimeError("Model not trained.")
        if self.scaler is None: raise RuntimeError("Scaler not fitted.")
             
        try:
            X_std = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_std)
            return probabilities
        except Exception as e:
             logger.exception(f"Error during probability prediction: {e}")
             raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model. Applies scaling."""
        if self.model is None: raise RuntimeError("Model not trained.")
        if self.scaler is None: raise RuntimeError("Scaler not fitted.")

        logger.info(f"Evaluating KNN model (k={self.model.n_neighbors})...")
        try:
            X_test_std = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_std)
            y_prob = self.model.predict_proba(X_test_std) # Get probabilities for all classes
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            
            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall,
                       'f1_score': f1, 'confusion_matrix': cm, 'classification_report': report}
            
            # Add AUC and LogLoss (handle binary vs multi-class)
            num_classes = len(np.unique(y_test))
            if num_classes >= 2:
                try:
                     if num_classes == 2:
                          positive_class_idx = np.where(self.model.classes_ == np.max(y_test))[0][0]
                          metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, positive_class_idx])
                     else: # Multi-class
                          metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                     metrics['log_loss'] = log_loss(y_test, y_prob)
                except ValueError as auc_err:
                     logger.warning(f"Could not calculate ROC AUC or LogLoss for KNN: {auc_err}") 
            
            logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            if 'roc_auc' in metrics: logger.info(f" ROC AUC: {metrics['roc_auc']:.4f}")
            elif 'roc_auc_ovr' in metrics: logger.info(f" ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
                
            return metrics
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            return {"error": str(e)}
            
    # KNN does not have intrinsic feature importance
    def plot_feature_importance(self, feature_names: List[str], output_path: str):
        """Feature importance is not applicable for KNN."""
        super().plot_feature_importance(feature_names, output_path) # Calls base method warning 