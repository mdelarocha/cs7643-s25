from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import os
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract Base Class for all baseline models."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize the model."""
        self.model = None
        self.scaler = None # Store scaler if needed
        self.model_params = model_params if model_params else {}
        logger.debug(f"Initializing {self.__class__.__name__} with params: {self.model_params}")

    @property
    @abstractmethod
    def needs_scaling(self) -> bool:
        """Property indicating if the model requires feature scaling."""
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'BaseModel':
        """Train the model. Should handle scaling internally if needs_scaling is True."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data. Should handle scaling internally."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities. Return None if not applicable."""
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data. Should handle scaling internally."""
        pass

    def save(self, file_path: str):
        """Save the trained model and scaler (if applicable) to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_params': self.model_params,
            'class_name': self.__class__.__name__ # Store class name for robust loading
        }
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
            logger.info(f"Saved {self.__class__.__name__} model and scaler to {file_path}")
        except Exception as e:
            logger.exception(f"Error saving model to {file_path}: {e}")

    @classmethod
    def load(cls, file_path: str) -> Optional['BaseModel']:
        """Load a model and scaler from a file."""
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            return None
        try:
            with open(file_path, 'rb') as f:
                load_data = pickle.load(f)
            
            # Basic check if loaded data is a dict and contains the model
            if not isinstance(load_data, dict) or 'model' not in load_data:
                 logger.error(f"Invalid data format in model file: {file_path}")
                 return None

            # Optional: Check class name consistency
            loaded_class_name = load_data.get('class_name')
            if loaded_class_name and loaded_class_name != cls.__name__:
                logger.warning(f"Loading model saved as {loaded_class_name} into an instance of {cls.__name__} from {file_path}")
            
            # Create an instance and load state
            # Ensure model_params are passed correctly during instantiation
            instance = cls(model_params=load_data.get('model_params', {}))
            instance.model = load_data['model']
            instance.scaler = load_data.get('scaler') # Load scaler if exists
            
            logger.info(f"Loaded {cls.__name__} model and scaler from {file_path}")
            return instance
        except Exception as e:
            logger.exception(f"Error loading model from {file_path}: {e}")
            return None

    # Optional: Method for plotting feature importances if applicable
    def plot_feature_importance(self, feature_names: List[str], output_path: str):
        """Plot feature importances if the model supports it."""
        logger.warning(f"Feature importance plotting not implemented for {self.__class__.__name__}")
        # Default implementation: Create an empty file or log a message?
        # For now, just pass
        pass 

    # Optional: Method for plotting clusters if applicable (e.g., for KMeans)
    def plot_clusters(self, X: np.ndarray, y: np.ndarray, output_path: str):
        """
        Plot clusters if the model supports it (e.g., KMeans).
        Should handle scaling internally if necessary.
        """
        logger.warning(f"Cluster plotting not implemented for {self.__class__.__name__}")
        # Default implementation: Create an empty file or log a message?
        # For now, just pass
        pass 

    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray, output_path: str, class_labels: Optional[List[str]] = None):
        """
        Generates and saves a confusion matrix plot with both normalized and count values.

        Args:
            X_test: Test features.
            y_test: True test labels.
            output_path: Path to save the plot image.
            class_labels: Optional list of string names for the classes. If None, uses unique sorted labels from y_test.
        """
        if self.model is None:
            logger.error(f"Cannot plot confusion matrix for {self.__class__.__name__}: model not trained.")
            return

        try:
            y_pred = self.predict(X_test) # Use the model's predict method (handles scaling internally)
            
            # Determine class labels
            if class_labels is None:
                unique_labels = sorted(np.unique(np.concatenate((y_test, y_pred))).tolist())
                class_labels = [str(label) for label in unique_labels] # Default to string representation
                if hasattr(self.model, 'classes_'): # Use actual fitted classes if available
                     try:
                          # Ensure labels match the order of model's internal classes
                          model_classes = self.model.classes_.tolist()
                          # Check if numeric labels match directly
                          if all(isinstance(x, (int, float, np.number)) for x in unique_labels) and \
                             all(isinstance(x, (int, float, np.number)) for x in model_classes) and \
                             sorted(unique_labels) == sorted(model_classes):
                                class_labels = [str(c) for c in model_classes] # Use model's class order
                                logger.debug(f"Using model's fitted classes for labels: {class_labels}")
                          else:
                              logger.warning(f"Mismatch or non-numeric types between y_test/y_pred labels and model.classes_. Using sorted unique labels: {class_labels}")
                     except Exception as e:
                          logger.warning(f"Could not reliably determine class order from model.classes_. Using sorted unique labels. Error: {e}")


            cm = confusion_matrix(y_test, y_pred)
            cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            fig.suptitle(f'Confusion Matrix - {self.__class__.__name__}', fontsize=16)

            # Plot Normalized Confusion Matrix
            sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", ax=axes[0], 
                        xticklabels=class_labels, yticklabels=class_labels, vmin=0, vmax=1)
            axes[0].set_title('Normalized Confusion Matrix')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')

            # Plot Count Confusion Matrix
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                        xticklabels=class_labels, yticklabels=class_labels)
            axes[1].set_title('Count Confusion Matrix')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            
            # Save the plot
            plt.savefig(output_path)
            plt.close(fig) # Close the figure to free memory
            logger.info(f"Saved confusion matrix plot for {self.__class__.__name__} to {output_path}")

        except Exception as e:
            logger.exception(f"Error plotting confusion matrix for {self.__class__.__name__}: {e}")