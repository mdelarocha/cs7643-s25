"""
Random Forest baseline model for Alzheimer's detection.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, log_loss
)

from ..base_model import BaseModel # Import base class

# Added imports for plotting
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle

logger = logging.getLogger(__name__)

# --- Random Forest Model Class ---
class RandomForestModel(BaseModel):
    """Random Forest model implementing the BaseModel interface."""

    @property
    def needs_scaling(self) -> bool:
        return False # Random Forest is generally less sensitive to feature scaling

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'RandomForestModel':
        """Fit the Random Forest model, optionally performing hyperparameter tuning."""
        logger.info(f"Fitting RandomForest model...")
        
        # No scaling needed for Random Forest generally
        self.scaler = None # Explicitly set scaler to None

        # Define parameter grid/distribution for tuning
        # Use model_params passed during init or defaults
        default_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        # Option to use RandomizedSearchCV for larger spaces
        use_random_search = self.model_params.get('use_random_search', False)
        param_config = self.model_params.get('param_config', default_params)
        n_iter = self.model_params.get('n_iter', 10) # For RandomizedSearch
        cv = self.model_params.get('cv', 5)
        # Default to f1_weighted for multi-class compatibility, allow override via params
        scoring = self.model_params.get('scoring', 'f1_weighted')

        # Perform Hyperparameter Tuning (GridSearch or RandomizedSearch)
        rf = RandomForestClassifier(random_state=42, class_weight='balanced') # Use balanced class weights

        if use_random_search:
            logger.info(f"Performing RandomizedSearchCV for RandomForest (CV={cv}, N_iter={n_iter}, Scoring={scoring}). Dist: {param_config}")
            search = RandomizedSearchCV(rf, param_distributions=param_config, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=42)
        else:
            logger.info(f"Performing GridSearchCV for RandomForest (CV={cv}, Scoring={scoring}). Grid: {param_config}")
            search = GridSearchCV(rf, param_grid=param_config, cv=cv, scoring=scoring, n_jobs=-1)
        
        try:
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            logger.info(f"Search complete. Best Params: {search.best_params_}, Best Score ({scoring}): {search.best_score_:.4f}")
            
            # Optional: Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_preds = self.model.predict(X_val)
                val_acc = accuracy_score(y_val, val_preds)
                logger.info(f"Validation Accuracy with best model: {val_acc:.4f}")
                
        except Exception as e:
            logger.exception(f"Error during hyperparameter search for RandomForest: {e}")
            logger.warning("Search failed. Fitting RandomForest with default parameters.")
            try:
                 # Fallback to default RF
                 default_rf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced') 
                 default_rf.fit(X_train, y_train)
                 self.model = default_rf
            except Exception as e_default:
                 logger.exception(f"Failed to fit even the default RandomForest: {e_default}")
                 raise

        if self.model is None:
             logger.error("RandomForest model fitting failed.")
             raise RuntimeError("Model could not be trained.")
             
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions. Scaling is not applied."""
        if self.model is None: raise RuntimeError("Model not trained.")
        # No scaling check needed
        
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
             logger.exception(f"Error during prediction: {e}")
             raise

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities. Scaling is not applied."""
        if self.model is None: raise RuntimeError("Model not trained.")
        # No scaling check needed

        if not hasattr(self.model, 'predict_proba'): return None # Should always exist for RF
             
        try:
            probabilities = self.model.predict_proba(X)
            return probabilities
        except Exception as e:
             logger.exception(f"Error during probability prediction: {e}")
             raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model. Scaling is not applied."""
        if self.model is None: raise RuntimeError("Model not trained.")
        # No scaling needed

        logger.info(f"Evaluating RandomForest model...")
        try:
            # Get predictions & probabilities
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test) # Get probabilities for all classes
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'classification_report': report,
            }
            
            # Add AUC and LogLoss (handle binary vs multi-class)
            num_classes = len(np.unique(y_test))
            if num_classes >= 2:
                try:
                     if num_classes == 2:
                          # Use probability of the positive class (assuming classes are 0, 1)
                          positive_class_idx = np.where(self.model.classes_ == np.max(y_test))[0][0]
                          roc_auc = roc_auc_score(y_test, y_prob[:, positive_class_idx])
                          metrics['roc_auc'] = roc_auc
                     else: # Multi-class
                          roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                          metrics['roc_auc_ovr'] = roc_auc_ovr
                     
                     # Calculate log loss
                     logloss = log_loss(y_test, y_prob)
                     metrics['log_loss'] = logloss
                except ValueError as auc_err:
                     logger.warning(f"Could not calculate ROC AUC or LogLoss: {auc_err}") 
            
            logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            if 'roc_auc' in metrics: logger.info(f" ROC AUC: {metrics['roc_auc']:.4f}")
            elif 'roc_auc_ovr' in metrics: logger.info(f" ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
                
            return metrics
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            return {"error": str(e)}
            
    def plot_feature_importance(self, feature_names: List[str], output_path: str):
        """Plot feature importances for the RandomForest model."""
        if self.model is None: raise RuntimeError("Model not trained.")
        if not hasattr(self.model, 'feature_importances_'):
             logger.error("Model does not have feature_importances_ attribute.")
             return
            
        try:
            importances = self.model.feature_importances_
            if len(importances) != len(feature_names):
                 logger.error(f"Mismatch between number of importances ({len(importances)}) and feature names ({len(feature_names)}). Using indices instead.")
                 feature_names_plot = [f"Feature {i}" for i in range(len(importances))]
            else:
                feature_names_plot = feature_names
                
            # Create dataframe
            importance_df = pd.DataFrame({'feature': feature_names_plot, 'importance': importances})
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Plot top N features
            top_n = min(20, len(importance_df))
            plt.figure(figsize=(10, top_n / 2.5))
            plt.barh(importance_df['feature'][:top_n], importance_df['importance'][:top_n])
            plt.xlabel("Feature Importance (Gini Importance)")
            plt.ylabel("Feature")
            plt.title(f"Top {top_n} Feature Importances (Random Forest)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved feature importance plot to {output_path}")
            
        except Exception as e:
             logger.exception(f"Error plotting feature importance: {e}") 

    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray, output_path: str):
        """Plot ROC curve for the RandomForest model."""
        if self.model is None or not hasattr(self.model, "predict_proba"):
            logger.error("Cannot plot ROC curve: model not trained or does not support predict_proba.")
            return
        # No scaling check needed for Random Forest

        try:
            n_classes = len(self.model.classes_)
            class_names = [str(c) for c in self.model.classes_]

            if n_classes <= 2: # Binary or pseudo-binary
                fig, ax = plt.subplots(figsize=(8, 6))
                try:
                    # Pass X_test directly as RF doesn't strictly need scaling
                    RocCurveDisplay.from_estimator(self.model, X_test, y_test, ax=ax, name=self.__class__.__name__)
                    ax.set_title(f'ROC Curve - {self.__class__.__name__}')
                    plt.tight_layout()
                    plt.savefig(output_path)
                    plt.close(fig)
                    logger.info(f"Saved ROC curve plot to {output_path}")
                except Exception as e:
                    logger.exception(f"Error generating ROC curve for {self.__class__.__name__}: {e}")
                    plt.close(fig)

            else: # Multi-class case (One-vs-Rest)
                y_test_bin = label_binarize(y_test, classes=self.model.classes_)
                y_score = self.model.predict_proba(X_test)

                fig, ax = plt.subplots(figsize=(8, 6))
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
                for i, color in zip(range(n_classes), colors):
                    try:
                        RocCurveDisplay.from_predictions(
                            y_test_bin[:, i],
                            y_score[:, i],
                            name=f"ROC curve of class {class_names[i]}",
                            color=color,
                            ax=ax,
                        )
                    except Exception as e_cls:
                        logger.warning(f"Could not plot ROC for class {class_names[i]}: {e_cls}")
                
                ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"Multi-class ROC Curve (One-vs-Rest) - {self.__class__.__name__}")
                ax.legend()
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close(fig)
                logger.info(f"Saved multi-class ROC curve plot to {output_path}")

        except Exception as e:
            logger.exception(f"General error plotting ROC curve for {self.__class__.__name__}: {e}")

    def plot_precision_recall_curve(self, X_test: np.ndarray, y_test: np.ndarray, output_path: str):
        """Plot Precision-Recall curve for the RandomForest model."""
        if self.model is None or not hasattr(self.model, "predict_proba"):
            logger.error("Cannot plot Precision-Recall curve: model not trained or does not support predict_proba.")
            return
        # No scaling check needed for Random Forest

        try:
            n_classes = len(self.model.classes_)
            class_names = [str(c) for c in self.model.classes_]

            if n_classes <= 2: # Binary or pseudo-binary
                fig, ax = plt.subplots(figsize=(8, 6))
                try:
                    PrecisionRecallDisplay.from_estimator(self.model, X_test, y_test, ax=ax, name=self.__class__.__name__)
                    ax.set_title(f'Precision-Recall Curve - {self.__class__.__name__}')
                    plt.tight_layout()
                    plt.savefig(output_path)
                    plt.close(fig)
                    logger.info(f"Saved Precision-Recall curve plot to {output_path}")
                except Exception as e:
                    logger.exception(f"Error generating Precision-Recall curve for {self.__class__.__name__}: {e}")
                    plt.close(fig)
            else: # Multi-class case (One-vs-Rest)
                y_test_bin = label_binarize(y_test, classes=self.model.classes_)
                y_score = self.model.predict_proba(X_test)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
                for i, color in zip(range(n_classes), colors):
                    try:
                        PrecisionRecallDisplay.from_predictions(
                            y_test_bin[:, i],
                            y_score[:, i],
                            name=f"PR curve of class {class_names[i]}",
                            color=color,
                            ax=ax,
                        )
                    except Exception as e_cls:
                         logger.warning(f"Could not plot Precision-Recall for class {class_names[i]}: {e_cls}")
                
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"Multi-class Precision-Recall Curve (One-vs-Rest) - {self.__class__.__name__}")
                ax.legend()
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close(fig)
                logger.info(f"Saved multi-class Precision-Recall curve plot to {output_path}")

        except Exception as e:
            logger.exception(f"General error plotting Precision-Recall curve for {self.__class__.__name__}: {e}") 