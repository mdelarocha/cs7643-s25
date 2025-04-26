"""
SVM baseline model for Alzheimer's detection.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, log_loss
)

from ..base_model import BaseModel # Import base class

logger = logging.getLogger(__name__)

# --- Helper function for scaling (Similar to LR, keep separate for now) ---
def _standardize_features(X_train: np.ndarray, X_val: Optional[np.ndarray] = None, X_test: Optional[np.ndarray] = None) \
                         -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
    """Standardize features using StandardScaler. Fits on train, transforms train, val, test."""
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    logger.info(f"Scaled training data (SVM). Mean: {np.mean(X_train_std):.3f}, Std: {np.std(X_train_std):.3f}")
    X_val_std = scaler.transform(X_val) if X_val is not None else None
    if X_val_std is not None: logger.debug("Scaled validation data (SVM).")
    X_test_std = scaler.transform(X_test) if X_test is not None else None
    if X_test_std is not None: logger.debug("Scaled test data (SVM).")
    return X_train_std, X_val_std, X_test_std, scaler

# --- SVM Model Class ---
class SVMModel(BaseModel):
    """Support Vector Machine (SVM) model implementing the BaseModel interface."""

    @property
    def needs_scaling(self) -> bool:
        return True

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'SVMModel':
        """Fit the SVM model, performing standardization and hyperparameter tuning."""
        logger.info(f"Fitting SVM model...")
        
        # 1. Standardize features
        try:
            X_train_std, X_val_std, _, self.scaler = _standardize_features(X_train, X_val)
        except Exception as e:
             logger.exception(f"Error during standardization for SVM: {e}")
             raise

        # 2. Define parameter grid for GridSearchCV
        default_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001, 'scale', 'auto'], # Scale/auto are often good defaults
            'kernel': ['rbf', 'poly', 'linear'] 
        }
        param_grid = self.model_params.get('param_grid', default_params)
        cv = self.model_params.get('cv', 5)
        # Use accuracy or AUC for scoring. AUC requires probability=True
        scoring = self.model_params.get('scoring', 'accuracy') 
        probability_required = scoring == 'roc_auc'

        # 3. Perform GridSearchCV
        logger.info(f"Performing GridSearchCV for SVM (CV={cv}, Scoring={scoring}). Grid: {param_grid}")
        # Enable probability estimates if needed for scoring or later evaluation
        svc = SVC(random_state=42, probability=True, class_weight='balanced') 
        grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        
        try:
            grid_search.fit(X_train_std, y_train)
            self.model = grid_search.best_estimator_
            logger.info(f"GridSearchCV complete. Best Params: {grid_search.best_params_}, Best Score ({scoring}): {grid_search.best_score_:.4f}")
            
            # Optional: Evaluate on validation set
            if X_val_std is not None and y_val is not None:
                val_preds = self.model.predict(X_val_std)
                val_acc = accuracy_score(y_val, val_preds)
                logger.info(f"Validation Accuracy with best model: {val_acc:.4f}")
                
        except Exception as e:
            logger.exception(f"Error during GridSearchCV for SVM: {e}")
            logger.warning("GridSearchCV failed. Fitting SVM with default parameters.")
            try:
                 default_svc = SVC(random_state=42, probability=True, class_weight='balanced') # Simple default
                 default_svc.fit(X_train_std, y_train)
                 self.model = default_svc
            except Exception as e_default:
                 logger.exception(f"Failed to fit even the default SVM: {e_default}")
                 raise

        if self.model is None:
             logger.error("SVM model fitting failed.")
             raise RuntimeError("Model could not be trained.")
             
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions. Applies scaling using the stored scaler."""
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
        
        # Check if probability=True was set during training
        if not getattr(self.model, 'probability', False):
             logger.warning("SVM model was not trained with probability=True. Cannot predict probabilities.")
             return None
             
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

        logger.info(f"Evaluating SVM model...")
        try:
            X_test_std = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_std)
            
            # Calculate base metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            
            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall,
                       'f1_score': f1, 'confusion_matrix': cm, 'classification_report': report}
            
            # Calculate Probabilistic Metrics (AUC, LogLoss) if possible
            if getattr(self.model, 'probability', False):
                try:
                    y_prob = self.model.predict_proba(X_test_std)
                    num_classes = len(np.unique(y_test))
                    if num_classes >= 2:
                         if num_classes == 2:
                              positive_class_idx = np.where(self.model.classes_ == np.max(y_test))[0][0]
                              metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, positive_class_idx])
                         else: # Multi-class
                              metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                         metrics['log_loss'] = log_loss(y_test, y_prob)
                except Exception as prob_err:
                    logger.warning(f"Could not calculate probability-based metrics (AUC, LogLoss) for SVM: {prob_err}")
            else:
                logger.warning("SVM model probability=False, skipping AUC and LogLoss calculation.")

            logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            if 'roc_auc' in metrics: logger.info(f" ROC AUC: {metrics['roc_auc']:.4f}")
            elif 'roc_auc_ovr' in metrics: logger.info(f" ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
                
            return metrics
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            return {"error": str(e)}
            
    # SVM (linear kernel) can have feature importances via coefficients
    # Other kernels (like RBF) don't have direct feature importances in the same way.
    def plot_feature_importance(self, feature_names: List[str], output_path: str):
        """Plot feature importances if the SVM model uses a linear kernel."""
        if self.model is None: raise RuntimeError("Model not trained.")
        
        if getattr(self.model, 'kernel', '') == 'linear' and hasattr(self.model, 'coef_'):
             try:
                 # Similar logic to Logistic Regression for coefficients
                 if self.model.coef_.ndim == 1:
                      importances = self.model.coef_[0]
                 elif self.model.coef_.shape[0] > 1:
                      importances = np.mean(np.abs(self.model.coef_), axis=0)
                 else:
                      logger.error(f"Unexpected coefficient shape for linear SVM: {self.model.coef_.shape}")
                      return
                 
                 if len(importances) != len(feature_names):
                     logger.error(f"Mismatch between SVM coefficients ({len(importances)}) and feature names ({len(feature_names)}).")
                     feature_names_plot = [f"Feature {i}" for i in range(len(importances))]
                 else:
                     feature_names_plot = feature_names

                 importance_df = pd.DataFrame({'feature': feature_names_plot, 'importance': importances})
                 importance_df = importance_df.sort_values('importance', key=abs, ascending=False)

                 top_n = min(20, len(importance_df))
                 plt.figure(figsize=(10, top_n / 2.5))
                 plt.barh(importance_df['feature'][:top_n], importance_df['importance'][:top_n])
                 plt.xlabel("Coefficient Value (Importance)")
                 plt.ylabel("Feature")
                 plt.title(f"Top {top_n} Feature Importances (Linear SVM Coefficients)")
                 plt.gca().invert_yaxis()
                 plt.tight_layout()
                 
                 plt.savefig(output_path)
                 plt.close()
                 logger.info(f"Saved linear SVM feature importance plot to {output_path}")
                 
             except Exception as e:
                  logger.exception(f"Error plotting linear SVM feature importance: {e}")
        else:
            logger.warning(f"Feature importance plotting not available for non-linear SVM kernels (kernel='{getattr(self.model, 'kernel', 'unknown')}').")
            super().plot_feature_importance(feature_names, output_path) # Call base method warning

# --- Remove old standalone functions ---
# train_svm
# evaluate_model
# standardize_features
# plot_decision_boundary (can be added back if needed, perhaps in evaluation utils)
# plot_roc_curve (can be added back if needed)
# plot_confusion_matrix (can be added back if needed) 