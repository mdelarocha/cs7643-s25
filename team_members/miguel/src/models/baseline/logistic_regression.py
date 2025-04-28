"""
Logistic Regression baseline model for Alzheimer's detection.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score, log_loss
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Use joblib for potentially more robust serialization than pickle
from typing import Optional, Tuple, Dict, Any, List

# Added imports for plotting
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle

from ..base_model import BaseModel # Import the base class

logger = logging.getLogger(__name__)

# --- Helper function for scaling (could be moved to a common utils if shared) ---
# Keep it here for now as it might be model-specific if different scalers are used
def _standardize_features(X_train: np.ndarray, X_val: Optional[np.ndarray] = None, X_test: Optional[np.ndarray] = None) \
                         -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
    """Standardize features using StandardScaler. Fits on train, transforms train, val, test."""
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    logger.info(f"Scaled training data. Original mean: {np.mean(X_train):.3f}, std: {np.std(X_train):.3f}. Scaled mean: {np.mean(X_train_std):.3f}, std: {np.std(X_train_std):.3f}")
    X_val_std = scaler.transform(X_val) if X_val is not None else None
    if X_val_std is not None: logger.debug("Scaled validation data.")
    X_test_std = scaler.transform(X_test) if X_test is not None else None
    if X_test_std is not None: logger.debug("Scaled test data.")
    return X_train_std, X_val_std, X_test_std, scaler

# --- Logistic Regression Model Class ---
class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementing the BaseModel interface."""

    @property
    def needs_scaling(self) -> bool:
        return True

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'LogisticRegressionModel':
        """Fit the Logistic Regression model, performing standardization and hyperparameter tuning."""
        logger.info(f"Fitting LogisticRegression model...")
        
        # 1. Standardize features
        try:
            X_train_std, X_val_std, _, self.scaler = _standardize_features(X_train, X_val) # Fit scaler
        except Exception as e:
             logger.exception(f"Error during standardization for Logistic Regression: {e}")
             raise # Re-raise exception to indicate fit failure

        # 2. Define parameter grid for GridSearchCV
        # Use model_params passed during init or defaults
        default_params = {
            'penalty': ['l2'], # Default to l2, l1 requires solver='liblinear' or 'saga'
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'], # lbfgs is common default
            'max_iter': [100, 200] # Increase max_iter if convergence issues
        }
        param_grid = self.model_params.get('param_grid', default_params)
        cv = self.model_params.get('cv', 5)
        # Default to f1_weighted for multi-class compatibility, allow override via params
        scoring = self.model_params.get('scoring', 'f1_weighted') 

        # 3. Perform GridSearchCV
        logger.info(f"Performing GridSearchCV for Logistic Regression (CV={cv}, Scoring={scoring}). Grid: {param_grid}")
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        
        try:
            grid_search.fit(X_train_std, y_train)
            self.model = grid_search.best_estimator_
            logger.info(f"GridSearchCV complete. Best Params: {grid_search.best_params_}, Best Score ({scoring}): {grid_search.best_score_:.4f}")
            
            # Optional: Evaluate on validation set if provided
            if X_val_std is not None and y_val is not None:
                val_preds = self.model.predict(X_val_std)
                val_acc = accuracy_score(y_val, val_preds)
                logger.info(f"Validation Accuracy with best model: {val_acc:.4f}")
                
        except Exception as e:
            logger.exception(f"Error during GridSearchCV or validation evaluation for Logistic Regression: {e}")
            # Decide if fit should fail or continue with a default model
            # For now, let's try fitting a default model if GridSearch fails
            logger.warning("GridSearchCV failed. Fitting Logistic Regression with default parameters.")
            try:
                 default_lr = LogisticRegression(random_state=42, max_iter=200) # Simple default
                 default_lr.fit(X_train_std, y_train)
                 self.model = default_lr
            except Exception as e_default:
                 logger.exception(f"Failed to fit even the default Logistic Regression: {e_default}")
                 raise # Re-raise if default also fails

        if self.model is None:
             logger.error("Logistic Regression model fitting failed.")
             raise RuntimeError("Model could not be trained.") # Ensure fit fails clearly
             
        return self # Return the instance

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions. Applies scaling using the stored scaler."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        if self.scaler is None:
             raise RuntimeError("Scaler was not fitted during training. Cannot predict.")
        
        try:
            X_std = self.scaler.transform(X)
            predictions = self.model.predict(X_std)
            return predictions
        except Exception as e:
             logger.exception(f"Error during prediction: {e}")
             # Depending on requirements, return empty array or re-raise
             raise

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities. Applies scaling."""
        if self.model is None: raise RuntimeError("Model not trained.")
        if self.scaler is None: raise RuntimeError("Scaler not fitted.")
        
        # Check if the underlying model supports predict_proba
        if not hasattr(self.model, 'predict_proba'):
             logger.warning("The fitted Logistic Regression model does not support predict_proba.")
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

        logger.info(f"Evaluating LogisticRegression model...")
        try:
            # Scale test data
            X_test_std = self.scaler.transform(X_test)
            
            # Get predictions
            y_pred = self.model.predict(X_test_std)
            
            # Get probabilities (handle potential errors or models without predict_proba)
            y_prob = None
            if hasattr(self.model, "predict_proba"):
                try:
                    y_prob = self.model.predict_proba(X_test_std)[:, 1] # Prob of positive class for AUC
                except Exception as proba_err:
                    logger.warning(f"Could not get prediction probabilities: {proba_err}")
            else:
                 logger.warning("Model does not support predict_proba.")

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
            
            # Add AUC if probabilities are available and it's a binary classification
            num_classes = len(np.unique(y_test))
            if y_prob is not None and num_classes == 2:
                try:
                     roc_auc = roc_auc_score(y_test, y_prob)
                     metrics['roc_auc'] = roc_auc
                     # Calculate log loss as well
                     logloss = log_loss(y_test, y_prob)
                     metrics['log_loss'] = logloss
                except ValueError as auc_err:
                     logger.warning(f"Could not calculate ROC AUC or LogLoss: {auc_err}") 
            elif num_classes > 2:
                # Handle multi-class AUC if needed (requires predict_proba for all classes)
                 try:
                     y_prob_all = self.model.predict_proba(X_test_std)
                     if y_prob_all.shape[1] == num_classes:
                          roc_auc_ovr = roc_auc_score(y_test, y_prob_all, multi_class='ovr', average='weighted')
                          metrics['roc_auc_ovr'] = roc_auc_ovr
                          logloss = log_loss(y_test, y_prob_all)
                          metrics['log_loss'] = logloss
                 except Exception as mc_auc_err:
                      logger.warning(f"Could not calculate multi-class ROC AUC/LogLoss: {mc_auc_err}")
            
            logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            if 'roc_auc' in metrics: logger.info(f" ROC AUC: {metrics['roc_auc']:.4f}")
            elif 'roc_auc_ovr' in metrics: logger.info(f" ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
                
            return metrics
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            return {"error": str(e)}
            
    def plot_feature_importance(self, feature_names: List[str], output_path: str):
        """Plot feature importances for the Logistic Regression model."""
        if self.model is None:
            logger.error("Cannot plot feature importance: model not trained.")
            return
        if not hasattr(self.model, 'coef_'):
            logger.error("Cannot plot feature importance: model does not have coefficients (coef_ attribute).")
            return
            
        try:
            # Get coefficients
            if self.model.coef_.shape[0] == 1: # Binary classification (shape is (1, n_features))
                 importances = self.model.coef_[0]
            elif self.model.coef_.shape[0] > 1: # Multi-class (OvR or Multinomial)
                 # Use the average absolute coefficient across classes as importance proxy
                 importances = np.mean(np.abs(self.model.coef_), axis=0)
                 logger.info("Using mean absolute coefficients across classes for multi-class feature importance.")
            else:
                 logger.error(f"Unexpected coefficient shape: {self.model.coef_.shape}")
                 return

            if len(importances) != len(feature_names):
                 logger.error(f"Mismatch between number of coefficients ({len(importances)}) and feature names ({len(feature_names)}). Using indices instead.")
                 feature_names_plot = [f"Feature {i}" for i in range(len(importances))]
            else:
                feature_names_plot = feature_names
                
            # Create dataframe for plotting
            importance_df = pd.DataFrame({'feature': feature_names_plot, 'importance': importances})
            importance_df = importance_df.sort_values('importance', key=abs, ascending=False)
            
            # Plot top N features (e.g., top 20)
            top_n = min(20, len(importance_df))
            plt.figure(figsize=(10, top_n / 2.5))
            plt.barh(importance_df['feature'][:top_n], importance_df['importance'][:top_n])
            plt.xlabel("Coefficient Value (Importance)")
            plt.ylabel("Feature")
            plt.title(f"Top {top_n} Feature Importances (Logistic Regression Coefficients)")
            plt.gca().invert_yaxis() # Display most important at top
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved feature importance plot to {output_path}")
            
        except Exception as e:
             logger.exception(f"Error plotting feature importance: {e}")

    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray, output_path: str):
        """Plot ROC curve for the Logistic Regression model."""
        if self.model is None or not hasattr(self.model, "predict_proba"):
            logger.error("Cannot plot ROC curve: model not trained or does not support predict_proba.")
            return
        if self.scaler is None:
            logger.error("Cannot plot ROC curve: scaler not available.")
            return

        try:
            X_test_std = self.scaler.transform(X_test)
            n_classes = len(self.model.classes_)
            class_names = [str(c) for c in self.model.classes_]

            if n_classes <= 2: # Binary or pseudo-binary
                fig, ax = plt.subplots(figsize=(8, 6))
                try:
                    RocCurveDisplay.from_estimator(self.model, X_test_std, y_test, ax=ax, name=self.__class__.__name__)
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
                y_score = self.model.predict_proba(X_test_std)

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
        """Plot Precision-Recall curve for the Logistic Regression model."""
        if self.model is None or not hasattr(self.model, "predict_proba"):
            logger.error("Cannot plot Precision-Recall curve: model not trained or does not support predict_proba.")
            return
        if self.scaler is None:
            logger.error("Cannot plot Precision-Recall curve: scaler not available.")
            return

        try:
            X_test_std = self.scaler.transform(X_test)
            n_classes = len(self.model.classes_)
            class_names = [str(c) for c in self.model.classes_]

            if n_classes <= 2: # Binary or pseudo-binary
                fig, ax = plt.subplots(figsize=(8, 6))
                try:
                    PrecisionRecallDisplay.from_estimator(self.model, X_test_std, y_test, ax=ax, name=self.__class__.__name__)
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
                y_score = self.model.predict_proba(X_test_std)
                
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

def plot_roc_curve(model, X_test, y_test, output_path=None):
    """
    Plot ROC curve for a binary logistic regression model.
    
    Args:
        model (LogisticRegression): Trained model.
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
    Plot confusion matrix for a logistic regression model.
    
    Args:
        model (LogisticRegression): Trained model.
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

def plot_feature_importance(model, feature_names, output_path=None):
    """
    Plot feature importance for logistic regression model.
    
    Args:
        model (LogisticRegression): Trained logistic regression model.
        feature_names (list): Names of features.
        output_path (str, optional): Path to save the plot. If None, the plot is displayed.
        
    Returns:
        matplotlib.figure.Figure: The figure object or None if plotting fails.
    """
    try:
        if not feature_names or len(feature_names) == 0:
            logger.warning("No feature names provided for plotting")
            return None
            
        if hasattr(model, 'coef_'):
            # For multi-class, average the coefficients across classes
            if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
                # Multi-class case
                coefs = np.abs(model.coef_).mean(axis=0)
            else:
                # Binary case
                coefs = np.abs(model.coef_[0])
                
            # Make sure we have the right number of features
            if len(coefs) == len(feature_names):
                # Sort by importance
                indices = np.argsort(coefs)[::-1]
                sorted_feature_names = [feature_names[i] for i in indices]
                sorted_coefs = coefs[indices]
                
                # Get top 20 features or all if less than 20
                top_n = min(20, len(sorted_feature_names))
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                plt.barh(range(top_n), sorted_coefs[:top_n], align='center')
                plt.yticks(range(top_n), sorted_feature_names[:top_n])
                plt.xlabel('Mean Absolute Coefficient')
                plt.ylabel('Feature')
                plt.title('Logistic Regression Feature Importance')
                plt.tight_layout()
                
                # Save or show
                if output_path:
                    plt.savefig(output_path)
                    logger.info(f"Saved feature importance plot to {output_path}")
                    plt.close()
                    return None
                else:
                    return plt.gcf()
            else:
                logger.warning(f"Number of coefficients ({len(coefs)}) doesn't match number of feature names ({len(feature_names)})")
                return None
        else:
            logger.warning("Model doesn't have coefficients for plotting")
            return None
            
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        return None 