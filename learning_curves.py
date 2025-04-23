#!/usr/bin/env python
"""
Script to generate learning curves for baseline models
to visualize how model performance changes with training data size.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define paths
OUTPUTS_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "model_comparison_plots")

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    """
    Load and preprocess data.
    Since we don't have direct access to the dataset here, 
    this function should load data from your processed data directory.
    """
    try:
        # For demonstration, we'll create synthetic data
        # In a real-world scenario, you would load your real data here
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic features and labels 
        # (replace this with your actual data loading code)
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=100, 
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=4,
            random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def plot_learning_curve(estimator, X, y, title, save_path, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a learning curve plot for a given model.
    
    Parameters:
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    X : array-like
        Training vector
    y : array-like
        Target values
    title : string
        Title for the chart
    save_path : string
        Path to save the figure
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Number of cross-validations
    n_jobs : int or None, optional
        Number of jobs to run in parallel
    train_sizes : array-like
        Relative or absolute numbers of training examples for learning curve
    """
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate learning curves for all baseline models."""
    # Load data
    X, y = load_data()
    
    if X is None or y is None:
        print("Error: Could not load data.")
        return
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Plot learning curves for each model
    for model_name, model in models.items():
        print(f"Generating learning curve for {model_name}...")
        
        save_path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_learning_curve.png")
        
        try:
            plot_learning_curve(
                model, X, y, 
                title=f"Learning Curve - {model_name}",
                save_path=save_path,
                ylim=(0.0, 1.01),
                cv=5
            )
            print(f"  Learning curve saved to {save_path}")
        except Exception as e:
            print(f"  Error generating learning curve for {model_name}: {str(e)}")
    
    # Create a comparative plot of validation curves
    print("Generating comparative validation curves...")
    plt.figure(figsize=(12, 8))
    
    for model_name, model in models.items():
        try:
            # Calculate learning curve data
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, train_sizes=np.linspace(.1, 1.0, 5))
            
            # Get mean test scores
            test_scores_mean = np.mean(test_scores, axis=1)
            
            # Plot validation curve for this model
            plt.plot(train_sizes, test_scores_mean, 'o-', label=model_name)
            
        except Exception as e:
            print(f"  Error generating comparative curve for {model_name}: {str(e)}")
    
    plt.title("Model Comparison - Validation Curves", fontsize=16)
    plt.xlabel("Training examples", fontsize=14)
    plt.ylabel("Validation Score", fontsize=14)
    plt.ylim(0.0, 1.01)
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    comparative_path = os.path.join(PLOTS_DIR, "model_comparison_validation_curves.png")
    plt.tight_layout()
    plt.savefig(comparative_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparative validation curves saved to {comparative_path}")
    print("All learning curves generated successfully!")

if __name__ == "__main__":
    main() 