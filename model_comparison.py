#!/usr/bin/env python
"""
Script to compare performance across all baseline models.
Generates confusion matrices and comparative performance metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

# Define paths
OUTPUTS_DIR = "outputs"
RANDOM_FOREST_PATH = os.path.join(OUTPUTS_DIR, "baseline_pipeline_20250423_174804/random_forest_metrics.json")
LOGISTIC_REGRESSION_PATH = os.path.join(OUTPUTS_DIR, "baseline_pipeline_20250423_174804/logistic_regression_metrics.json")
SVM_PATH = os.path.join(OUTPUTS_DIR, "additional_models_20250423_180253/baseline_pipeline_20250423_180254/svm_metrics.json")
KNN_PATH = os.path.join(OUTPUTS_DIR, "additional_models_20250423_180253/baseline_pipeline_20250423_180254/knn_metrics.json")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "model_comparison_plots")

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_metrics(file_path):
    """Load model metrics from JSON file."""
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading metrics from {file_path}: {str(e)}")
        return None

def plot_confusion_matrix(cm, title, save_path):
    """Plot a confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Custom colormap (white to blue)
    cmap = LinearSegmentedColormap.from_list('blue_gradient', ['white', '#5788c9'])
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                cbar=True, square=True, linewidths=.5)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.yticks(rotation=0)
    
    # Add class labels
    labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    plt.xticks(np.arange(len(labels)) + 0.5, labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_models_bar_chart(metrics_dict, save_path):
    """Create a bar chart comparing model performances."""
    plt.figure(figsize=(12, 8))
    
    # Extract metrics
    model_names = list(metrics_dict.keys())
    accuracy = [metrics_dict[model].get('accuracy', 0) for model in model_names]
    
    # Get class-wise F1 scores
    class_f1 = {
        'Class 0': [metrics_dict[model]['classification_report'].get('0', {}).get('f1-score', 0) 
                   for model in model_names],
        'Class 1': [metrics_dict[model]['classification_report'].get('1', {}).get('f1-score', 0) 
                   for model in model_names],
        'Class 2': [metrics_dict[model]['classification_report'].get('2', {}).get('f1-score', 0) 
                   for model in model_names],
        'Class 3': [metrics_dict[model]['classification_report'].get('3', {}).get('f1-score', 0) 
                   for model in model_names]
    }
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy,
        'F1 (Class 0)': class_f1['Class 0'],
        'F1 (Class 1)': class_f1['Class 1'],
        'F1 (Class 2)': class_f1['Class 2'],
        'F1 (Class 3)': class_f1['Class 3']
    })
    
    # Plotting
    df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    # Use seaborn for a nicer plot
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted)
    
    # Formatting
    plt.title('Performance Comparison Across Models', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate comparison plots."""
    # Load metrics for each model
    random_forest_metrics = load_metrics(RANDOM_FOREST_PATH)
    logistic_regression_metrics = load_metrics(LOGISTIC_REGRESSION_PATH)
    svm_metrics = load_metrics(SVM_PATH)
    knn_metrics = load_metrics(KNN_PATH)
    
    # Collect all metrics in a dictionary
    metrics_dict = {
        'Random Forest': random_forest_metrics,
        'Logistic Regression': logistic_regression_metrics,
        'SVM': svm_metrics,
        'KNN': knn_metrics
    }
    
    # Filter out None values (failed to load)
    metrics_dict = {k: v for k, v in metrics_dict.items() if v is not None}
    
    # Plot confusion matrices
    for model_name, metrics in metrics_dict.items():
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            title = f'Confusion Matrix - {model_name}'
            save_path = os.path.join(PLOTS_DIR, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
            plot_confusion_matrix(cm, title, save_path)
    
    # Create bar chart comparing model performances
    compare_models_bar_chart(metrics_dict, os.path.join(PLOTS_DIR, 'model_performance_comparison.png'))
    
    # Print summary of model performances
    print("\n===== MODEL PERFORMANCE COMPARISON =====")
    model_accuracies = [(model, metrics.get('accuracy', 0)) for model, metrics in metrics_dict.items()]
    model_accuracies.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy
    
    for model, accuracy in model_accuracies:
        print(f"{model}: Accuracy = {accuracy:.4f}")
        
        # Print class-wise F1 scores
        for class_id in ['0', '1', '2', '3']:
            f1 = metrics_dict[model]['classification_report'].get(class_id, {}).get('f1-score', 0)
            print(f"  - Class {class_id} F1-score: {f1:.4f}")
    
if __name__ == "__main__":
    main() 