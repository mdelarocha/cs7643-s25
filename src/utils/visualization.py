"""
Visualization utilities for MRI data and model results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import nibabel as nib
import logging

logger = logging.getLogger(__name__)

def visualize_mri_slices(volume, slice_indices=None, view='axial', output_file=None):
    """
    Visualize MRI slices from a 3D volume.
    
    Args:
        volume (np.ndarray): 3D MRI volume
        slice_indices (list, optional): List of slice indices to display.
            If None, evenly spaced slices will be selected.
        view (str): Viewing plane - 'axial', 'coronal', or 'sagittal'
        output_file (str, optional): Path to save the visualization.
            If None, the plot will be displayed but not saved.
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Determine which axis to slice along based on the view
    if view == 'axial':
        axis = 2
        title = 'Axial View'
    elif view == 'coronal':
        axis = 1
        title = 'Coronal View'
    elif view == 'sagittal':
        axis = 0
        title = 'Sagittal View'
    else:
        raise ValueError(f"Invalid view: {view}. Must be 'axial', 'coronal', or 'sagittal'")
    
    # Determine the number of slices
    n_slices = volume.shape[axis]
    
    # Select slice indices if not provided
    if slice_indices is None:
        # Choose 6 evenly spaced slices
        slice_indices = np.linspace(0, n_slices - 1, 6, dtype=int)
    
    # Create a figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each slice
    for i, idx in enumerate(slice_indices):
        if i >= len(axes):
            break
        
        # Get the slice
    if axis == 0:
            slice_data = volume[idx, :, :]
    elif axis == 1:
            slice_data = volume[:, idx, :]
    else:
            slice_data = volume[:, :, idx]
        
        # Plot the slice
        axes[i].imshow(slice_data.T, cmap='gray', origin='lower')
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis('off')
    
    # Set figure title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
    
    return fig

def visualize_feature_distributions(features_df, target_col=None, n_features=10, output_file=None):
    """
    Visualize feature distributions and correlations.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        target_col (str, optional): Name of the target column for color-coding.
            If None, no color-coding will be applied.
        n_features (int): Number of features to visualize
        output_file (str, optional): Path to save the visualization.
            If None, the plot will be displayed but not saved.
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Select numeric columns only
    numeric_df = features_df.select_dtypes(include=['number'])
    
    # Remove target column from features if it exists
    if target_col and target_col in numeric_df.columns:
        feature_cols = [col for col in numeric_df.columns if col != target_col]
    else:
        feature_cols = numeric_df.columns.tolist()
    
    # Select top n features (or all if less than n)
    n_features = min(n_features, len(feature_cols))
    selected_features = feature_cols[:n_features]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Feature histograms
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot histograms for each feature
    for feature in selected_features:
        if numeric_df[feature].nunique() > 1:  # Only plot if feature has variation
            sns.kdeplot(numeric_df[feature], label=feature, ax=ax1)
    
    ax1.set_title("Feature Distributions")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.legend(loc='best')
    
    # Feature boxplots
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create a long-form DataFrame for seaborn
    plot_df = pd.melt(numeric_df[selected_features], var_name='Feature', value_name='Value')
    
    # Plot boxplots
    sns.boxplot(x='Feature', y='Value', data=plot_df, ax=ax2)
    ax2.set_title("Feature Boxplots")
    ax2.tick_params(axis='x', rotation=90)
    
    # Feature correlation heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df[selected_features].corr()
    
    # Plot correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    ax3.set_title("Feature Correlations")
    
    # Feature-target relationship (if target is provided)
    ax4 = fig.add_subplot(gs[1, 1])
    
    if target_col and target_col in numeric_df.columns:
        # Plot feature vs. target
        for feature in selected_features[:3]:  # Plot only top 3 features
            if numeric_df[feature].nunique() > 1:  # Only plot if feature has variation
                sns.scatterplot(x=feature, y=target_col, data=numeric_df, label=feature, ax=ax4)
        
        ax4.set_title("Feature-Target Relationships")
        ax4.set_xlabel("Feature Value")
        ax4.set_ylabel("Target Value")
        ax4.legend()
    else:
        # If no target, plot pair plot of top 3 features
        top_features = selected_features[:3]
        sns.scatterplot(x=top_features[0], y=top_features[1], data=numeric_df, ax=ax4)
        ax4.set_title(f"{top_features[0]} vs {top_features[1]}")
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
    
    return fig

def visualize_embeddings(features_df, target_col=None, method="tsne", n_components=2, output_file=None):
    """
    Visualize data embeddings using dimensionality reduction.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        target_col (str, optional): Name of the target column for color-coding.
            If None, no color-coding will be applied.
        method (str): Dimensionality reduction method - 'pca', 'tsne', or 'umap'
        n_components (int): Number of components for dimensionality reduction
        output_file (str, optional): Path to save the visualization.
            If None, the plot will be displayed but not saved.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Select numeric columns only
    numeric_df = features_df.select_dtypes(include=['number'])
    
    # Remove target column from features if it exists
    if target_col and target_col in numeric_df.columns:
        feature_cols = [col for col in numeric_df.columns if col != target_col]
        target = numeric_df[target_col]
    else:
        feature_cols = numeric_df.columns.tolist()
        target = None
    
    # Get feature matrix
    X = numeric_df[feature_cols].values
    
    # Perform dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components)
        method_name = "PCA"
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
        method_name = "t-SNE"
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        method_name = "UMAP"
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'pca', 'tsne', or 'umap'")
    
    # Fit and transform the data
    X_reduced = reducer.fit_transform(X)
    
    # Create a DataFrame for plotting
    if n_components == 2:
        embedding_df = pd.DataFrame({
            'Component 1': X_reduced[:, 0],
            'Component 2': X_reduced[:, 1]
        })
    elif n_components == 3:
        embedding_df = pd.DataFrame({
            'Component 1': X_reduced[:, 0],
            'Component 2': X_reduced[:, 1],
            'Component 3': X_reduced[:, 2]
        })
    else:
        # Only use first 3 components for visualization
        embedding_df = pd.DataFrame({
            'Component 1': X_reduced[:, 0],
            'Component 2': X_reduced[:, 1],
            'Component 3': X_reduced[:, 2] if X_reduced.shape[1] > 2 else np.zeros(X_reduced.shape[0])
        })
    
    # Add target column if available
    if target is not None:
        embedding_df['Target'] = target
    
    # Create figure
    if n_components == 3 and X_reduced.shape[1] >= 3:
        # 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the points
        if target is not None:
            # Color by target
            unique_targets = embedding_df['Target'].unique()
            for value in unique_targets:
                subset = embedding_df[embedding_df['Target'] == value]
                ax.scatter(
                    subset['Component 1'],
                    subset['Component 2'],
                    subset['Component 3'],
                    label=f"Class {value}"
                )
        else:
            # No target, use single color
            ax.scatter(
                embedding_df['Component 1'],
                embedding_df['Component 2'],
                embedding_df['Component 3']
            )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    else:
        # 2D plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the points
        if target is not None:
            # Color by target
            unique_targets = embedding_df['Target'].unique()
            for value in unique_targets:
                subset = embedding_df[embedding_df['Target'] == value]
                ax.scatter(
                    subset['Component 1'],
                    subset['Component 2'],
                    label=f"Class {value}"
                )
            else:
            # No target, use single color
            ax.scatter(
                embedding_df['Component 1'],
                embedding_df['Component 2']
            )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    # Set title and legend
    plt.title(f"{method_name} Embedding")
    if target is not None:
        plt.legend()
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
    
    return fig

def plot_confusion_matrix(cm, classes=None, normalize=False, title=None, cmap=plt.cm.Blues, output_file=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        classes (list, optional): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        title (str, optional): Title for the plot
        cmap (matplotlib.colors.Colormap): Colormap to use
        output_file (str, optional): Path to save the visualization.
            If None, the plot will be displayed but not saved.
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if classes is None:
        if len(cm) == 2:
            classes = ['Negative', 'Positive']
        else:
            classes = [str(i) for i in range(len(cm))]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set classes labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Confusion Matrix")
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
    
    return fig

def plot_roc_curve(fpr, tpr, auc_score, output_file=None):
    """
    Plot an ROC curve.
    
    Args:
        fpr (np.ndarray): False positive rates
        tpr (np.ndarray): True positive rates
        auc_score (float): Area under the curve
        output_file (str, optional): Path to save the visualization.
            If None, the plot will be displayed but not saved.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    
    # Plot the random guess line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
    
    return fig
