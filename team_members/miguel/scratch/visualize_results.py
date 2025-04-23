#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization tool for MRI data and model results.
"""

import os
import glob
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from joblib import load
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DIR = "data/processed"
FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"
VISUALIZATIONS_DIR = "visualizations"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize MRI data and model results")
    parser.add_argument(
        "--subject_id",
        type=str,
        help="Subject ID to visualize (for MRI slices visualization)",
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default=None,
        help="Path to the features CSV file (if None, the most recent file will be used)",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Path to the model file (if None, the most recent file will be used)",
    )
    parser.add_argument(
        "--visualize_mri",
        action="store_true",
        help="Visualize MRI slices",
    )
    parser.add_argument(
        "--visualize_features",
        action="store_true",
        help="Visualize feature distributions and correlations",
    )
    parser.add_argument(
        "--visualize_embeddings",
        action="store_true",
        help="Visualize data embeddings using dimensionality reduction",
    )
    parser.add_argument(
        "--visualize_results",
        action="store_true",
        help="Visualize model results",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of components for dimensionality reduction",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=["pca", "tsne", "umap"],
        help="Dimensionality reduction method",
    )
    
    return parser.parse_args()


def get_latest_file(directory, pattern):
    """
    Get the most recent file matching a pattern in a directory.
    
    Args:
        directory (str): Directory to search in
        pattern (str): Glob pattern to match files
    
    Returns:
        str: Path to the most recent file
    """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {directory}")
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_features(feature_file=None):
    """
    Load features from a CSV file.
    
    Args:
        feature_file (str, optional): Path to the features CSV file.
            If None, the most recent file will be used.
    
    Returns:
        pd.DataFrame: DataFrame containing the features
    """
    if feature_file is None:
        feature_file = get_latest_file(FEATURES_DIR, "mri_features_*.csv")
        logger.info(f"Using most recent feature file: {feature_file}")
    
    # Load features
    features_df = pd.read_csv(feature_file)
    logger.info(f"Loaded features for {len(features_df)} subjects with {features_df.shape[1]} features")
    
    return features_df


def load_model_and_metadata(model_file=None):
    """
    Load model and associated metadata.
    
    Args:
        model_file (str, optional): Path to the model file.
            If None, the most recent file will be used.
    
    Returns:
        tuple: (model, scaler, selected_features)
    """
    if model_file is None:
        model_file = get_latest_file(MODELS_DIR, "*.joblib")
        logger.info(f"Using most recent model file: {model_file}")
    
    # Load model
    model = load(model_file)
    logger.info(f"Loaded model: {type(model).__name__}")
    
    # Try to load scaler
    timestamp = os.path.basename(model_file).split("_")[1].split(".")[0]
    scaler_file = os.path.join(MODELS_DIR, f"scaler_{timestamp}.joblib")
    
    try:
        scaler = load(scaler_file)
        logger.info("Loaded feature scaler")
    except FileNotFoundError:
        logger.warning("Feature scaler not found")
        scaler = None
    
    # Try to load selected features
    selected_features_file = os.path.join(MODELS_DIR, f"selected_features_{timestamp}.txt")
    
    try:
        with open(selected_features_file, "r") as f:
            selected_features = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(selected_features)} selected features")
    except FileNotFoundError:
        logger.warning("Selected features file not found")
        selected_features = None
    
    return model, scaler, selected_features


def load_mri_volume(subject_id):
    """
    Load MRI volume for a specific subject.
    
    Args:
        subject_id (str): Subject ID
    
    Returns:
        np.ndarray: MRI volume
    """
    # Look for preprocessed MRI files
    processed_file = os.path.join(PROCESSED_DIR, f"{subject_id}_processed.nii.gz")
    
    if os.path.exists(processed_file):
        # Load preprocessed MRI volume
        mri_img = nib.load(processed_file)
        volume = mri_img.get_fdata()
        logger.info(f"Loaded preprocessed MRI volume for subject {subject_id}")
    else:
        # Look for raw MRI files
        subject_dir = os.path.join("data/raw", subject_id)
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"No directory found for subject {subject_id}")
        
        mri_files = glob.glob(os.path.join(subject_dir, "*.nii*"))
        if not mri_files:
            raise FileNotFoundError(f"No MRI files found for subject {subject_id}")
        
        # Load the first MRI file
        mri_img = nib.load(mri_files[0])
        volume = mri_img.get_fdata()
        logger.info(f"Loaded raw MRI volume for subject {subject_id}")
    
    return volume


def visualize_mri_slices(subject_id, output_dir=VISUALIZATIONS_DIR):
    """
    Visualize slices of an MRI volume.
    
    Args:
        subject_id (str): Subject ID
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MRI volume
    volume = load_mri_volume(subject_id)
    
    # Get central slices for each axis
    x_center = volume.shape[0] // 2
    y_center = volume.shape[1] // 2
    z_center = volume.shape[2] // 2
    
    # Create slices for each axis
    sagittal_slice = volume[x_center, :, :]  # X axis
    coronal_slice = volume[:, y_center, :]   # Y axis
    axial_slice = volume[:, :, z_center]     # Z axis
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot slices
    axes[0].imshow(sagittal_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"Sagittal Slice (X={x_center})")
    axes[0].set_xlabel("Y axis")
    axes[0].set_ylabel("Z axis")
    
    axes[1].imshow(coronal_slice.T, cmap="gray", origin="lower")
    axes[1].set_title(f"Coronal Slice (Y={y_center})")
    axes[1].set_xlabel("X axis")
    axes[1].set_ylabel("Z axis")
    
    axes[2].imshow(axial_slice, cmap="gray", origin="lower")
    axes[2].set_title(f"Axial Slice (Z={z_center})")
    axes[2].set_xlabel("X axis")
    axes[2].set_ylabel("Y axis")
    
    plt.tight_layout()
    plt.suptitle(f"MRI Slices for Subject {subject_id}", y=1.05)
    
    # Save figure
    output_file = os.path.join(output_dir, f"mri_slices_{subject_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved MRI slices visualization to {output_file}")
    
    # Create a 3D montage of slices
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 5, figure=fig)
    
    # Sagittal slices
    slice_positions = np.linspace(0, volume.shape[0]-1, 5, dtype=int)
    for i, pos in enumerate(slice_positions):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(volume[pos, :, :].T, cmap="gray", origin="lower")
        ax.set_title(f"X={pos}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Coronal slices
    slice_positions = np.linspace(0, volume.shape[1]-1, 5, dtype=int)
    for i, pos in enumerate(slice_positions):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(volume[:, pos, :].T, cmap="gray", origin="lower")
        ax.set_title(f"Y={pos}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Axial slices
    slice_positions = np.linspace(0, volume.shape[2]-1, 5, dtype=int)
    for i, pos in enumerate(slice_positions):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(volume[:, :, pos], cmap="gray", origin="lower")
        ax.set_title(f"Z={pos}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.suptitle(f"MRI Slice Montage for Subject {subject_id}", y=0.98)
    
    # Save montage
    output_file = os.path.join(output_dir, f"mri_montage_{subject_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved MRI montage visualization to {output_file}")


def visualize_feature_distributions(features_df, output_dir=VISUALIZATIONS_DIR):
    """
    Visualize distributions of features by dementia status.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate metadata from features
    metadata_cols = ["subject_id", "cdr_score", "has_dementia"]
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    # Choose top features based on correlation with has_dementia
    correlations = []
    for col in feature_cols:
        corr = np.abs(np.corrcoef(features_df[col], features_df["has_dementia"])[0, 1])
        correlations.append((col, corr))
    
    # Sort by correlation magnitude
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 10 features
    top_features = [corr[0] for corr in correlations[:10]]
    
    # Create a figure for each top feature
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        
        # Create distribution plot by dementia status
        sns.histplot(
            data=features_df,
            x=feature,
            hue="has_dementia",
            kde=True,
            element="step",
            palette=["green", "red"],
            bins=20,
        )
        
        plt.title(f"Distribution of {feature} by Dementia Status")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.legend(["No Dementia", "Dementia"])
        
        # Save figure
        output_file = os.path.join(output_dir, f"feature_dist_{feature}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
    
    logger.info(f"Saved feature distribution visualizations to {output_dir}")
    
    # Create correlation matrix visualization
    plt.figure(figsize=(12, 10))
    corr_matrix = features_df[top_features + ["has_dementia"]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title("Correlation Matrix of Top Features")
    plt.tight_layout()
    
    # Save correlation matrix
    output_file = os.path.join(output_dir, "feature_correlation_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation matrix to {output_file}")
    
    # Create boxplots for top features
    plt.figure(figsize=(15, 10))
    data_to_plot = features_df.melt(
        id_vars=["has_dementia"],
        value_vars=top_features,
        var_name="Feature",
        value_name="Value"
    )
    
    sns.boxplot(
        data=data_to_plot,
        x="Feature",
        y="Value",
        hue="has_dementia",
        palette=["green", "red"],
    )
    
    plt.title("Feature Distributions by Dementia Status")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["No Dementia", "Dementia"])
    plt.tight_layout()
    
    # Save boxplots
    output_file = os.path.join(output_dir, "feature_boxplots.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature boxplots to {output_file}")


def visualize_embeddings(features_df, method="tsne", n_components=2, output_dir=VISUALIZATIONS_DIR):
    """
    Visualize data embeddings using dimensionality reduction.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        method (str): Dimensionality reduction method ('pca', 'tsne', or 'umap')
        n_components (int): Number of components for dimensionality reduction
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate metadata from features
    metadata_cols = ["subject_id", "cdr_score", "has_dementia"]
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    # Prepare data
    X = features_df[feature_cols].values
    y = features_df["has_dementia"].values
    
    # Apply dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=n_components)
        embedding_name = "PCA"
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
        embedding_name = "t-SNE"
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedding_name = "UMAP"
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Fit and transform data
    X_embedded = reducer.fit_transform(X)
    
    # Create DataFrame with embeddings
    embedding_df = pd.DataFrame(X_embedded, columns=[f"Component {i+1}" for i in range(n_components)])
    embedding_df["has_dementia"] = y
    embedding_df["cdr_score"] = features_df["cdr_score"].values
    
    # Plot embeddings by dementia status
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        # 2D scatterplot
        sns.scatterplot(
            data=embedding_df,
            x="Component 1",
            y="Component 2",
            hue="has_dementia",
            style="has_dementia",
            palette=["green", "red"],
            s=100,
            alpha=0.8,
        )
        
        plt.title(f"{embedding_name} Embedding of MRI Features by Dementia Status")
        plt.legend(["No Dementia", "Dementia"])
        
    elif n_components == 3:
        # 3D scatterplot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        # Plot points with dementia
        dementia_mask = embedding_df["has_dementia"] == 1
        ax.scatter(
            embedding_df.loc[dementia_mask, "Component 1"],
            embedding_df.loc[dementia_mask, "Component 2"],
            embedding_df.loc[dementia_mask, "Component 3"],
            c="red",
            label="Dementia",
            s=100,
            alpha=0.8,
        )
        
        # Plot points without dementia
        ax.scatter(
            embedding_df.loc[~dementia_mask, "Component 1"],
            embedding_df.loc[~dementia_mask, "Component 2"],
            embedding_df.loc[~dementia_mask, "Component 3"],
            c="green",
            label="No Dementia",
            s=100,
            alpha=0.8,
        )
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(f"{embedding_name} Embedding of MRI Features by Dementia Status")
        ax.legend()
    
    # Save embedding visualization
    output_file = os.path.join(output_dir, f"{method}_embedding.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {embedding_name} embedding visualization to {output_file}")
    
    # Plot embeddings by CDR score
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        # 2D scatterplot
        sns.scatterplot(
            data=embedding_df,
            x="Component 1",
            y="Component 2",
            hue="cdr_score",
            s=100,
            alpha=0.8,
            palette="viridis",
        )
        
        plt.title(f"{embedding_name} Embedding of MRI Features by CDR Score")
        
    elif n_components == 3:
        # 3D scatterplot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        # Get unique CDR scores
        cdr_scores = embedding_df["cdr_score"].unique()
        
        # Create a colormap
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(cdr_scores)))
        
        # Plot points for each CDR score
        for i, cdr in enumerate(cdr_scores):
            mask = embedding_df["cdr_score"] == cdr
            ax.scatter(
                embedding_df.loc[mask, "Component 1"],
                embedding_df.loc[mask, "Component 2"],
                embedding_df.loc[mask, "Component 3"],
                c=[colors[i]],
                label=f"CDR = {cdr}",
                s=100,
                alpha=0.8,
            )
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(f"{embedding_name} Embedding of MRI Features by CDR Score")
        ax.legend()
    
    # Save embedding visualization
    output_file = os.path.join(output_dir, f"{method}_embedding_by_cdr.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {embedding_name} embedding by CDR score visualization to {output_file}")


def visualize_model_results(output_dir=RESULTS_DIR):
    """
    Visualize model evaluation results.
    
    Args:
        output_dir (str): Directory to save visualizations
    """
    # Find metrics files
    metrics_files = glob.glob(os.path.join(output_dir, "metrics_*.csv"))
    
    if not metrics_files:
        logger.warning("No metrics files found")
        return
    
    # Create DataFrame with metrics from all runs
    metrics_dfs = []
    for file in metrics_files:
        df = pd.read_csv(file)
        # Extract model type from filename
        filename = os.path.basename(file)
        # Add filename as a column
        df["filename"] = filename
        metrics_dfs.append(df)
    
    all_metrics = pd.concat(metrics_dfs, ignore_index=True)
    
    # Create a parallel coordinates plot to compare models
    plt.figure(figsize=(12, 8))
    
    # Format metrics columns for the plot
    metrics_cols = ["accuracy", "precision", "recall", "f1"]
    for col in metrics_cols:
        all_metrics[col] = all_metrics[col].astype(float)
    
    # Create the parallel coordinates plot
    pd.plotting.parallel_coordinates(
        all_metrics,
        "filename",
        cols=metrics_cols,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(all_metrics)],
    )
    
    plt.title("Comparison of Model Performance Metrics")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison visualization to {output_file}")
    
    # Find feature importance files
    importance_files = glob.glob(os.path.join(output_dir, "feature_importances_*.csv"))
    
    if importance_files:
        # Use the most recent file
        importance_file = sorted(importance_files, key=os.path.getmtime)[-1]
        importance_df = pd.read_csv(importance_file)
        
        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        # Take top 20 features
        top_features = importance_df.head(20)
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x="importance", y="feature", palette="viridis")
        plt.title("Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, "top_feature_importances.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved feature importance visualization to {output_file}")
    else:
        logger.warning("No feature importance files found")


def main():
    """Main function to visualize MRI data and model results."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Load features
    features_df = load_features(args.feature_file)
    
    # Visualize MRI slices
    if args.visualize_mri:
        if args.subject_id:
            visualize_mri_slices(args.subject_id)
        else:
            # Use the first subject in the features DataFrame
            subject_id = features_df["subject_id"].iloc[0]
            logger.info(f"No subject ID provided, using first subject: {subject_id}")
            visualize_mri_slices(subject_id)
    
    # Visualize feature distributions
    if args.visualize_features:
        visualize_feature_distributions(features_df)
    
    # Visualize embeddings
    if args.visualize_embeddings:
        visualize_embeddings(
            features_df,
            method=args.method,
            n_components=args.n_components
        )
    
    # Visualize model results
    if args.visualize_results:
        visualize_model_results()
    
    logger.info("Visualization process completed")


if __name__ == "__main__":
    main() 