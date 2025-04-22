"""
Visualization utilities for brain MRI data.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_brain_slice(volume, slice_idx=None, axis=2, figsize=(10, 10)):
    """
    Plot a single slice from a 3D brain scan.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        slice_idx (int, optional): Index of the slice to plot. If None, plots the middle slice.
        axis (int, optional): Axis along which to slice (0: sagittal, 1: coronal, 2: axial).
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if axis == 0:
        brain_slice = volume[slice_idx, :, :]
        title = f"Sagittal Slice {slice_idx}"
    elif axis == 1:
        brain_slice = volume[:, slice_idx, :]
        title = f"Coronal Slice {slice_idx}"
    else:
        brain_slice = volume[:, :, slice_idx]
        title = f"Axial Slice {slice_idx}"
    
    ax.imshow(brain_slice.T, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def plot_brain_slices(volume, n_slices=9, axis=2, figsize=(15, 15)):
    """
    Plot multiple slices from a 3D brain scan.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        n_slices (int, optional): Number of slices to plot.
        axis (int, optional): Axis along which to slice (0: sagittal, 1: coronal, 2: axial).
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Calculate slice indices
    total_slices = volume.shape[axis]
    step = total_slices // (n_slices + 1)
    slice_indices = list(range(step, total_slices, step))[:n_slices]
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_slices)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each slice
    for i, slice_idx in enumerate(slice_indices):
        if i < len(axes):
            if axis == 0:
                brain_slice = volume[slice_idx, :, :]
                title = f"Sagittal {slice_idx}"
            elif axis == 1:
                brain_slice = volume[:, slice_idx, :]
                title = f"Coronal {slice_idx}"
            else:
                brain_slice = volume[:, :, slice_idx]
                title = f"Axial {slice_idx}"
            
            axes[i].imshow(brain_slice.T, cmap="gray", origin="lower")
            axes[i].set_title(title)
            axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig

def plot_brain_three_plane(volume, slice_indices=None, figsize=(15, 5)):
    """
    Plot a slice from each of the three anatomical planes (axial, coronal, sagittal).
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        slice_indices (list, optional): List of 3 indices for [axial, coronal, sagittal] slices.
                                       If None, uses the middle slice for each plane.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    if slice_indices is None:
        slice_indices = [shape // 2 for shape in volume.shape]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Axial slice (top-down view)
    axes[0].imshow(volume[:, :, slice_indices[2]].T, cmap="gray", origin="lower")
    axes[0].set_title(f"Axial Slice {slice_indices[2]}")
    axes[0].axis('off')
    
    # Coronal slice (front view)
    axes[1].imshow(volume[:, slice_indices[1], :].T, cmap="gray", origin="lower")
    axes[1].set_title(f"Coronal Slice {slice_indices[1]}")
    axes[1].axis('off')
    
    # Sagittal slice (side view)
    axes[2].imshow(volume[slice_indices[0], :, :].T, cmap="gray", origin="lower")
    axes[2].set_title(f"Sagittal Slice {slice_indices[0]}")
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig
