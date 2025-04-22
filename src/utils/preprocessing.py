"""
Preprocessing utilities for MRI data.
"""

import numpy as np
import nibabel as nib
import os
import logging
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

def load_mri_volume(file_path):
    """
    Load an MRI volume from a file.
    
    Args:
        file_path (str): Path to the MRI file.
        
    Returns:
        numpy.ndarray: The loaded MRI volume data.
    """
    try:
        if file_path.endswith('.img'):
            # Check if header file exists
            hdr_path = file_path.replace('.img', '.hdr')
            if not os.path.exists(hdr_path):
                logger.warning(f"Header file for {file_path} not found")
                return None
            # Load as Analyze format
            img = nib.AnalyzeImage.load(file_path)
        else:
            # Load as other format (NIfTI, etc.)
            img = nib.load(file_path)
            
        data = img.get_fdata()
        
        # Handle 4D volumes with singleton 4th dimension
        if len(data.shape) == 4 and data.shape[3] == 1:
            data = np.squeeze(data, axis=3)
            logger.info(f"Squeezed 4D volume to 3D: {data.shape}")
            
        return data
    except Exception as e:
        logger.error(f"Error loading MRI volume from {file_path}: {str(e)}")
        return None

def extract_core_slices(volume, num_slices=140):
    """
    Extract the core central slices from an MRI volume.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        num_slices (int, optional): Number of slices to extract.
        
    Returns:
        numpy.ndarray: Extracted core slices.
    """
    if volume is None:
        return None
    
    # Assuming slices are along axis 2 (as in the example)
    if volume.shape[2] > num_slices:
        start = (volume.shape[2] - num_slices) // 2
        end = start + num_slices
        
        # Ensure we don't go out of bounds
        start = max(0, start)
        end = min(volume.shape[2], end)
        
        core_slices = volume[:, :, start:end]
        logger.info(f"Extracted {end-start} core slices from original {volume.shape[2]} slices")
    else:
        # If volume has fewer slices than requested, use all slices
        logger.info(f"Volume has fewer than {num_slices} slices ({volume.shape[2]}), using all available slices")
        core_slices = volume
    
    return core_slices

def normalize_volume(volume):
    """
    Normalize an MRI volume to [0, 1] range.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        numpy.ndarray: Normalized brain volume.
    """
    if volume is None:
        return None
    
    # Find min and max values
    min_val = np.min(volume)
    max_val = np.max(volume)
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(volume)
    
    # Normalize to [0, 1]
    normalized = (volume - min_val) / (max_val - min_val)
    
    return normalized

def resize_volume(volume, target_shape=(176, 208, 176)):
    """
    Resize a volume to a target shape using trilinear interpolation.
    
    This is useful for handling different sized brain scans (176x208 vs 256x256).
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        target_shape (tuple, optional): Target shape to resize to.
        
    Returns:
        numpy.ndarray: Resized volume.
    """
    if volume is None:
        return None
    
    # Calculate zoom factors for each dimension
    zoom_factors = tuple(float(target) / float(source) for target, source in zip(target_shape, volume.shape))
    
    # Resize the volume using scipy's zoom function
    resized = zoom(volume, zoom_factors, order=1)  # order=1 for linear interpolation
    
    logger.info(f"Resized volume from {volume.shape} to {resized.shape}")
    return resized

def extract_2d_slices(volume, axis=2):
    """
    Extract 2D slices from a 3D volume along a specified axis.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        axis (int, optional): Axis along which to extract slices (0: sagittal, 1: coronal, 2: axial).
        
    Returns:
        list: List of 2D slice arrays.
    """
    if volume is None:
        return []
    
    slices = []
    for i in range(volume.shape[axis]):
        if axis == 0:
            slices.append(volume[i, :, :])
        elif axis == 1:
            slices.append(volume[:, i, :])
        else:
            slices.append(volume[:, :, i])
    
    return slices

def analyze_volume_dimensions(volume):
    """
    Analyze the dimensions of a volume and provide useful information.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        dict: Dictionary with dimension information.
    """
    if volume is None:
        return None
    
    info = {
        'shape': volume.shape,
        'size_mb': volume.nbytes / (1024 * 1024),
        'dimension_type': 'unknown'
    }
    
    # Identify common dimension types
    if (volume.shape[0] == 176 and volume.shape[1] == 208) or (volume.shape[0] == 208 and volume.shape[1] == 176):
        info['dimension_type'] = '176x208'
        info['space'] = 'Talairach space (standard)'
    elif volume.shape[0] == 256 and volume.shape[1] == 256:
        info['dimension_type'] = '256x256'
        info['space'] = 'Native acquisition space'
    
    return info

def standardize_volume(volume, target_shape=(176, 208, 176)):
    """
    Standardize a volume by resizing it to a common dimension.
    
    This is useful for handling different sized brain scans, particularly when
    some scans are 256x256 (native acquisition space) and others are 176x208 
    (transformed to Talairach space).
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        target_shape (tuple, optional): Target shape to resize to.
        
    Returns:
        numpy.ndarray: Standardized volume.
    """
    if volume is None:
        return None
    
    # Check if already in target shape
    if volume.shape == target_shape:
        return volume
    
    # Analyze current dimensions
    info = analyze_volume_dimensions(volume)
    logger.info(f"Standardizing volume of type {info['dimension_type']} from {volume.shape} to {target_shape}")
    
    # Resize to target shape
    return resize_volume(volume, target_shape)

def preprocess_mri_file(file_path, normalize=True, extract_core=True, num_core_slices=140, standardize=False):
    """
    Load and preprocess an MRI file.
    
    Args:
        file_path (str): Path to the MRI file.
        normalize (bool, optional): Whether to normalize the volume.
        extract_core (bool, optional): Whether to extract core slices.
        num_core_slices (int, optional): Number of core slices to extract.
        standardize (bool, optional): Whether to standardize dimensions to 176x208x176.
        
    Returns:
        numpy.ndarray: Preprocessed MRI volume.
    """
    # Load the volume
    volume = load_mri_volume(file_path)
    
    if volume is None:
        return None
    
    # Log original dimensions
    logger.info(f"Original volume shape: {volume.shape}")
    
    # Standardize dimensions if requested
    if standardize:
        volume = standardize_volume(volume)
    
    # Extract core slices if requested
    if extract_core:
        volume = extract_core_slices(volume, num_slices=num_core_slices)
    
    # Normalize if requested
    if normalize:
        volume = normalize_volume(volume)
    
    return volume
