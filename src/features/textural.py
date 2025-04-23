"""
Textural feature extraction module for MRI volumes.
"""

import numpy as np
import logging
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from scipy.ndimage import sobel

logger = logging.getLogger(__name__)

def extract_glcm_features(volume, slice_idx=None):
    """
    Extract Gray Level Co-occurrence Matrix (GLCM) features from an MRI volume or slice.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data or 2D slice.
        slice_idx (int, optional): If provided, extract features from this slice index.
        
    Returns:
        dict: Dictionary of GLCM features.
    """
    if volume is None:
        logger.error("Cannot extract GLCM features from None volume")
        return {}
    
    features = {}
    
    # If volume is 3D and slice_idx is not provided, use middle slice
    if len(volume.shape) == 3:
        if slice_idx is None:
            slice_idx = volume.shape[2] // 2
        slice_data = volume[:, :, slice_idx]
    else:
        slice_data = volume
    
    # Normalize and quantize to 8 bits (256 levels)
    if slice_data.max() > 0:
        slice_norm = 255 * (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    else:
        slice_norm = slice_data * 0
    
    slice_uint8 = slice_norm.astype(np.uint8)
    
    # Define GLCM parameters
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Calculate GLCM
    try:
        glcm = graycomatrix(slice_uint8, distances=distances, angles=angles, 
                          levels=256, symmetric=True, normed=True)
        
        # Extract GLCM properties
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        for prop in props:
            prop_values = graycoprops(glcm, prop)
            
            # Average over all directions and distances
            features[f'glcm_{prop}'] = np.mean(prop_values)
            
            # Features for specific distances and angles
            for i, distance in enumerate(distances):
                for j, angle in enumerate(angles):
                    angle_deg = int(angle * 180 / np.pi)
                    features[f'glcm_{prop}_d{distance}_a{angle_deg}'] = prop_values[i, j]
        
        logger.info(f"Extracted {len(features)} GLCM features")
    except Exception as e:
        logger.error(f"Error extracting GLCM features: {str(e)}")
    
    return features

def extract_lbp_features(volume, slice_idx=None, radius=3, n_points=24):
    """
    Extract Local Binary Pattern (LBP) features from an MRI volume or slice.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data or 2D slice.
        slice_idx (int, optional): If provided, extract features from this slice index.
        radius (int, optional): Radius parameter for LBP.
        n_points (int, optional): Number of points for LBP.
        
    Returns:
        dict: Dictionary of LBP features.
    """
    if volume is None:
        logger.error("Cannot extract LBP features from None volume")
        return {}
    
    features = {}
    
    # If volume is 3D and slice_idx is not provided, use middle slice
    if len(volume.shape) == 3:
        if slice_idx is None:
            # Extract features from multiple slices and average
            slices_to_use = [volume.shape[2] // 4, volume.shape[2] // 2, 3 * volume.shape[2] // 4]
            lbp_features_list = []
            
            for idx in slices_to_use:
                slice_data = volume[:, :, idx]
                lbp_features = extract_lbp_features(slice_data, None, radius, n_points)
                lbp_features_list.append(lbp_features)
            
            # Average features across slices
            for key in lbp_features_list[0].keys():
                features[key] = np.mean([feat[key] for feat in lbp_features_list])
            
            return features
        else:
            slice_data = volume[:, :, slice_idx]
    else:
        slice_data = volume
    
    # Normalize slice to [0, 1]
    if slice_data.max() > slice_data.min():
        slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    else:
        slice_norm = slice_data * 0
    
    try:
        # Calculate LBP
        lbp = local_binary_pattern(slice_norm, n_points, radius, method='uniform')
        
        # Create histogram of LBP values
        n_bins = n_points + 2  # For uniform LBP
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Store histogram bins as features
        for i, hist_val in enumerate(hist):
            features[f'lbp_hist_bin_{i}'] = hist_val
        
        # Calculate statistics of LBP image
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)
        features['lbp_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        logger.info(f"Extracted {len(features)} LBP features")
    except Exception as e:
        logger.error(f"Error extracting LBP features: {str(e)}")
    
    return features

def extract_gabor_features(volume, slice_idx=None):
    """
    Extract Gabor filter features from an MRI volume or slice.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data or 2D slice.
        slice_idx (int, optional): If provided, extract features from this slice index.
        
    Returns:
        dict: Dictionary of Gabor filter features.
    """
    if volume is None:
        logger.error("Cannot extract Gabor features from None volume")
        return {}
    
    features = {}
    
    # If volume is 3D and slice_idx is not provided, use middle slice
    if len(volume.shape) == 3:
        if slice_idx is None:
            slice_idx = volume.shape[2] // 2
        slice_data = volume[:, :, slice_idx]
    else:
        slice_data = volume
    
    # Normalize slice to [0, 1]
    if slice_data.max() > slice_data.min():
        slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    else:
        slice_norm = slice_data * 0
    
    try:
        # Define Gabor filter parameters
        frequencies = [0.1, 0.2, 0.3, 0.4]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Apply Gabor filters
        for freq in frequencies:
            for theta in orientations:
                filt_real, filt_imag = gabor(slice_norm, frequency=freq, theta=theta)
                
                # Calculate statistics of filter response
                theta_deg = int(theta * 180 / np.pi)
                
                features[f'gabor_mean_f{freq:.1f}_t{theta_deg}'] = np.mean(filt_real)
                features[f'gabor_std_f{freq:.1f}_t{theta_deg}'] = np.std(filt_real)
                features[f'gabor_max_f{freq:.1f}_t{theta_deg}'] = np.max(filt_real)
                
                # Calculate magnitude response
                magnitude = np.sqrt(filt_real**2 + filt_imag**2)
                features[f'gabor_mag_mean_f{freq:.1f}_t{theta_deg}'] = np.mean(magnitude)
                features[f'gabor_mag_std_f{freq:.1f}_t{theta_deg}'] = np.std(magnitude)
        
        logger.info(f"Extracted {len(features)} Gabor filter features")
    except Exception as e:
        logger.error(f"Error extracting Gabor features: {str(e)}")
    
    return features

def extract_edge_features(volume):
    """
    Extract edge-based features from an MRI volume.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        dict: Dictionary of edge-based features.
    """
    if volume is None:
        logger.error("Cannot extract edge features from None volume")
        return {}
    
    features = {}
    
    try:
        # Apply Sobel edge detector in each dimension
        edge_x = sobel(volume, axis=0)
        edge_y = sobel(volume, axis=1)
        edge_z = sobel(volume, axis=2)
        
        # Calculate edge magnitude
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2 + edge_z**2)
        
        # Calculate edge density (ratio of edge pixels to total pixels)
        edge_threshold = np.percentile(edge_magnitude, 90)  # Threshold at 90th percentile
        edge_mask = edge_magnitude > edge_threshold
        edge_density = np.sum(edge_mask) / edge_magnitude.size
        
        # Store features
        features['edge_density'] = edge_density
        features['edge_magnitude_mean'] = np.mean(edge_magnitude)
        features['edge_magnitude_std'] = np.std(edge_magnitude)
        features['edge_magnitude_max'] = np.max(edge_magnitude)
        
        # Calculate edge direction histograms (rough approximation)
        # Classify each voxel based on the dominant gradient direction
        dominant_x = (np.abs(edge_x) > np.abs(edge_y)) & (np.abs(edge_x) > np.abs(edge_z))
        dominant_y = (np.abs(edge_y) > np.abs(edge_x)) & (np.abs(edge_y) > np.abs(edge_z))
        dominant_z = (np.abs(edge_z) > np.abs(edge_x)) & (np.abs(edge_z) > np.abs(edge_y))
        
        # Calculate ratio of each direction
        features['edge_dominant_x_ratio'] = np.sum(dominant_x) / (np.sum(dominant_x) + np.sum(dominant_y) + np.sum(dominant_z) + 1e-10)
        features['edge_dominant_y_ratio'] = np.sum(dominant_y) / (np.sum(dominant_x) + np.sum(dominant_y) + np.sum(dominant_z) + 1e-10)
        features['edge_dominant_z_ratio'] = np.sum(dominant_z) / (np.sum(dominant_x) + np.sum(dominant_y) + np.sum(dominant_z) + 1e-10)
        
        logger.info(f"Extracted {len(features)} edge-based features")
    except Exception as e:
        logger.error(f"Error extracting edge features: {str(e)}")
    
    return features

def extract_texture_slice_features(volume, slice_indices=None):
    """
    Extract texture features from specific slices of an MRI volume.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        slice_indices (list, optional): List of slice indices to use.
        
    Returns:
        dict: Dictionary of texture features.
    """
    if volume is None:
        logger.error("Cannot extract texture features from None volume")
        return {}
    
    features = {}
    
    # If slice_indices not provided, use default slices
    if slice_indices is None:
        # Use slices at 25%, 50%, and 75% through the volume
        slice_indices = [
            volume.shape[2] // 4,
            volume.shape[2] // 2,
            3 * volume.shape[2] // 4
        ]
    
    # Extract features for each slice and average
    all_slice_features = []
    for slice_idx in slice_indices:
        slice_features = {}
        
        # Extract different texture features
        glcm_feats = extract_glcm_features(volume, slice_idx)
        lbp_feats = extract_lbp_features(volume, slice_idx)
        gabor_feats = extract_gabor_features(volume, slice_idx)
        
        # Combine all slice features
        slice_features.update(glcm_feats)
        slice_features.update(lbp_feats)
        slice_features.update(gabor_feats)
        
        all_slice_features.append(slice_features)
    
    # Average features across slices
    for key in all_slice_features[0].keys():
        features[key] = np.mean([feat[key] for feat in all_slice_features])
    
    logger.info(f"Extracted {len(features)} texture features across {len(slice_indices)} slices")
    return features

def extract_textural_features(volume):
    """
    Extract all textural features from an MRI volume.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        dict: Dictionary of all textural features.
    """
    features = {}
    
    # Extract slice-based texture features
    texture_slice_features = extract_texture_slice_features(volume)
    features.update(texture_slice_features)
    
    # Extract edge features (3D)
    edge_features = extract_edge_features(volume)
    features.update(edge_features)
    
    logger.info(f"Extracted a total of {len(features)} textural features")
    return features

def extract_textural_features_batch(volumes):
    """
    Extract textural features from a batch of MRI volumes.
    
    Args:
        volumes (list): List of 3D MRI volume data.
        
    Returns:
        list: List of feature dictionaries for each volume.
    """
    features_list = []
    
    for i, volume in enumerate(volumes):
        logger.info(f"Extracting textural features for volume {i+1}/{len(volumes)}")
        features = extract_textural_features(volume)
        features_list.append(features)
    
    return features_list 