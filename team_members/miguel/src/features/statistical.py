"""
Statistical feature extraction module for MRI volumes.
"""

import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def extract_intensity_features(volume):
    """
    Extract intensity-based features from an MRI volume.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        dict: Dictionary of intensity-based features.
    """
    if volume is None:
        logger.error("Cannot extract features from None volume")
        return {}
    
    features = {}
    
    # Basic statistics
    features['mean_intensity'] = np.mean(volume)
    features['median_intensity'] = np.median(volume)
    features['std_intensity'] = np.std(volume)
    features['min_intensity'] = np.min(volume)
    features['max_intensity'] = np.max(volume)
    features['intensity_range'] = features['max_intensity'] - features['min_intensity']
    
    # Histogram statistics
    hist, bin_edges = np.histogram(volume.flatten(), bins=50, density=True)
    features['histogram_skewness'] = stats.skew(hist)
    features['histogram_kurtosis'] = stats.kurtosis(hist)
    
    # Percentile-based features
    features['intensity_25th_percentile'] = np.percentile(volume, 25)
    features['intensity_75th_percentile'] = np.percentile(volume, 75)
    features['intensity_90th_percentile'] = np.percentile(volume, 90)
    features['interquartile_range'] = features['intensity_75th_percentile'] - features['intensity_25th_percentile']
    
    # Count-based features
    features['volume_non_zero_voxels'] = np.count_nonzero(volume)
    features['volume_total_voxels'] = volume.size
    features['volume_non_zero_ratio'] = features['volume_non_zero_voxels'] / features['volume_total_voxels']
    
    logger.info(f"Extracted {len(features)} intensity-based features")
    return features

def extract_regional_features(volume):
    """
    Extract region-based features from an MRI volume.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        dict: Dictionary of region-based features.
    """
    if volume is None:
        logger.error("Cannot extract features from None volume")
        return {}
    
    features = {}
    
    # Split volume into quadrants (8 regions)
    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2
    
    # Define regions
    regions = {
        'top_left_front': volume[:x_mid, :y_mid, :z_mid],
        'top_right_front': volume[x_mid:, :y_mid, :z_mid],
        'bottom_left_front': volume[:x_mid, y_mid:, :z_mid],
        'bottom_right_front': volume[x_mid:, y_mid:, :z_mid],
        'top_left_back': volume[:x_mid, :y_mid, z_mid:],
        'top_right_back': volume[x_mid:, :y_mid, z_mid:],
        'bottom_left_back': volume[:x_mid, y_mid:, z_mid:],
        'bottom_right_back': volume[x_mid:, y_mid:, z_mid:]
    }
    
    # Calculate features for each region
    for region_name, region_volume in regions.items():
        features[f'{region_name}_mean'] = np.mean(region_volume)
        features[f'{region_name}_std'] = np.std(region_volume)
        features[f'{region_name}_volume'] = np.count_nonzero(region_volume)
    
    # Calculate ratios between regions
    total_volume = sum(features[f'{region}_volume'] for region in regions.keys())
    for region_name in regions.keys():
        features[f'{region_name}_volume_ratio'] = features[f'{region_name}_volume'] / total_volume
    
    # Calculate front-to-back ratio (potentially useful for Alzheimer's)
    front_volume = (features['top_left_front_volume'] + features['top_right_front_volume'] + 
                   features['bottom_left_front_volume'] + features['bottom_right_front_volume'])
    back_volume = (features['top_left_back_volume'] + features['top_right_back_volume'] + 
                  features['bottom_left_back_volume'] + features['bottom_right_back_volume'])
    features['front_to_back_ratio'] = front_volume / (back_volume + 1e-6)  # Avoid division by zero
    
    logger.info(f"Extracted {len(features)} region-based features")
    return features

def extract_hippocampus_approximation(volume):
    """
    Approximate hippocampus region and extract features (simplified approach).
    This is a very rough approximation for scratchpad-level code.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        dict: Dictionary of hippocampus-related features.
    """
    if volume is None or len(volume.shape) != 3:
        logger.error("Cannot extract hippocampus features from None or non-3D volume")
        return {}
    
    features = {}
    
    # Very crude approximation of hippocampal region (middle slices, central portion)
    x_start, x_end = int(volume.shape[0] * 0.4), int(volume.shape[0] * 0.6)
    y_start, y_end = int(volume.shape[1] * 0.4), int(volume.shape[1] * 0.6)
    z_start, z_end = int(volume.shape[2] * 0.4), int(volume.shape[2] * 0.6)
    
    # Extract hippocampus-like region
    hippocampus = volume[x_start:x_end, y_start:y_end, z_start:z_end]
    
    # Calculate features
    features['hippocampus_mean'] = np.mean(hippocampus)
    features['hippocampus_std'] = np.std(hippocampus)
    features['hippocampus_volume'] = np.count_nonzero(hippocampus)
    features['hippocampus_to_brain_ratio'] = features['hippocampus_volume'] / np.count_nonzero(volume)
    
    logger.info("Extracted approximate hippocampus features (crude approximation)")
    return features

def extract_ventricle_approximation(volume):
    """
    Approximate ventricle region and extract features (simplified approach).
    This is a very rough approximation for scratchpad-level code.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        dict: Dictionary of ventricle-related features.
    """
    if volume is None:
        logger.error("Cannot extract ventricle features from None volume")
        return {}
    
    features = {}
    
    # Ventricles typically have lower intensity
    # Use a threshold to estimate ventricle regions (very approximate)
    # For better results, proper segmentation would be needed
    threshold = np.percentile(volume[volume > 0], 15)  # Lower 15% of non-zero values
    ventricle_mask = (volume > 0) & (volume < threshold)
    ventricle_region = volume * ventricle_mask
    
    # Extract features
    features['ventricle_volume'] = np.count_nonzero(ventricle_mask)
    features['ventricle_to_brain_ratio'] = features['ventricle_volume'] / np.count_nonzero(volume > 0)
    features['ventricle_mean_intensity'] = np.mean(ventricle_region[ventricle_mask])
    
    logger.info("Extracted approximate ventricle features (threshold-based)")
    return features

def extract_statistical_features(volume):
    """
    Extract all statistical features from an MRI volume.
    
    Args:
        volume (numpy.ndarray): 3D brain volume data.
        
    Returns:
        dict: Dictionary of all statistical features.
    """
    features = {}
    
    # Combine all feature sets
    features.update(extract_intensity_features(volume))
    features.update(extract_regional_features(volume))
    features.update(extract_hippocampus_approximation(volume))
    features.update(extract_ventricle_approximation(volume))
    
    logger.info(f"Extracted a total of {len(features)} statistical features")
    return features

def extract_statistical_features_batch(volumes):
    """
    Extract statistical features from a batch of MRI volumes.
    
    Args:
        volumes (list): List of 3D MRI volume data.
        
    Returns:
        list: List of feature dictionaries for each volume.
    """
    features_list = []
    
    for i, volume in enumerate(volumes):
        logger.info(f"Extracting statistical features for volume {i+1}/{len(volumes)}")
        features = extract_statistical_features(volume)
        features_list.append(features)
    
    return features_list

def dict_list_to_array(dict_list):
    """
    Convert a list of feature dictionaries to a feature array.
    
    Args:
        dict_list (list): List of feature dictionaries.
        
    Returns:
        tuple: (X, feature_names) - Feature array and feature names.
    """
    if not dict_list:
        return np.array([]), []
    
    # Get feature names from the first dictionary
    feature_names = list(dict_list[0].keys())
    
    # Extract features into a 2D array
    X = np.zeros((len(dict_list), len(feature_names)))
    
    for i, feature_dict in enumerate(dict_list):
        for j, feature_name in enumerate(feature_names):
            X[i, j] = feature_dict.get(feature_name, 0)
    
    return X, feature_names 