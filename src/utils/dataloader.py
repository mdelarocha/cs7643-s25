"""
Dataloader utilities for MRI data.
"""

import os
import pandas as pd
import numpy as np
import logging
from src.utils.preprocessing import preprocess_mri_file, extract_2d_slices

logger = logging.getLogger(__name__)

def load_oasis_metadata(metadata_path):
    """
    Load OASIS dataset metadata from CSV file.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        
    Returns:
        pandas.DataFrame: The loaded metadata.
    """
    try:
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata from {metadata_path} with {len(metadata)} entries")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_path}: {str(e)}")
        return None

def load_mri_files(file_paths, preprocess=True, normalize=True, extract_core=True):
    """
    Load multiple MRI files and preprocess them.
    
    Args:
        file_paths (list): List of paths to MRI files.
        preprocess (bool, optional): Whether to preprocess the volumes.
        normalize (bool, optional): Whether to normalize the volumes.
        extract_core (bool, optional): Whether to extract core slices.
        
    Returns:
        list: List of loaded and preprocessed MRI volumes.
    """
    volumes = []
    
    for file_path in file_paths:
        try:
            if preprocess:
                volume = preprocess_mri_file(file_path, normalize=normalize, extract_core=extract_core)
            else:
                from src.utils.preprocessing import load_mri_volume
                volume = load_mri_volume(file_path)
                
            if volume is not None:
                volumes.append(volume)
                
        except Exception as e:
            logger.error(f"Error loading MRI file {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(volumes)} MRI volumes out of {len(file_paths)} files")
    return volumes

def create_dataset_from_metadata(metadata_df, data_dir, preprocess=True, normalize=True, extract_core=True):
    """
    Create a dataset from metadata and MRI files.
    
    Args:
        metadata_df (pandas.DataFrame): DataFrame containing subject IDs and file paths.
        data_dir (str): Base directory for MRI files.
        preprocess (bool, optional): Whether to preprocess the volumes.
        normalize (bool, optional): Whether to normalize the volumes.
        extract_core (bool, optional): Whether to extract core slices.
        
    Returns:
        tuple: (X, y) - Features and labels.
    """
    X = []
    y = []
    
    for _, row in metadata_df.iterrows():
        try:
            # Construct file path - adjust based on your metadata structure
            subject_id = row.get('Subject ID')
            visit = row.get('Visit')
            filename = row.get('MRI_file', f"{subject_id}_{visit}.nii.gz")
            
            file_path = os.path.join(data_dir, filename)
            
            # Load and preprocess the MRI volume
            volume = preprocess_mri_file(file_path, normalize=normalize, extract_core=extract_core)
            
            if volume is not None:
                X.append(volume)
                
                # Get the label - adjust based on your task (classification, regression, etc.)
                label = row.get('CDR', 0)  # Clinical Dementia Rating
                y.append(label)
        
        except Exception as e:
            logger.error(f"Error processing entry for subject {subject_id}: {str(e)}")
    
    logger.info(f"Created dataset with {len(X)} samples and {len(y)} labels")
    return np.array(X), np.array(y)

def create_2d_slice_dataset(volumes, labels=None, axis=2):
    """
    Convert 3D volumes to a dataset of 2D slices for 2D CNN models.
    
    Args:
        volumes (list): List of 3D MRI volumes.
        labels (list, optional): List of labels for each volume.
        axis (int, optional): Axis along which to extract slices.
        
    Returns:
        tuple: (X_slices, y_slices) - Features and labels for 2D slices.
    """
    X_slices = []
    y_slices = []
    
    for i, volume in enumerate(volumes):
        slices = extract_2d_slices(volume, axis=axis)
        
        X_slices.extend(slices)
        
        if labels is not None:
            # Repeat the same label for all slices from the same volume
            label = labels[i]
            y_slices.extend([label] * len(slices))
    
    # Convert to appropriate shape for ML models
    X_slices = np.array(X_slices)
    
    # Add channel dimension for CNN input if needed
    if len(X_slices.shape) == 3:
        X_slices = X_slices[..., np.newaxis]
    
    if labels is not None:
        y_slices = np.array(y_slices)
        logger.info(f"Created 2D slice dataset with {len(X_slices)} slices and {len(y_slices)} labels")
        return X_slices, y_slices
    else:
        logger.info(f"Created 2D slice dataset with {len(X_slices)} slices (no labels)")
        return X_slices
