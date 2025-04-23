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
        
        # Standardize column names - handle different variations of subject ID column
        if 'ID' in metadata.columns and 'Subject ID' not in metadata.columns:
            metadata = metadata.rename(columns={'ID': 'Subject ID'})
            logger.info("Renamed 'ID' column to 'Subject ID'")
        
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
            # Get subject ID
            subject_id = row.get('Subject ID')
            if subject_id is None:
                logger.warning(f"Row missing 'Subject ID': {row}")
                continue
                
            # Get CDR score (label)
            cdr_score = row.get('CDR', None)
            if cdr_score is None or pd.isna(cdr_score):
                logger.debug(f"Skipping subject {subject_id} with missing CDR score")
                continue
            
            # Look for MRI files in subject directory
            subject_dir = os.path.join(data_dir, subject_id)
            
            # If subject directory exists
            if os.path.isdir(subject_dir):
                # Find masked MRI files (*.img files)
                mri_files = []
                for root, _, files in os.walk(subject_dir):
                    for file in files:
                        if file.endswith('.img') and ('masked' in file or 'fseg' in file):
                            mri_files.append(os.path.join(root, file))
                
                if not mri_files:
                    logger.warning(f"No masked MRI files found for subject {subject_id}")
                    continue
                
                # Use the first MRI file found
                file_path = mri_files[0]
                
                # Load and preprocess the MRI volume
                volume = preprocess_mri_file(file_path, normalize=normalize, extract_core=extract_core)
                
                if volume is not None:
                    X.append(volume)
                    y.append(cdr_score)
            else:
                logger.warning(f"Subject directory not found: {subject_dir}")
        
        except Exception as e:
            logger.error(f"Error processing entry for subject {subject_id}: {str(e)}")
    
    if X and y:
        logger.info(f"Created dataset with {len(X)} samples and {len(y)} labels")
        
        # Convert labels to proper format for classification
        try:
            # Convert labels to numeric values if they are not already
            y_array = np.array(y)
            
            # Check if labels are numeric or strings
            if isinstance(y_array[0], (str, np.str_)):
                logger.info(f"Converting string labels to numeric values: {np.unique(y_array)}")
                # Use a mapping for string labels (e.g., 'Positive', 'Negative')
                label_map = {label: i for i, label in enumerate(np.unique(y_array))}
                y_numeric = np.array([label_map[label] for label in y_array])
                logger.info(f"Mapped labels: {label_map}")
            else:
                # If numeric, ensure they're floats for regression or integers for classification
                if np.issubdtype(y_array.dtype, np.floating):
                    # For CDR scores, round to common values (0, 0.5, 1, etc.)
                    # For classification, convert to integers
                    # Get unique values to determine if classification or regression
                    unique_vals = np.unique(y_array)
                    logger.info(f"Found unique label values: {unique_vals}")
                    
                    if len(unique_vals) <= 5:  # Likely classification with few classes
                        # Map CDR scores to integer classes (e.g., 0->0, 0.5->1, 1->2, etc.)
                        label_map = {float(val): i for i, val in enumerate(sorted(unique_vals))}
                        y_numeric = np.array([label_map[float(val)] for val in y_array])
                        logger.info(f"Mapped numeric labels to integers: {label_map}")
                    else:
                        # Keep as is for regression
                        y_numeric = y_array
                else:
                    # Already integers, keep as is
                    y_numeric = y_array
            
            logger.info(f"Final label format: {y_numeric.dtype}, shape: {y_numeric.shape}, unique values: {np.unique(y_numeric)}")
            return np.array(X), y_numeric
        except Exception as e:
            logger.error(f"Error converting labels: {str(e)}")
            # Return original format if conversion fails
            return np.array(X), np.array(y)
    else:
        logger.warning("No valid samples found in dataset")
        return np.array([]), np.array([])

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
