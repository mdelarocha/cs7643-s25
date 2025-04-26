"""
Dataloader utilities for MRI data.
"""

import os
import pandas as pd
import numpy as np
import logging
from src.utils.preprocessing import preprocess_mri_file, extract_2d_slices, load_mri_volume
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

def load_oasis_metadata(file_path: str, combine_cdr: bool = False) -> Optional[pd.DataFrame]:
    """
    Load OASIS dataset metadata from CSV.
    Optionally combines CDR scores 1.0 and 2.0 into a single category >= 1.0.

    Args:
        file_path (str): Path to the metadata CSV file.
        combine_cdr (bool): If True, map CDR 1.0 and 2.0 to 1.0 in a new 'CDR_Combined' column.
                          Otherwise, use the original 'CDR' column.

    Returns:
        pd.DataFrame: Loaded metadata or None if error.
    """
    if not os.path.exists(file_path):
        logger.error(f"Metadata file not found: {file_path}")
        return None
    
    try:
        metadata_df = pd.read_csv(file_path)
        logger.info(f"Loaded metadata from {file_path}. Shape: {metadata_df.shape}")
        
        # Data Cleaning / Preprocessing Steps (as identified before)
        # 1. Rename columns for consistency
        metadata_df.rename(columns={
            'EDUC': 'Education', 
            'SES': 'SocioeconomicStatus', 
            'MMSE': 'MiniMentalStateExam', 
            'CDR': 'ClinicalDementiaRating', 
            'eTIV': 'EstimatedTotalIntracranialVolume', 
            'nWBV': 'NormalizedWholeBrainVolume', 
            'ASF': 'AtlasScalingFactor'
        }, inplace=True)

        # 2. Handle missing values (Example: fill SES with median, drop rows with missing MMSE/CDR)
        if 'SocioeconomicStatus' in metadata_df.columns:
            median_ses = metadata_df['SocioeconomicStatus'].median()
            metadata_df['SocioeconomicStatus'].fillna(median_ses, inplace=True)
            logger.info(f"Filled missing SocioeconomicStatus with median value: {median_ses}")
            
        # Drop rows where key clinical scores are missing
        initial_rows = len(metadata_df)
        metadata_df.dropna(subset=['MiniMentalStateExam', 'ClinicalDementiaRating'], inplace=True)
        rows_dropped = initial_rows - len(metadata_df)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows due to missing MMSE or CDR values.")

        # Ensure ClinicalDementiaRating is numeric
        if 'ClinicalDementiaRating' in metadata_df.columns:
             metadata_df['ClinicalDementiaRating'] = pd.to_numeric(metadata_df['ClinicalDementiaRating'], errors='coerce')
             metadata_df.dropna(subset=['ClinicalDementiaRating'], inplace=True) # Drop if coercion failed
        else:
             logger.error("'ClinicalDementiaRating' column not found after renaming.")
             return None # Cannot proceed without CDR

        # 3. Feature Engineering / Transformation (Example: Combine CDR scores)
        if combine_cdr:
             logger.info("Combining CDR scores: 0.0 -> 0, >=0.5 -> 1 in 'CDR_Combined' column.")
             # Map 0.0 to 0, and anything 0.5 or greater to 1
             metadata_df['CDR_Combined'] = metadata_df['ClinicalDementiaRating'].apply(lambda x: 0 if x == 0.0 else 1)
             # Verify unique values
             unique_combined = metadata_df['CDR_Combined'].unique()
             logger.info(f"Unique values in 'CDR_Combined' after mapping: {unique_combined}")
             if not np.all(np.isin(unique_combined, [0, 1])):
                  logger.error(f"Unexpected values found in 'CDR_Combined' after mapping: {unique_combined}. Expected only 0 and 1.")
                  # Decide how to handle: drop rows, raise error, etc.
                  # For now, let's proceed but log the error.
        else:
             # If not combining, ensure the original CDR column is suitable as a label
             # Map original CDR scores to three classes: 0 (non-demented), 1 (very mild), 2 (mild/moderate)
             logger.info("Mapping original 'ClinicalDementiaRating' to three classes: 0 -> 0, 0.5 -> 1, >=1 -> 2.")

             def map_cdr_three_class(cdr_score):
                 if cdr_score == 0.0:
                     return 0
                 elif cdr_score == 0.5:
                     return 1
                 elif cdr_score >= 1.0:
                     return 2
                 else:
                     return np.nan # Handle unexpected values

             metadata_df['CDR'] = metadata_df['ClinicalDementiaRating'].apply(map_cdr_three_class)
             
             # Drop rows where mapping failed (e.g., negative CDR?)
             initial_rows_map = len(metadata_df)
             metadata_df.dropna(subset=['CDR'], inplace=True)
             rows_dropped_map = initial_rows_map - len(metadata_df)
             if rows_dropped_map > 0:
                 logger.warning(f"Dropped {rows_dropped_map} rows due to invalid CDR values during three-class mapping.")

             # Ensure the final column is integer type
             metadata_df['CDR'] = metadata_df['CDR'].astype(int)

             unique_three_class = metadata_df['CDR'].unique()
             logger.info(f"Unique values in 'CDR' after three-class mapping: {unique_three_class}")
             expected_labels = [0, 1, 2]
             if not np.all(np.isin(unique_three_class, expected_labels)):
                 logger.warning(f"Unexpected values found in 'CDR' after mapping: {unique_three_class}. Expected values within {expected_labels}.")

        # Log final shape and info
        logger.info(f"Metadata processing complete. Final shape: {metadata_df.shape}")
        logger.debug(f"Metadata columns: {metadata_df.columns.tolist()}")
        logger.debug(f"Metadata info:\n{metadata_df.info()}")
        
        return metadata_df
        
    except FileNotFoundError:
        logger.error(f"Metadata file not found at path: {file_path}")
        return None
    except Exception as e:
        logger.exception(f"Error loading or processing metadata from {file_path}: {e}")
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
                volume = load_mri_volume(file_path)
                
            if volume is not None:
                volumes.append(volume)
                
        except Exception as e:
            logger.error(f"Error loading MRI file {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(volumes)} MRI volumes out of {len(file_paths)} files")
    return volumes

def combine_cdr_scores(cdr_scores):
    """
    Combine CDR scores 1 and 2 into a single category.
    
    Args:
        cdr_scores (numpy.ndarray): Array of CDR scores.
        
    Returns:
        numpy.ndarray: Processed CDR scores with values 1 and 2 combined into 1.
    """
    # Convert to numpy array if not already
    y_array = np.array(cdr_scores)
    
    # Create a new array to hold the combined scores
    y_combined = y_array.copy()
    
    # Combine CDR scores 1 and 2 into a single category (>=1)
    mask_1_or_2 = (y_array == 1.0) | (y_array == 2.0)
    y_combined[mask_1_or_2] = 1.0
    
    # Log the changes
    original_counts = pd.Series(y_array).value_counts().sort_index()
    combined_counts = pd.Series(y_combined).value_counts().sort_index()
    
    logger.info(f"Original CDR score distribution: {original_counts.to_dict()}")
    logger.info(f"Combined CDR score distribution: {combined_counts.to_dict()}")
    
    return y_combined

def create_dataset_from_metadata(metadata_df: pd.DataFrame, data_dir: str, label_col: str = 'CDR_Combined') -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
    """
    Create a dataset (list of MRI volumes and labels) from metadata.
    Assumes the label column already exists and contains the desired labels (e.g., 0 or 1).

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata, including filenames and labels.
        data_dir (str): Base directory where MRI files are stored.
        label_col (str): Name of the column containing the target labels (e.g., 'CDR_Combined' or 'ClinicalDementiaRating').
        
    Returns:
        tuple: (volumes, labels) or (None, None) if error.
               volumes: List of loaded MRI volumes (numpy arrays).
               labels: Numpy array of corresponding labels.
    """
    if metadata_df is None or metadata_df.empty:
        logger.error("Metadata DataFrame is empty or None. Cannot create dataset.")
        return None, None
        
    if label_col not in metadata_df.columns:
         logger.error(f"Label column '{label_col}' not found in metadata. Available columns: {metadata_df.columns.tolist()}")
         return None, None

    volumes = []
    labels = []
    missing_files = 0
    load_errors = 0

    logger.info(f"Creating dataset from {len(metadata_df)} metadata entries using label column '{label_col}'.")
    for index, row in metadata_df.iterrows():
        # Construct file path (handle potential variations in how filename/ID is stored)
        subject_id = row.get('ID')
        if subject_id:
            subject_base_dir = os.path.join(data_dir, subject_id)
            file_path = None
            if os.path.isdir(subject_base_dir):
                # Search common locations and file types
                potential_locations = [
                    subject_base_dir, # Check root of subject folder
                    os.path.join(subject_base_dir, 'RAW'),
                    os.path.join(subject_base_dir, 'PROCESSED', 'MPRAGE', 'T88_111'), # Keep previous attempt as fallback
                    os.path.join(subject_base_dir, 'ANALYZE')
                ]
                found_nifti = None
                found_analyze = None
                
                for loc in potential_locations:
                    if os.path.isdir(loc):
                        for fname in os.listdir(loc):
                            if fname.endswith(('.nii.gz', '.nii')): # Prioritize NIfTI
                                found_nifti = os.path.join(loc, fname)
                                break # Found best type, stop searching this location
                            elif fname.endswith('.img') and not found_nifti: # Find ANALYZE only if NIfTI not found yet
                                 found_analyze = os.path.join(loc, fname)
                        if found_nifti: # Found NIfTI in this location, stop searching other locations
                             break 
                             
                # Assign path based on priority
                if found_nifti:
                     file_path = found_nifti
                     logger.debug(f"Found NIfTI scan file for {subject_id}: {file_path}")
                elif found_analyze:
                     file_path = found_analyze
                     logger.debug(f"Found ANALYZE scan file for {subject_id}: {file_path}")
                # else: file_path remains None
            else:
                logger.warning(f"Subject base directory not found: {subject_base_dir}")
                
        else:
             logger.warning(f"Missing 'ID' for row index {index}. Skipping.")
             continue

        if file_path and os.path.exists(file_path):
            volume_data = load_mri_volume(file_path) # Use the utility from preprocessing
            if volume_data is not None:
                volumes.append(volume_data)
                labels.append(row[label_col])
            else:
                load_errors += 1
                logger.warning(f"Failed to load volume data for {file_path}")
        else:
            missing_files += 1
            if subject_id: # Log which subject had missing file
                 logger.warning(f"MRI file not found or path invalid for Subject ID: {subject_id} (Expected path structure: {file_path})")
            else:
                 logger.warning(f"MRI file path could not be constructed or file missing for row index {index}")

    logger.info(f"Dataset creation finished. Loaded {len(volumes)} volumes.")
    if missing_files > 0:
        logger.warning(f"Could not find {missing_files} MRI files.")
    if load_errors > 0:
        logger.warning(f"Failed to load data for {load_errors} files.")
        
    if not volumes or not labels:
        logger.error("Failed to load any valid volumes or labels. Check data paths and metadata file linkage.")
        return None, None

    return volumes, np.array(labels)

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
