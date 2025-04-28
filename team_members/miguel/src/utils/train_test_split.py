"""
Utilities for creating train/test splits of MRI data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def split_data_by_subject(metadata_df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train/validation/test sets by subject, ensuring that data from the same subject
    doesn't appear in multiple splits.
    
    Args:
        metadata_df (pandas.DataFrame): DataFrame containing subject IDs and other metadata.
        test_size (float, optional): Proportion of the data to include in the test split.
        val_size (float, optional): Proportion of the data to include in the validation split.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        tuple: (train_df, val_df, test_df) - The split DataFrames.
    """
    # Get unique subject IDs
    subject_ids = metadata_df['Subject ID'].unique()
    
    # First, split into train and test
    train_subjects, test_subjects = train_test_split(
        subject_ids, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Then split train into train and validation
    if val_size > 0:
        effective_val_size = val_size / (1 - test_size)
        train_subjects, val_subjects = train_test_split(
            train_subjects,
            test_size=effective_val_size,
            random_state=random_state
        )
        
        # Create DataFrames for each split
        train_df = metadata_df[metadata_df['Subject ID'].isin(train_subjects)].copy()
        val_df = metadata_df[metadata_df['Subject ID'].isin(val_subjects)].copy()
        test_df = metadata_df[metadata_df['Subject ID'].isin(test_subjects)].copy()
        
        logger.info(f"Split data into train ({len(train_df)} entries), "
                   f"validation ({len(val_df)} entries), and test ({len(test_df)} entries) sets")
        
        return train_df, val_df, test_df
    else:
        # Create DataFrames for train and test only
        train_df = metadata_df[metadata_df['Subject ID'].isin(train_subjects)].copy()
        test_df = metadata_df[metadata_df['Subject ID'].isin(test_subjects)].copy()
        
        logger.info(f"Split data into train ({len(train_df)} entries) and test ({len(test_df)} entries) sets")
        
        return train_df, None, test_df

def split_data_by_range(metadata_df, train_start_idx=50, train_end_idx=199, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data using a specific range of indices for training set, and the remaining data for test/validation.
    
    Args:
        metadata_df (pandas.DataFrame): DataFrame containing subject IDs and other metadata.
        train_start_idx (int): Start index for the training set.
        train_end_idx (int): End index for the training set.
        test_size (float, optional): Proportion of the remaining data to include in the test split.
        val_size (float, optional): Proportion of the remaining data to include in the validation split.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        tuple: (train_df, val_df, test_df) - The split DataFrames.
    """
    # Verify that indices are within range
    if train_start_idx < 0 or train_end_idx >= len(metadata_df):
        logger.warning(f"Invalid index range: {train_start_idx}-{train_end_idx} for dataframe of length {len(metadata_df)}")
        logger.warning("Defaulting to standard split")
        return split_data_by_subject(metadata_df, test_size, val_size, random_state)
    
    # Get range for training set
    train_indices = list(range(train_start_idx, min(train_end_idx + 1, len(metadata_df))))
    
    # Get remainder indices for test/validation
    remaining_indices = [i for i in range(len(metadata_df)) if i < train_start_idx or i > train_end_idx]
    
    # Create training dataframe
    train_df = metadata_df.iloc[train_indices].copy()
    
    # If no remaining data, return only training set
    if not remaining_indices:
        logger.warning("No data left for test/validation. Returning only training set.")
        return train_df, None, None
    
    # Split remaining data into test and validation
    remaining_df = metadata_df.iloc[remaining_indices].copy()
    
    # Split remaining data into test and validation
    if val_size > 0:
        # Get proportion for test from the remaining data
        effective_test_size = test_size / (test_size + val_size)
        
        # Split remaining data
        test_df, val_df = train_test_split(
            remaining_df,
            test_size=(1 - effective_test_size),  # validation gets the rest
            random_state=random_state
        )
        
        logger.info(f"Split data: train ({len(train_df)} entries, indices {train_start_idx}-{train_end_idx}), "
                   f"validation ({len(val_df)} entries), and test ({len(test_df)} entries)")
        
        return train_df, val_df, test_df
    else:
        # All remaining data goes to test
        test_df = remaining_df
        
        logger.info(f"Split data: train ({len(train_df)} entries, indices {train_start_idx}-{train_end_idx}) "
                   f"and test ({len(test_df)} entries)")
        
        return train_df, None, test_df

def create_stratified_split(metadata_df, label_column='CDR', test_size=0.2, val_size=0.1, random_state=42):
    """
    Create a stratified train/validation/test split based on labels.
    
    Args:
        metadata_df (pandas.DataFrame): DataFrame containing subject IDs and labels.
        label_column (str, optional): Column name for the label.
        test_size (float, optional): Proportion of the data to include in the test split.
        val_size (float, optional): Proportion of the data to include in the validation split.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        tuple: (train_df, val_df, test_df) - The split DataFrames.
    """
    # First, split into train and test
    train_df, test_df = train_test_split(
        metadata_df,
        test_size=test_size,
        stratify=metadata_df[label_column],
        random_state=random_state
    )
    
    # Then split train into train and validation
    if val_size > 0:
        effective_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_df,
            test_size=effective_val_size,
            stratify=train_df[label_column],
            random_state=random_state
        )
        
        logger.info(f"Created stratified split: train ({len(train_df)} entries), "
                   f"validation ({len(val_df)} entries), and test ({len(test_df)} entries)")
        
        return train_df, val_df, test_df
    else:
        logger.info(f"Created stratified split: train ({len(train_df)} entries) and test ({len(test_df)} entries)")
        return train_df, None, test_df

def save_split_indices(train_indices, val_indices, test_indices, output_dir):
    """
    Save train/validation/test split indices to files.
    
    Args:
        train_indices (list): Indices for the training set.
        val_indices (list or None): Indices for the validation set.
        test_indices (list): Indices for the test set.
        output_dir (str): Directory to save the index files.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train indices
        np.save(os.path.join(output_dir, 'train_indices.npy'), np.array(train_indices))
        
        # Save validation indices if provided
        if val_indices is not None:
            np.save(os.path.join(output_dir, 'val_indices.npy'), np.array(val_indices))
        
        # Save test indices
        np.save(os.path.join(output_dir, 'test_indices.npy'), np.array(test_indices))
        
        logger.info(f"Saved split indices to {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving split indices: {str(e)}")
        return False
