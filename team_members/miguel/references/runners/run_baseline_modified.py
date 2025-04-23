#!/usr/bin/env python
"""
Modified script to run only the feature extraction step of the baseline pipeline.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

from src.utils.dataloader import load_oasis_metadata, create_dataset_from_metadata
from src.utils.train_test_split import split_data_by_subject
from src.features.statistical import extract_statistical_features, extract_statistical_features_batch, dict_list_to_array
from src.features.textural import extract_textural_features, extract_textural_features_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('baseline_extract.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
METADATA_PATH = "data/oasis-cross-sectional.csv"
DATA_DIR = "data/raw/OASIS_0001"
OUTPUT_DIR = "outputs/features"

def create_synthetic_metadata(data_files):
    """
    Create synthetic metadata for the sample files.
    """
    # Get sample file for duplication
    sample_file = data_files[0]
    
    # Create synthetic metadata entries - using fewer to speed up processing
    subjects = []
    cdrs = []
    filenames = []
    
    # Create 10 synthetic entries
    for i in range(1, 11):
        subjects.append(f"OAS1_SYNTH_{i:03d}_MR1")
        # Assign samples to different classes: normal (0), mild (0.5), and moderate dementia (1)
        if i <= 5:
            cdrs.append(0)  # Normal
        elif i <= 8:
            cdrs.append(0.5)  # Very mild dementia
        else:
            cdrs.append(1)  # Mild dementia
        filenames.append(sample_file)
    
    # Create DataFrame
    return pd.DataFrame({
        'Subject ID': subjects,
        'CDR': cdrs,
        'MRI_file': filenames
    })

def extract_features_manually():
    """
    Extract features manually from MRI volumes.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of MRI files
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.img')]
    
    if not data_files:
        logger.error("No MRI files found in data directory")
        return False
    
    # Create synthetic metadata
    metadata = create_synthetic_metadata(data_files)
    
    # Save the metadata
    metadata_path = os.path.join(OUTPUT_DIR, "synthetic_metadata.csv")
    metadata.to_csv(metadata_path, index=False)
    logger.info(f"Saved synthetic metadata to {metadata_path}")
    
    # Split data into train/val/test
    train_df, val_df, test_df = split_data_by_subject(metadata, test_size=0.2, val_size=0.1)
    
    # Create datasets
    logger.info("Creating train dataset...")
    X_train_raw, y_train = create_dataset_from_metadata(train_df, DATA_DIR)
    
    logger.info("Creating validation dataset...")
    X_val_raw, y_val = create_dataset_from_metadata(val_df, DATA_DIR) if val_df is not None else (None, None)
    
    logger.info("Creating test dataset...")
    X_test_raw, y_test = create_dataset_from_metadata(test_df, DATA_DIR)
    
    # Extract statistical features from training set
    logger.info("Extracting statistical features from training set...")
    try:
        train_stat_features = extract_statistical_features_batch(X_train_raw)
        X_train_stat, train_stat_names = dict_list_to_array(train_stat_features)
        logger.info(f"Extracted {X_train_stat.shape[1]} statistical features from training set")
        
        # Save features
        with open(os.path.join(OUTPUT_DIR, "train_stat_features.pkl"), 'wb') as f:
            pickle.dump({
                'features': X_train_stat,
                'labels': y_train,
                'feature_names': train_stat_names
            }, f)
    except Exception as e:
        logger.error(f"Error extracting statistical features: {str(e)}")
        return False
    
    # Extract statistical features from test set
    logger.info("Extracting statistical features from test set...")
    try:
        test_stat_features = extract_statistical_features_batch(X_test_raw)
        X_test_stat, _ = dict_list_to_array(test_stat_features)
        logger.info(f"Extracted {X_test_stat.shape[1]} statistical features from test set")
        
        # Save features
        with open(os.path.join(OUTPUT_DIR, "test_stat_features.pkl"), 'wb') as f:
            pickle.dump({
                'features': X_test_stat,
                'labels': y_test,
                'feature_names': train_stat_names  # Use same names as training
            }, f)
    except Exception as e:
        logger.error(f"Error extracting statistical features: {str(e)}")
        return False
    
    # Try to extract just one set of textural features as a test
    try:
        logger.info("Trying to extract textural features from first volume...")
        first_volume = X_train_raw[0]
        textural_features = extract_textural_features(first_volume)
        logger.info(f"Successfully extracted {len(textural_features)} textural features")
        
        # Save sample textural features
        with open(os.path.join(OUTPUT_DIR, "sample_textural_features.pkl"), 'wb') as f:
            pickle.dump(textural_features, f)
    except Exception as e:
        logger.error(f"Error extracting textural features: {str(e)}")
    
    return True

if __name__ == "__main__":
    logger.info("Starting feature extraction test")
    
    # Extract features
    success = extract_features_manually()
    
    if success:
        logger.info("Feature extraction completed successfully")
    else:
        logger.error("Feature extraction failed") 