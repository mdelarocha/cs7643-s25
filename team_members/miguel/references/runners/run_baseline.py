#!/usr/bin/env python
"""
Script to run the baseline pipeline on a sample of the OASIS dataset.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

from src.pipelines.baseline_pipeline import run_baseline_pipeline
from src.utils.dataloader import load_oasis_metadata

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('baseline_run.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
METADATA_PATH = "data/oasis-cross-sectional.csv"
DATA_DIR = "data/raw/OASIS_0001"
OUTPUT_DIR = "outputs"

def prepare_metadata_for_sample():
    """
    Prepare metadata for the sample MRI files we've downloaded.
    We need to adjust the metadata to match the filenames we have.
    """
    # Load the original metadata
    metadata = load_oasis_metadata(METADATA_PATH)
    
    # Rename ID column to match what the pipeline expects
    if 'ID' in metadata.columns and 'Subject ID' not in metadata.columns:
        metadata = metadata.rename(columns={'ID': 'Subject ID'})
    
    # Get the list of files we've downloaded
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.img')]
    
    # Extract subject IDs from filenames
    subject_ids = []
    for filename in data_files:
        # Extract OAS1_XXXX_MRX part from filename
        parts = filename.split('_')
        if len(parts) >= 3:
            if parts[0] == 'OAS1' and parts[2].startswith('MR'):
                subject_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                subject_ids.append(subject_id)
            else:
                # For other filename patterns, try to find the subject ID
                for i in range(len(parts)):
                    if parts[i].startswith('OAS1') or parts[i] == 'OAS1':
                        if i + 2 < len(parts) and parts[i+2].startswith('MR'):
                            subject_id = f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
                            subject_ids.append(subject_id)
                            break
    
    # Make subject IDs unique
    subject_ids = list(set(subject_ids))
    logger.info(f"Found {len(subject_ids)} unique subject IDs in the sample data")
    
    # Create synthetic metadata with enough entries for train/test split
    sample_metadata = create_synthetic_metadata(data_files)
    
    # Log the number of rows
    logger.info(f"Sample metadata contains {len(sample_metadata)} entries")
    
    return sample_metadata

def create_synthetic_metadata(data_files):
    """
    Create synthetic metadata for the sample files.
    Since we only have one subject, we'll create synthetic variations
    to have enough data for training and testing.
    
    Args:
        data_files (list): List of MRI filenames.
        
    Returns:
        pd.DataFrame: Synthetic metadata.
    """
    # If no files, create default entry
    if not data_files:
        logger.warning("No MRI files found. Creating default synthetic data.")
        return pd.DataFrame({
            'Subject ID': [f"OAS1_SYNTH_{i:03d}_MR1" for i in range(1, 21)],
            'M/F': np.random.choice(['M', 'F'], size=20),
            'Age': np.random.randint(60, 90, size=20),
            'CDR': np.random.choice([0, 0.5, 1], size=20, p=[0.5, 0.3, 0.2]),
            'Visit': [1] * 20,
            'MRI_file': ['dummy.img'] * 20
        })
    
    # Get sample file for duplication
    sample_file = data_files[0]
    
    # Create synthetic metadata entries
    subjects = []
    genders = []
    ages = []
    cdrs = []
    visits = []
    filenames = []
    
    # Create 20 synthetic entries
    for i in range(1, 21):
        subjects.append(f"OAS1_SYNTH_{i:03d}_MR1")
        genders.append(np.random.choice(['M', 'F']))
        ages.append(np.random.randint(60, 90))
        cdrs.append(np.random.choice([0, 0.5, 1], p=[0.5, 0.3, 0.2]))
        visits.append(1)
        filenames.append(sample_file)
    
    # Create DataFrame
    return pd.DataFrame({
        'Subject ID': subjects,
        'M/F': genders,
        'Age': ages,
        'CDR': cdrs,
        'Visit': visits,
        'MRI_file': filenames
    })

def find_matching_file(subject_id, data_files):
    """
    Find a matching MRI file for the given subject ID.
    
    Args:
        subject_id (str): Subject ID in the format 'OAS1_XXXX_MRX'.
        data_files (list): List of available MRI filenames.
        
    Returns:
        str: The matching filename, or None if no match found.
    """
    # Check for exact match
    matching_files = [f for f in data_files if subject_id in f]
    
    # If multiple files, prefer T88 processed files
    if len(matching_files) > 1:
        t88_files = [f for f in matching_files if 't88' in f.lower()]
        if t88_files:
            return t88_files[0]
    
    # Return any matching file or None
    return matching_files[0] if matching_files else None

def run_pipeline_on_sample():
    """
    Run the baseline pipeline on the sample data.
    """
    # Prepare metadata for our sample
    sample_metadata = prepare_metadata_for_sample()
    
    # Save the sample metadata for reference
    sample_metadata_path = os.path.join(OUTPUT_DIR, "sample_metadata.csv")
    sample_metadata.to_csv(sample_metadata_path, index=False)
    logger.info(f"Saved sample metadata to {sample_metadata_path}")
    
    # Run the pipeline
    logger.info("Running baseline pipeline on sample data...")
    
    # Define model types to train (use fewer to speed up processing)
    model_types = ['logistic_regression', 'random_forest']
    
    # Define feature types to extract (use fewer features to speed up processing)
    feature_types = ['statistical']
    
    # Run the pipeline with smaller test size due to limited data
    models, results = run_baseline_pipeline(
        metadata_path=sample_metadata_path,
        data_dir=DATA_DIR,
        model_types=model_types,
        feature_types=feature_types,
        output_dir=OUTPUT_DIR,
        test_size=0.2,
        val_size=0.1,
        n_features=20
    )
    
    if models is not None and results is not None:
        logger.info("Pipeline completed successfully")
        return True
    else:
        logger.error("Pipeline failed")
        return False

if __name__ == "__main__":
    logger.info("Starting baseline pipeline run on sample data")
    
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run the pipeline
    success = run_pipeline_on_sample()
    
    if success:
        logger.info("Baseline pipeline completed successfully")
    else:
        logger.error("Baseline pipeline failed") 