#!/usr/bin/env python
"""
Script to download additional MRI data from GCS and run the analysis pipeline.
"""

import os
import sys
import logging
import argparse
import re
from tqdm import tqdm

from src.utils.gcs_utils import authenticate_gcs, list_bucket_contents, download_blob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download_and_process.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
BUCKET_NAME = "oasis-1-dataset-13635"
DATA_DIR = "data/raw/OASIS_EXPANDED"
MAX_SUBJECTS = 9999  # No limit - download all available subjects

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download MRI data and run analysis pipeline")
    parser.add_argument("--masked_only", action="store_true", help="Download only masked MRI files")
    parser.add_argument("--max_subjects", type=int, default=MAX_SUBJECTS, 
                        help=f"Maximum number of subjects to download (default: {MAX_SUBJECTS})")
    parser.add_argument("--run_pipeline", action="store_true", help="Run the pipeline after downloading")
    return parser.parse_args()

def identify_subject_pattern(blob_names):
    """
    Analyze the blob names to identify patterns for subject IDs.
    
    Args:
        blob_names (list): List of blob names
        
    Returns:
        list: Unique subject IDs found in the blob names
    """
    # Pattern for OASIS subject IDs (e.g., OAS1_0001_MR1)
    pattern = r'(OAS1_\d{4}_MR\d+)'
    
    subject_ids = set()
    for blob_name in blob_names:
        match = re.search(pattern, blob_name)
        if match:
            subject_ids.add(match.group(1))
    
    return sorted(list(subject_ids))

def get_masked_files(blob_names, subject_id):
    """
    Filter blob names to get masked files for a specific subject.
    
    Args:
        blob_names (list): List of blob names
        subject_id (str): Subject ID to filter by
        
    Returns:
        list: Filtered list of blob names containing masked files
    """
    # Patterns for masked files (different possible formats)
    patterns = [
        r"{0}.*masked.*\.img$".format(subject_id),
        r"{0}.*masked.*\.hdr$".format(subject_id),
        r"{0}.*fseg.*\.img$".format(subject_id),
        r"{0}.*fseg.*\.hdr$".format(subject_id)
    ]
    
    masked_files = []
    for blob_name in blob_names:
        if any(re.search(pattern, blob_name) for pattern in patterns):
            masked_files.append(blob_name)
    
    return masked_files

def download_subject_data(client, subject_id, blob_names, masked_only=True):
    """
    Download MRI data for a specific subject.
    
    Args:
        client: GCS client
        subject_id (str): Subject ID
        blob_names (list): List of all blob names
        masked_only (bool): Whether to download only masked files
        
    Returns:
        int: Number of files downloaded
    """
    # Create subject directory
    subject_dir = os.path.join(DATA_DIR, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    
    # Filter files for this subject
    if masked_only:
        subject_files = get_masked_files(blob_names, subject_id)
    else:
        subject_files = [b for b in blob_names if subject_id in b]
    
    # Download files
    files_downloaded = 0
    for blob_name in subject_files:
        # Extract filename from blob path
        filename = os.path.basename(blob_name)
        local_path = os.path.join(subject_dir, filename)
        
        # Download if file doesn't exist locally
        if not os.path.exists(local_path):
            success = download_blob(BUCKET_NAME, blob_name, local_path, client)
            if success:
                files_downloaded += 1
    
    return files_downloaded

def download_oasis_data(masked_only=True, max_subjects=MAX_SUBJECTS):
    """
    Download OASIS MRI data from GCS bucket.
    
    Args:
        masked_only (bool): Whether to download only masked files
        max_subjects (int): Maximum number of subjects to download
        
    Returns:
        int: Total number of files downloaded
    """
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Authenticate with GCS
    client = authenticate_gcs()
    
    # List all objects in the bucket
    logger.info(f"Listing contents of bucket '{BUCKET_NAME}'...")
    blob_names = list_bucket_contents(BUCKET_NAME, client=client)
    logger.info(f"Found {len(blob_names)} objects in the bucket")
    
    # Identify unique subject IDs
    subject_ids = identify_subject_pattern(blob_names)
    logger.info(f"Found {len(subject_ids)} unique subject IDs")
    
    # Limit the number of subjects
    if max_subjects and max_subjects < len(subject_ids):
        subject_ids = subject_ids[:max_subjects]
        logger.info(f"Limited to {max_subjects} subjects")
    
    # Download data for each subject
    total_files = 0
    for subject_id in tqdm(subject_ids, desc="Downloading subjects"):
        files = download_subject_data(client, subject_id, blob_names, masked_only)
        total_files += files
        logger.info(f"Downloaded {files} files for subject {subject_id}")
    
    logger.info(f"Downloaded a total of {total_files} files for {len(subject_ids)} subjects")
    return total_files

def run_feature_extraction_pipeline():
    """Run the feature extraction pipeline on the downloaded data."""
    try:
        from run_baseline_modified import extract_features_manually
        logger.info("Running feature extraction pipeline...")
        success = extract_features_manually()
        if success:
            logger.info("Feature extraction completed successfully")
        else:
            logger.error("Feature extraction failed")
        return success
    except Exception as e:
        logger.error(f"Error running feature extraction: {str(e)}")
        return False

def run_model_training_pipeline():
    """Run the model training pipeline on the extracted features."""
    try:
        import run_baseline_train_balanced
        logger.info("Running model training pipeline...")
        # Execute the script's main function
        success = run_baseline_train_balanced.train_baseline_models()
        if success:
            logger.info("Model training completed successfully")
        else:
            logger.error("Model training failed")
        return success
    except Exception as e:
        logger.error(f"Error running model training: {str(e)}")
        return False

def update_pipeline_config():
    """Update pipeline configuration to use the expanded dataset."""
    # Update the data directory in run_baseline_modified.py
    try:
        with open("run_baseline_modified.py", "r") as f:
            content = f.read()
        
        # Update the DATA_DIR constant
        updated_content = re.sub(
            r'DATA_DIR = "data/raw/OASIS_0001"',
            f'DATA_DIR = "{DATA_DIR}"',
            content
        )
        
        with open("run_baseline_modified.py", "w") as f:
            f.write(updated_content)
        
        logger.info("Updated pipeline configuration to use expanded dataset")
        return True
    except Exception as e:
        logger.error(f"Error updating pipeline configuration: {str(e)}")
        return False

if __name__ == "__main__":
    args = parse_args()
    
    logger.info("Starting OASIS data download and processing")
    
    # Download data
    total_files = download_oasis_data(
        masked_only=args.masked_only,
        max_subjects=args.max_subjects
    )
    
    if args.run_pipeline and total_files > 0:
        # Update pipeline configuration
        if update_pipeline_config():
            # Run the feature extraction pipeline
            if run_feature_extraction_pipeline():
                # Run the model training pipeline
                run_model_training_pipeline()
    
    logger.info("Download and processing complete") 