#!/usr/bin/env python
"""
Script to download the complete OASIS dataset from the GCS bucket.
"""

import os
import logging
import time
from tqdm import tqdm
from src.utils.gcs_utils import authenticate_gcs, list_bucket_contents, download_blob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download_dataset.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
BUCKET_NAME = "oasis-1-dataset-13635"
DATA_DIR = "data/raw"
METADATA_FILE = "oasis-cross-sectional.csv"

def download_metadata():
    """Download metadata file if it doesn't exist."""
    metadata_path = os.path.join("data", METADATA_FILE)
    
    if os.path.exists(metadata_path):
        logger.info(f"Metadata file already exists at {metadata_path}")
        return True
    
    logger.info("Downloading metadata file...")
    client = authenticate_gcs()
    return download_blob(BUCKET_NAME, METADATA_FILE, metadata_path, client)

def download_all_mri_data():
    """Download all MRI data from the GCS bucket."""
    client = authenticate_gcs()
    
    # List all blobs in the bucket
    logger.info("Listing all files in the GCS bucket...")
    all_blobs = list_bucket_contents(BUCKET_NAME, client=client)
    
    # Filter for MRI data files (ending with .img or .hdr)
    mri_files = [blob for blob in all_blobs if blob.endswith('.img') or blob.endswith('.hdr')]
    logger.info(f"Found {len(mri_files)} MRI-related files to download")
    
    # Group files by subject to create organized directories
    subjects = {}
    for blob in mri_files:
        # Extract subject ID (e.g., OAS1_0001_MR1)
        parts = os.path.basename(blob).split('_')
        if len(parts) >= 3:
            subject_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(blob)
    
    logger.info(f"Found data for {len(subjects)} unique subjects")
    
    # Download files for each subject
    total_files = len(mri_files)
    downloaded = 0
    skipped = 0
    
    with tqdm(total=total_files, desc="Downloading files") as pbar:
        for subject_id, blobs in subjects.items():
            # Create subject directory
            subject_dir = os.path.join(DATA_DIR, subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            
            # Download all files for this subject
            for blob in blobs:
                destination = os.path.join(DATA_DIR, blob)
                
                # Skip if file already exists
                if os.path.exists(destination):
                    logger.debug(f"File already exists: {destination}")
                    skipped += 1
                    pbar.update(1)
                    continue
                
                # Download file
                success = download_blob(BUCKET_NAME, blob, destination, client)
                if success:
                    downloaded += 1
                pbar.update(1)
                
                # Add a small delay to avoid overwhelming the API
                time.sleep(0.1)
    
    logger.info(f"Download complete: {downloaded} files downloaded, {skipped} files skipped (already existed)")
    return downloaded > 0

def main():
    """Main function to download the complete dataset."""
    logger.info("Starting download of OASIS dataset")
    
    # Ensure data directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download metadata
    metadata_success = download_metadata()
    if not metadata_success:
        logger.error("Failed to download metadata file")
        return False
    
    # Download MRI data
    mri_success = download_all_mri_data()
    if not mri_success:
        logger.warning("No new MRI files were downloaded")
    
    logger.info("Dataset download process completed")
    return True

if __name__ == "__main__":
    main() 