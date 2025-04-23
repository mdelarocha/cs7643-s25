"""
Utilities for downloading OASIS dataset files from Google Cloud Storage.
"""

import os
import logging
import re
import time
from tqdm import tqdm

from src.utils.gcs_utils import authenticate_gcs, list_bucket_contents, download_blob

logger = logging.getLogger(__name__)

# Constants
BUCKET_NAME = "oasis-1-dataset-13635"
DATA_DIR = "data/raw"
METADATA_FILE = "oasis-cross-sectional.csv"

def download_metadata(output_dir="data", force=False):
    """
    Download the OASIS metadata file.
    
    Args:
        output_dir (str): Directory to save the metadata file
        force (bool): Whether to force download even if file exists
        
    Returns:
        str: Path to the downloaded metadata file, or None if download failed
    """
    metadata_path = os.path.join(output_dir, METADATA_FILE)
    
    if os.path.exists(metadata_path) and not force:
        logger.info(f"Metadata file already exists at {metadata_path}")
        return metadata_path
    
    logger.info("Downloading metadata file...")
    client = authenticate_gcs()
    success = download_blob(BUCKET_NAME, METADATA_FILE, metadata_path, client)
    
    if success:
        logger.info(f"Metadata downloaded to {metadata_path}")
        return metadata_path
    else:
        logger.error("Failed to download metadata file")
        return None

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

def download_subject_data(client, subject_id, blob_names, data_dir=DATA_DIR, masked_only=True):
    """
    Download MRI data for a specific subject.
    
    Args:
        client: GCS client
        subject_id (str): Subject ID
        blob_names (list): List of all blob names
        data_dir (str): Directory to save the downloaded files
        masked_only (bool): Whether to download only masked files
        
    Returns:
        int: Number of files downloaded
    """
    # Create subject directory
    subject_dir = os.path.join(data_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    
    # Filter files for this subject
    if masked_only:
        subject_files = get_masked_files(blob_names, subject_id)
    else:
        subject_files = [b for b in blob_names if subject_id in b]
    
    # Download files
    files_downloaded = 0
    for blob_name in subject_files:
        # Create directory structure if needed
        destination = os.path.join(data_dir, blob_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download if file doesn't exist locally
        if not os.path.exists(destination):
            success = download_blob(BUCKET_NAME, blob_name, destination, client)
            if success:
                files_downloaded += 1
            
            # Add a small delay to avoid overwhelming the API
            time.sleep(0.1)
    
    return files_downloaded

def download_oasis_data(data_dir=DATA_DIR, masked_only=True, max_subjects=None):
    """
    Download OASIS MRI data from GCS bucket.
    
    Args:
        data_dir (str): Directory to save the downloaded files
        masked_only (bool): Whether to download only masked files
        max_subjects (int): Maximum number of subjects to download
        
    Returns:
        int: Total number of files downloaded
    """
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
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
        files = download_subject_data(client, subject_id, blob_names, data_dir, masked_only)
        total_files += files
        logger.info(f"Downloaded {files} files for subject {subject_id}")
    
    logger.info(f"Downloaded a total of {total_files} files for {len(subject_ids)} subjects")
    return total_files

def download_sample_data(n_subjects=5, data_dir=DATA_DIR, masked_only=True):
    """
    Download a sample of the OASIS dataset (useful for development and testing).
    
    Args:
        n_subjects (int): Number of subjects to download
        data_dir (str): Directory to save the downloaded files
        masked_only (bool): Whether to download only masked files
        
    Returns:
        tuple: (metadata_path, total_files_downloaded)
    """
    # Download metadata
    metadata_path = download_metadata()
    
    # Download sample MRI data
    total_files = download_oasis_data(data_dir, masked_only, max_subjects=n_subjects)
    
    return metadata_path, total_files 