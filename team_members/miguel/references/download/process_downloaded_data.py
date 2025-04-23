#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process downloaded MRI data, extract features, and prepare for model training.
"""

import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from src.utils.dataloader import load_mri_files, load_oasis_metadata
from src.utils.preprocessing import preprocess_mri_file
from src.features.build_features import extract_features_from_volume

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data/raw/OASIS_EXPANDED"
METADATA_PATH = "data/raw/oasis_cross-sectional.csv"
PROCESSED_DIR = "data/processed"
FEATURES_DIR = "data/features"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process downloaded MRI data and extract features"
    )
    parser.add_argument(
        "--extract_slices",
        action="store_true",
        help="Extract 2D slices from 3D volumes",
    )
    parser.add_argument(
        "--extract_features",
        action="store_true",
        help="Extract features from MRI volumes",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process",
    )
    
    return parser.parse_args()


def find_mri_files_in_subject_dirs():
    """
    Find all MRI files in the subject directories.
    
    Returns:
        dict: Dictionary mapping subject IDs to their MRI files
    """
    subject_files = defaultdict(list)
    
    # Get all subject directories
    subject_dirs = glob.glob(os.path.join(DATA_DIR, "OAS1_*"))
    logger.info(f"Found {len(subject_dirs)} subject directories")
    
    for subject_dir in tqdm(subject_dirs, desc="Finding MRI files"):
        subject_id = os.path.basename(subject_dir)
        
        # Find all masked img files (not the header files)
        img_files = glob.glob(os.path.join(subject_dir, "*.img"))
        
        for img_file in img_files:
            # Check if this is a masked file
            if "masked" in img_file or "fseg" in img_file:
                subject_files[subject_id].append(img_file)
    
    # Filter out subjects with no masked files
    subject_files = {k: v for k, v in subject_files.items() if v}
    
    logger.info(f"Found {len(subject_files)} subjects with masked MRI files")
    
    return subject_files


def preprocess_and_save_volumes(subject_files, metadata_df, max_subjects=None):
    """
    Preprocess MRI volumes and save them to disk.
    
    Args:
        subject_files (dict): Dictionary mapping subject IDs to MRI files
        metadata_df (pd.DataFrame): Metadata DataFrame
        max_subjects (int, optional): Maximum number of subjects to process
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Create a mapping of subject ID to CDR score
    subject_to_cdr = {}
    for _, row in metadata_df.iterrows():
        subject_id = row['ID']
        # Extract subject ID in the format OAS1_XXXX_MRX
        if isinstance(subject_id, str) and subject_id.startswith('OAS1'):
            subject_to_cdr[subject_id] = row['CDR']
    
    # Process only a subset of subjects if specified
    if max_subjects is not None:
        subject_ids = list(subject_files.keys())[:max_subjects]
    else:
        subject_ids = list(subject_files.keys())
    
    logger.info(f"Processing {len(subject_ids)} subjects")
    
    processed_count = 0
    skipped_count = 0
    
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        # Skip if subject not in metadata
        if subject_id not in subject_to_cdr:
            logger.warning(f"Subject {subject_id} not found in metadata, skipping")
            skipped_count += 1
            continue
        
        cdr_score = subject_to_cdr[subject_id]
        
        # Process only subjects with valid CDR scores
        if pd.isna(cdr_score):
            logger.warning(f"Subject {subject_id} has no CDR score, skipping")
            skipped_count += 1
            continue
            
        # Create output directory for this subject
        subject_dir = os.path.join(PROCESSED_DIR, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Get all MRI files for this subject
        mri_files = subject_files[subject_id]
        
        # Use only the first masked file for now
        if mri_files:
            mri_file = mri_files[0]
            
            try:
                # Load and preprocess the MRI volume
                volume = preprocess_mri_file(
                    mri_file, 
                    normalize=True,
                    extract_core=True
                )
                
                # Save the preprocessed volume
                output_path = os.path.join(subject_dir, "volume.npy")
                np.save(output_path, volume)
                
                # Save the metadata
                metadata = {
                    "subject_id": subject_id,
                    "cdr_score": cdr_score,
                    "file_path": mri_file,
                    "shape": volume.shape,
                }
                
                metadata_path = os.path.join(subject_dir, "metadata.csv")
                pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {mri_file}: {str(e)}")
                skipped_count += 1
    
    logger.info(f"Processed {processed_count} subjects, skipped {skipped_count} subjects")


def extract_features(max_subjects=None):
    """
    Extract features from preprocessed MRI volumes.
    
    Args:
        max_subjects (int, optional): Maximum number of subjects to process
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # Get all subject directories
    subject_dirs = glob.glob(os.path.join(PROCESSED_DIR, "OAS1_*"))
    logger.info(f"Found {len(subject_dirs)} preprocessed subject directories")
    
    # Process only a subset of subjects if specified
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    
    features_list = []
    
    for subject_dir in tqdm(subject_dirs, desc="Extracting features"):
        subject_id = os.path.basename(subject_dir)
        
        # Check if volume and metadata exist
        volume_path = os.path.join(subject_dir, "volume.npy")
        metadata_path = os.path.join(subject_dir, "metadata.csv")
        
        if not os.path.exists(volume_path) or not os.path.exists(metadata_path):
            logger.warning(f"Missing volume or metadata for {subject_id}, skipping")
            continue
        
        # Load volume and metadata
        volume = np.load(volume_path)
        metadata = pd.read_csv(metadata_path)
        
        # Extract features
        try:
            features = extract_features_from_volume(volume)
            
            # Add subject info
            features["subject_id"] = subject_id
            features["cdr_score"] = metadata["cdr_score"].iloc[0]
            features["has_dementia"] = 1 if features["cdr_score"] > 0 else 0
            
            features_list.append(features)
            
        except Exception as e:
            logger.error(f"Error extracting features for {subject_id}: {str(e)}")
    
    # Create features DataFrame
    if features_list:
        features_df = pd.DataFrame(features_list)
        
        # Save features
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(FEATURES_DIR, f"mri_features_{timestamp}.csv")
        features_df.to_csv(output_path, index=False)
        
        logger.info(f"Extracted features for {len(features_list)} subjects")
        logger.info(f"Saved features to {output_path}")
    else:
        logger.warning("No features extracted")


def extract_2d_slices(max_subjects=None):
    """
    Extract 2D slices from 3D MRI volumes.
    
    Args:
        max_subjects (int, optional): Maximum number of subjects to process
    """
    slices_dir = os.path.join(PROCESSED_DIR, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    
    # Get all subject directories
    subject_dirs = glob.glob(os.path.join(PROCESSED_DIR, "OAS1_*"))
    logger.info(f"Found {len(subject_dirs)} preprocessed subject directories")
    
    # Process only a subset of subjects if specified
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    
    slices_metadata = []
    
    for subject_dir in tqdm(subject_dirs, desc="Extracting 2D slices"):
        subject_id = os.path.basename(subject_dir)
        
        # Check if volume and metadata exist
        volume_path = os.path.join(subject_dir, "volume.npy")
        metadata_path = os.path.join(subject_dir, "metadata.csv")
        
        if not os.path.exists(volume_path) or not os.path.exists(metadata_path):
            logger.warning(f"Missing volume or metadata for {subject_id}, skipping")
            continue
        
        # Load volume and metadata
        volume = np.load(volume_path)
        metadata = pd.read_csv(metadata_path)
        cdr_score = metadata["cdr_score"].iloc[0]
        
        # Extract middle slices from each axis
        try:
            # Create directory for this subject's slices
            subject_slices_dir = os.path.join(slices_dir, subject_id)
            os.makedirs(subject_slices_dir, exist_ok=True)
            
            # Extract slices from all 3 axes
            for axis in range(3):
                # Get middle slice index and some surrounding slices
                middle_idx = volume.shape[axis] // 2
                slice_indices = [
                    middle_idx - 4,
                    middle_idx - 2,
                    middle_idx,
                    middle_idx + 2,
                    middle_idx + 4
                ]
                
                # Filter out invalid indices
                slice_indices = [idx for idx in slice_indices 
                                if 0 <= idx < volume.shape[axis]]
                
                # Extract and save slices
                for slice_idx in slice_indices:
                    if axis == 0:
                        slice_data = volume[slice_idx, :, :]
                    elif axis == 1:
                        slice_data = volume[:, slice_idx, :]
                    else:  # axis == 2
                        slice_data = volume[:, :, slice_idx]
                    
                    # Save slice
                    slice_filename = f"axis{axis}_slice{slice_idx}.npy"
                    slice_path = os.path.join(subject_slices_dir, slice_filename)
                    np.save(slice_path, slice_data)
                    
                    # Add to metadata
                    slices_metadata.append({
                        "subject_id": subject_id,
                        "cdr_score": cdr_score,
                        "has_dementia": 1 if cdr_score > 0 else 0,
                        "axis": axis,
                        "slice_index": slice_idx,
                        "slice_path": slice_path
                    })
            
        except Exception as e:
            logger.error(f"Error extracting slices for {subject_id}: {str(e)}")
    
    # Create slices metadata DataFrame
    if slices_metadata:
        slices_df = pd.DataFrame(slices_metadata)
        
        # Save metadata
        metadata_path = os.path.join(slices_dir, "slices_metadata.csv")
        slices_df.to_csv(metadata_path, index=False)
        
        logger.info(f"Extracted {len(slices_metadata)} slices from {len(subject_dirs)} subjects")
        logger.info(f"Saved slices metadata to {metadata_path}")
    else:
        logger.warning("No slices extracted")


def main():
    """Main function to process MRI data."""
    args = parse_args()
    
    logger.info("Starting MRI data processing")
    
    # Find all MRI files in subject directories
    subject_files = find_mri_files_in_subject_dirs()
    
    # Load metadata
    try:
        metadata_df = load_oasis_metadata(METADATA_PATH)
        logger.info(f"Loaded metadata for {len(metadata_df)} subjects")
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return
    
    # Preprocess and save MRI volumes
    preprocess_and_save_volumes(subject_files, metadata_df, args.max_subjects)
    
    # Extract 2D slices if requested
    if args.extract_slices:
        extract_2d_slices(args.max_subjects)
    
    # Extract features if requested
    if args.extract_features:
        extract_features(args.max_subjects)
    
    logger.info("MRI data processing completed")


if __name__ == "__main__":
    main() 