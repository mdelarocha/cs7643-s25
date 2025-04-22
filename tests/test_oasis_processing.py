"""
Test script to download data from the OASIS bucket, extract core slices,
and create a train-test split.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.gcs_utils import authenticate_gcs, list_bucket_contents, download_blob
from src.utils.preprocessing import preprocess_mri_file, extract_core_slices
from src.utils.visualization import plot_brain_slice, plot_brain_slices, plot_brain_three_plane
from src.utils.train_test_split import split_data_by_subject, create_stratified_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OASIS_BUCKET_NAME = "oasis-1-dataset-13635"
TEST_DATA_DIR = "tests/data/oasis_sample"
OUTPUT_DIR = "tests/data/processed"
METADATA_CSV = "data/oasis-cross-sectional.csv"
NUM_SAMPLES = 5  # Number of samples to download and process

def download_sample_data():
    """
    Download a few sample MRI files from the OASIS bucket.
    """
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    logger.info(f"Downloading {NUM_SAMPLES} sample files from OASIS bucket")
    
    client = authenticate_gcs()
    try:
        # Try different prefixes to find OASIS data
        prefixes = ["disc1/OAS1", "oasis_raw/", "OAS1_0001_MR1/", ""]
        
        for prefix in prefixes:
            logger.info(f"Searching with prefix: '{prefix}'")
            blobs = list_bucket_contents(OASIS_BUCKET_NAME, prefix=prefix, client=client)
            
            if blobs:
                logger.info(f"Found {len(blobs)} files with prefix '{prefix}'")
                
                # Find base names of .img files without downloading .hdr files yet
                img_files = [blob for blob in blobs if blob.endswith('.img')]
                
                if img_files:
                    # Try to get a mix of different dimensions by looking at filename patterns
                    size_176_pattern = [blob for blob in img_files if "_t88_" in blob][:2]  # Often 176x208x176
                    size_256_pattern = [blob for blob in img_files if "_sbj_" in blob or "_anon." in blob][:3]  # Often 256x256x...
                    
                    # Combine to ensure we have a mix of dimensions
                    diverse_samples = size_176_pattern + size_256_pattern
                    
                    if len(diverse_samples) >= NUM_SAMPLES:
                        sample_img_files = diverse_samples[:NUM_SAMPLES]
                        logger.info(f"Selected {len(sample_img_files)} sample files with diverse dimensions")
                    else:
                        # Fallback to any img files
                        sample_img_files = img_files[:NUM_SAMPLES]
                        logger.info(f"Selected {len(sample_img_files)} sample files")
                    
                    # For each .img file, also get its .hdr file
                    for img_blob in sample_img_files:
                        # Download the .img file
                        img_local_path = os.path.join(TEST_DATA_DIR, os.path.basename(img_blob))
                        download_blob(
                            bucket_name=OASIS_BUCKET_NAME,
                            source_blob_name=img_blob,
                            destination_file_path=img_local_path,
                            client=client
                        )
                        
                        # Derive the .hdr file name
                        hdr_blob = img_blob.replace('.img', '.hdr')
                        hdr_local_path = os.path.join(TEST_DATA_DIR, os.path.basename(hdr_blob))
                        
                        # Download the .hdr file
                        download_blob(
                            bucket_name=OASIS_BUCKET_NAME,
                            source_blob_name=hdr_blob,
                            destination_file_path=hdr_local_path,
                            client=client
                        )
                    
                    logger.info(f"Downloaded image and header files to {TEST_DATA_DIR}")
                    return True
        
        logger.warning("Could not find appropriate image files in the bucket")
        return False
    
    except Exception as e:
        logger.error(f"Error downloading sample data: {str(e)}")
        return False

def analyze_scan_dimensions():
    """
    Analyze the dimensions of the downloaded MRI scans.
    """
    # Get all .img files in the test data directory
    mri_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.img')]
    
    if not mri_files:
        logger.warning("No MRI files found for dimension analysis")
        return {}
    
    dimensions = {}
    for mri_file in mri_files:
        img_path = os.path.join(TEST_DATA_DIR, mri_file)
        hdr_path = img_path.replace('.img', '.hdr')
        
        if not os.path.exists(hdr_path):
            logger.warning(f"Header file missing for {mri_file}, skipping")
            continue
            
        try:
            img = nib.AnalyzeImage.load(img_path)
            data = img.get_fdata()
            
            # Handle 4D volumes
            if len(data.shape) == 4 and data.shape[3] == 1:
                data = np.squeeze(data, axis=3)
                
            dimensions[mri_file] = {
                'shape': data.shape,
                'size_mb': data.nbytes / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dimensions for {mri_file}: {str(e)}")
    
    # Summarize dimensions
    unique_dims = set(tuple(info['shape']) for info in dimensions.values())
    logger.info(f"Found {len(unique_dims)} different volume dimensions:")
    for dim in unique_dims:
        count = sum(1 for info in dimensions.values() if info['shape'] == dim)
        logger.info(f"  - Dimension {dim}: {count} files")
    
    return dimensions

def extract_and_visualize_slices():
    """
    Load downloaded MRI files, extract core slices, and visualize.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Analyze scan dimensions first
    dimensions = analyze_scan_dimensions()
    
    # Get all .img files in the test data directory
    mri_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.img')]
    
    if not mri_files:
        logger.warning("No MRI files found in the test data directory")
        return False
    
    logger.info(f"Found {len(mri_files)} MRI files for processing")
    processed_count = 0
    
    for i, mri_file in enumerate(mri_files):
        img_path = os.path.join(TEST_DATA_DIR, mri_file)
        hdr_path = img_path.replace('.img', '.hdr')
        
        # Check if both .img and .hdr files exist
        if not os.path.exists(hdr_path):
            logger.warning(f"Header file for {mri_file} not found, skipping")
            continue
        
        try:
            # Load the MRI volume using nibabel with explicit specification of Analyze format
            logger.info(f"Loading MRI file: {mri_file}")
            
            # Try to load using Analyze format
            try:
                img = nib.AnalyzeImage.load(img_path)
                logger.info(f"Loaded as Analyze format")
            except Exception as e:
                logger.warning(f"Failed to load as Analyze, trying generic loader: {str(e)}")
                img = nib.load(img_path)
            
            data = img.get_fdata()
            
            logger.info(f"Original volume shape: {data.shape}")
            
            # If it's a 4D volume with singleton 4th dimension, squeeze it out
            if len(data.shape) == 4 and data.shape[3] == 1:
                data = np.squeeze(data, axis=3)
                logger.info(f"Squeezed volume shape: {data.shape}")
            
            # Check if the shape is reasonable for processing
            if len(data.shape) != 3 or min(data.shape) < 10:
                logger.warning(f"Unusual volume shape after squeezing: {data.shape}, skipping")
                continue
            
            # Extract core slices along the 3rd dimension (axis 2)
            # Calculate start and end indices for the core slices
            if data.shape[2] > 140:
                start = (data.shape[2] - 140) // 2
                end = start + 140
                logger.info(f"Extracting 140 core slices from {data.shape[2]} total slices")
            else:
                # If the volume has fewer than 140 slices, use all of them
                start = 0
                end = data.shape[2]
                logger.info(f"Volume has only {data.shape[2]} slices, using all available slices")
            
            core_slices = data[:, :, start:end]
            
            logger.info(f"Core slices shape: {core_slices.shape}")
            
            # Log information about different dimensions
            if (data.shape[0] == 176 and data.shape[1] == 208) or (data.shape[0] == 208 and data.shape[1] == 176):
                logger.info(f"This is a 176x208 type scan (likely transformed to standard space)")
            elif data.shape[0] == 256 and data.shape[1] == 256:
                logger.info(f"This is a 256x256 type scan (likely in original acquisition space)")
            else:
                logger.info(f"This scan has non-standard dimensions: {data.shape[0]}x{data.shape[1]}")
            
            # Save visualization of original and core slices
            vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Visualize original middle slice
            fig = plot_brain_slice(data)
            plt.savefig(os.path.join(vis_dir, f"{i}_original_slice.png"))
            plt.close(fig)
            
            # Visualize core middle slice
            fig = plot_brain_slice(core_slices)
            plt.savefig(os.path.join(vis_dir, f"{i}_core_slice.png"))
            plt.close(fig)
            
            # Visualize multiple slices from the core
            fig = plot_brain_slices(core_slices, n_slices=9)
            plt.savefig(os.path.join(vis_dir, f"{i}_core_multiple_slices.png"))
            plt.close(fig)
            
            # Visualize three orthogonal planes
            fig = plot_brain_three_plane(core_slices)
            plt.savefig(os.path.join(vis_dir, f"{i}_three_plane.png"))
            plt.close(fig)
            
            # Save the core slices as a numpy array
            np.save(os.path.join(OUTPUT_DIR, f"core_slices_{i}.npy"), core_slices)
            logger.info(f"Saved core slices for {mri_file}")
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing MRI file {mri_file}: {str(e)}")
    
    logger.info(f"Completed extraction and visualization of slices - processed {processed_count} files")
    return processed_count > 0

def create_dummy_metadata():
    """
    Create a dummy metadata file for testing train-test split.
    """
    # Get the processed files
    processed_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
    
    if not processed_files:
        logger.warning("No processed files found for creating metadata")
        return None
    
    # Create dummy metadata with a more even distribution of CDR scores
    # to avoid stratification issues with small sample size
    subject_ids = [f"OAS1_{i:04d}" for i in range(1, len(processed_files) + 1)]
    
    # For a small test dataset, ensure we have at least 2 samples per class
    if len(processed_files) < 8:
        # For 5 samples, create 3 normal (0) and 2 with mild dementia (0.5)
        cdr_scores = [0, 0, 0, 0.5, 0.5][:len(processed_files)]
    else:
        cdr_scores = np.random.choice([0, 0.5, 1, 2], size=len(processed_files), p=[0.6, 0.2, 0.15, 0.05])
    
    metadata = pd.DataFrame({
        'Subject ID': subject_ids,
        'MRI_file': processed_files,
        'CDR': cdr_scores,
        'Age': np.random.uniform(65, 95, size=len(processed_files)),
        'Gender': np.random.choice(['M', 'F'], size=len(processed_files))
    })
    
    # Save the metadata
    metadata_path = os.path.join(OUTPUT_DIR, "test_metadata.csv")
    metadata.to_csv(metadata_path, index=False)
    logger.info(f"Created dummy metadata with {len(metadata)} entries and saved to {metadata_path}")
    
    return metadata

def test_train_test_split(metadata):
    """
    Test train-test split on the dummy metadata.
    """
    if metadata is None:
        # Try to load from file
        metadata_path = os.path.join(OUTPUT_DIR, "test_metadata.csv")
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
        else:
            logger.error("No metadata available for train-test split")
            return False
    
    logger.info("Testing train-test split on metadata")
    
    # Test split by subject
    train_df, val_df, test_df = split_data_by_subject(
        metadata, test_size=0.2, val_size=0.1, random_state=42
    )
    
    logger.info(f"Split by subject: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Check if we have enough samples for stratification
    # Count min samples per class
    class_counts = metadata['CDR'].value_counts()
    min_samples_per_class = class_counts.min()
    
    if min_samples_per_class >= 2:
        try:
            # Test stratified split
            train_df2, val_df2, test_df2 = create_stratified_split(
                metadata, label_column='CDR', test_size=0.2, val_size=0.1, random_state=42
            )
            
            logger.info(f"Stratified split: Train={len(train_df2)}, Val={len(val_df2)}, Test={len(test_df2)}")
            
            # Save both splits
            split_dir = os.path.join(OUTPUT_DIR, "splits")
            os.makedirs(split_dir, exist_ok=True)
            
            train_df.to_csv(os.path.join(split_dir, "train_subject.csv"), index=False)
            val_df.to_csv(os.path.join(split_dir, "val_subject.csv"), index=False)
            test_df.to_csv(os.path.join(split_dir, "test_subject.csv"), index=False)
            
            train_df2.to_csv(os.path.join(split_dir, "train_stratified.csv"), index=False)
            val_df2.to_csv(os.path.join(split_dir, "val_stratified.csv"), index=False)
            test_df2.to_csv(os.path.join(split_dir, "test_stratified.csv"), index=False)
        
        except Exception as e:
            logger.warning(f"Stratified split failed: {str(e)}")
            logger.info("Saving only subject-based split")
            
            # Save only the subject-based split
            split_dir = os.path.join(OUTPUT_DIR, "splits")
            os.makedirs(split_dir, exist_ok=True)
            
            train_df.to_csv(os.path.join(split_dir, "train_subject.csv"), index=False)
            val_df.to_csv(os.path.join(split_dir, "val_subject.csv"), index=False)
            test_df.to_csv(os.path.join(split_dir, "test_subject.csv"), index=False)
    
    else:
        logger.warning(f"Not enough samples per class for stratification (min={min_samples_per_class})")
        logger.info("Saving only subject-based split")
        
        # Save only the subject-based split
        split_dir = os.path.join(OUTPUT_DIR, "splits")
        os.makedirs(split_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(split_dir, "train_subject.csv"), index=False)
        val_df.to_csv(os.path.join(split_dir, "val_subject.csv"), index=False)
        test_df.to_csv(os.path.join(split_dir, "test_subject.csv"), index=False)
    
    logger.info(f"Saved train-test splits to {split_dir}")
    return True

def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting OASIS dataset processing test")
    
    # Step 1: Download sample data
    if not os.path.exists(TEST_DATA_DIR) or len(os.listdir(TEST_DATA_DIR)) == 0:
        download_sample_data()
    else:
        logger.info(f"Using existing data in {TEST_DATA_DIR}")
    
    # Step 2: Extract and visualize core slices
    extract_and_visualize_slices()
    
    # Step 3: Create dummy metadata for testing
    metadata = create_dummy_metadata()
    
    # Step 4: Test train-test split
    if metadata is not None:
        test_train_test_split(metadata)
    else:
        logger.warning("Skipping train-test split due to lack of metadata")
    
    logger.info("Completed OASIS dataset processing test")

if __name__ == "__main__":
    main() 