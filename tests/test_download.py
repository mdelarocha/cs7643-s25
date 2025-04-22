"""
Test script to download a sample of the OASIS-1 dataset.
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.gcs_utils import authenticate_gcs, list_bucket_contents, download_blob
from src.utils.visualization import plot_brain_slice, plot_brain_slices, plot_brain_three_plane

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OASIS_BUCKET_NAME = "oasis-1-dataset-13635"
TEST_DATA_DIR = "tests/data/oasis_sample"


def test_download_sample() -> None:
    """
    Download a small sample of the OASIS-1 dataset for testing purposes.
    Downloads a few files from disc1 or oasis_raw directory.
    """
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    logger.info("Downloading a sample of OASIS-1 dataset for testing")
    
    # First, list available files in the bucket
    client = authenticate_gcs()
    try:
        # Try different prefixes to find data
        prefixes = ["disc1/", "oasis_raw/", "OAS1_0001_MR1/", ""]
        found_files = False
        
        for prefix in prefixes:
            logger.info(f"Searching with prefix: '{prefix}'")
            # List contents with the prefix
            blobs = list_bucket_contents(bucket_name=OASIS_BUCKET_NAME, prefix=prefix, client=client)
            
            if blobs:
                logger.info(f"Found {len(blobs)} files with prefix '{prefix}'")
                
                # Look for .img and .hdr files to ensure we get MRI scans with potentially different dimensions
                img_files = [blob for blob in blobs if blob.endswith('.img')]
                
                # Filter for different image styles/dimensions if possible by looking at patterns in filenames
                # Some scans are 176 pixels while others are 256 pixels in certain dimensions
                size_176_pattern = [blob for blob in img_files if "_t88_" in blob][:2]  # Often 176x208x176
                size_256_pattern = [blob for blob in img_files if "_sbj_" in blob or "_anon." in blob][:2]  # Often 256x256x...
                
                # Combine to get a mix of dimensions
                diverse_samples = size_176_pattern + size_256_pattern
                
                if diverse_samples:
                    logger.info(f"Found {len(diverse_samples)} MRI files with diverse dimensions")
                    sample_files = diverse_samples
                elif img_files:
                    # If we couldn't identify by patterns, just take a few image files
                    logger.info(f"Found {len(img_files)} .img files")
                    sample_files = img_files[:4]
                else:
                    # Fallback to any files
                    logger.info("No MRI files found, using other files")
                    sample_files = blobs[:4]
                
                if sample_files:
                    found_files = True
                    logger.info(f"Downloading {len(sample_files)} sample files")
                    # Download each sample file and its header file if it's an .img file
                    for blob_name in sample_files:
                        local_path = os.path.join(TEST_DATA_DIR, os.path.basename(blob_name))
                        download_blob(
                            bucket_name=OASIS_BUCKET_NAME,
                            source_blob_name=blob_name,
                            destination_file_path=local_path,
                            client=client
                        )
                        
                        # If it's an .img file, also download the .hdr file
                        if blob_name.endswith('.img'):
                            hdr_blob = blob_name.replace('.img', '.hdr')
                            hdr_local_path = os.path.join(TEST_DATA_DIR, os.path.basename(hdr_blob))
                            download_blob(
                                bucket_name=OASIS_BUCKET_NAME,
                                source_blob_name=hdr_blob,
                                destination_file_path=hdr_local_path,
                                client=client
                            )
                    
                    logger.info(f"Successfully downloaded sample files to {TEST_DATA_DIR}")
                    break
            
        if not found_files:
            logger.warning("Could not find any files in the bucket with the tried prefixes")
            
    except Exception as e:
        logger.error(f"Error downloading sample files: {str(e)}")


def detect_dimension_differences():
    """
    Detect differences in scan dimensions from the downloaded sample data.
    """
    if not os.path.exists(TEST_DATA_DIR):
        logger.error(f"Test directory {TEST_DATA_DIR} does not exist")
        return
    
    # Find .img files
    img_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.img')]
    
    if not img_files:
        logger.warning("No .img files found in test directory")
        return
    
    logger.info("Checking scan dimensions for different files:")
    dimension_info = []
    
    for img_file in img_files:
        file_path = os.path.join(TEST_DATA_DIR, img_file)
        try:
            # Check if header file exists
            hdr_path = file_path.replace('.img', '.hdr')
            if not os.path.exists(hdr_path):
                logger.warning(f"Header file for {img_file} not found, skipping")
                continue
                
            # Load the Analyze image file
            img = nib.AnalyzeImage.load(file_path)
            data = img.get_fdata()
            
            # Handle 4D volumes by squeezing out singleton dimensions
            if len(data.shape) == 4 and data.shape[3] == 1:
                data = np.squeeze(data, axis=3)
            
            # Record dimension information
            dimension_info.append({
                'filename': img_file,
                'dimensions': data.shape,
                'size_mb': data.nbytes / (1024 * 1024)
            })
            
            logger.info(f"File: {img_file} | Dimensions: {data.shape} | Size: {data.nbytes / (1024 * 1024):.2f} MB")
        
        except Exception as e:
            logger.error(f"Error reading {img_file}: {str(e)}")
    
    # Report on dimension variations
    if dimension_info:
        unique_dims = set(tuple(item['dimensions']) for item in dimension_info)
        logger.info(f"Found {len(unique_dims)} different dimension sizes in the sample data:")
        for dim in unique_dims:
            matching_files = [item['filename'] for item in dimension_info if item['dimensions'] == dim]
            logger.info(f"  Dimensions {dim}: {len(matching_files)} files")
    
    return dimension_info


def test_visualization() -> None:
    """
    Test visualization utilities on downloaded sample MRI files.
    """
    if not os.path.exists(TEST_DATA_DIR):
        logger.error(f"Test directory {TEST_DATA_DIR} does not exist")
        return
    
    # Find MRI files (both Analyze and NIfTI formats)
    mri_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith(('.img', '.nii', '.nii.gz'))]
    
    if not mri_files:
        logger.warning("No MRI files found in test directory")
        return
    
    vis_dir = os.path.join(TEST_DATA_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    for mri_file in mri_files:
        file_path = os.path.join(TEST_DATA_DIR, mri_file)
        logger.info(f"Visualizing {mri_file}")
        
        try:
            # Load the MRI file based on its format
            if mri_file.endswith('.img'):
                # Check if header file exists
                hdr_path = file_path.replace('.img', '.hdr')
                if not os.path.exists(hdr_path):
                    logger.warning(f"Header file for {mri_file} not found, skipping")
                    continue
                
                # Load Analyze format
                img = nib.AnalyzeImage.load(file_path)
            else:
                # Load NIfTI format
                img = nib.load(file_path)
                
            data = img.get_fdata()
            
            # Handle 4D volumes by squeezing out singleton dimensions
            if len(data.shape) == 4 and data.shape[3] == 1:
                data = np.squeeze(data, axis=3)
                logger.info(f"Squeezed 4D volume to 3D: {data.shape}")
                
            # Test dimension handling with standardized core slice extraction
            if data.shape[2] > 140:
                start = (data.shape[2] - 140) // 2
                end = start + 140
                core_slices = data[:, :, start:end]
                logger.info(f"Extracted core slices: shape {core_slices.shape} from original shape {data.shape}")
            else:
                logger.info(f"Volume has fewer than 140 slices: {data.shape[2]}, using all slices")
                core_slices = data
            
            # Visualize middle slice
            fig = plot_brain_slice(data)
            plt.savefig(os.path.join(vis_dir, f"{mri_file}_slice.png"))
            plt.close(fig)
            
            # Visualize multiple slices
            fig = plot_brain_slices(data, n_slices=9)
            plt.savefig(os.path.join(vis_dir, f"{mri_file}_slices.png"))
            plt.close(fig)
            
            # Visualize three planes
            fig = plot_brain_three_plane(data)
            plt.savefig(os.path.join(vis_dir, f"{mri_file}_three_plane.png"))
            plt.close(fig)
            
            logger.info(f"Saved visualizations for {mri_file} to {vis_dir}")
            
        except Exception as e:
            logger.error(f"Error visualizing {mri_file}: {str(e)}")


def main():
    """
    Main function to run tests.
    """
    logger.info("Starting OASIS dataset download and visualization test")
    
    # First download sample files
    test_download_sample()
    
    # Check for dimension differences in the downloaded data
    logger.info("Detecting dimension differences in the downloaded MRI files")
    detect_dimension_differences()
    
    # Test visualization with different dimensions
    logger.info("Testing visualization with different dimensions")
    test_visualization()
    
    logger.info("Completed download and visualization test")


if __name__ == "__main__":
    main() 