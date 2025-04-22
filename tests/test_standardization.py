"""
Test script to demonstrate standardization of different sized brain scans.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.preprocessing import (
    load_mri_volume, 
    analyze_volume_dimensions,
    standardize_volume, 
    extract_core_slices
)
from src.utils.visualization import plot_brain_slice, plot_brain_three_plane

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TEST_DATA_DIR = "tests/data/oasis_sample"
OUTPUT_DIR = "tests/data/standardized"


def standardize_and_visualize():
    """
    Demonstrate standardization of MRI volumes with different dimensions.
    """
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all .img files in the test data directory
    mri_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.img')]
    
    if not mri_files:
        logger.warning("No MRI files found in the test data directory")
        return False
    
    # Dictionary to collect data for comparison visualization
    comparison_data = {}
    
    for i, mri_file in enumerate(mri_files):
        file_path = os.path.join(TEST_DATA_DIR, mri_file)
        
        # Skip if header file is missing
        hdr_path = file_path.replace('.img', '.hdr')
        if not os.path.exists(hdr_path):
            logger.warning(f"Header file for {mri_file} not found, skipping")
            continue
            
        try:
            # Load the volume
            logger.info(f"Processing {mri_file}")
            data = load_mri_volume(file_path)
            
            if data is None:
                logger.warning(f"Failed to load {mri_file}")
                continue
                
            # Analyze dimensions
            info = analyze_volume_dimensions(data)
            logger.info(f"Original dimensions: {info['shape']} - Type: {info['dimension_type']}")
            
            # Save core slices of original data
            core_data = extract_core_slices(data)
            
            # Create standardized version (176x208x176)
            standard_data = standardize_volume(data)
            logger.info(f"Standardized dimensions: {standard_data.shape}")
            
            # Save core slices of standardized data
            standard_core = extract_core_slices(standard_data)
            
            # Save for comparison visualization
            comparison_data[mri_file] = {
                'original': data,
                'standard': standard_data,
                'orig_core': core_data,
                'std_core': standard_core,
                'info': info
            }
            
            # Save middle slices for comparison
            vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Original middle slice
            fig = plot_brain_three_plane(data)
            plt.suptitle(f"Original: {info['dimension_type']} - {info['shape']}")
            plt.savefig(os.path.join(vis_dir, f"{i}_original_three_plane.png"))
            plt.close(fig)
            
            # Standardized middle slice  
            fig = plot_brain_three_plane(standard_data)
            plt.suptitle(f"Standardized to 176x208x176")
            plt.savefig(os.path.join(vis_dir, f"{i}_standardized_three_plane.png"))
            plt.close(fig)
            
            logger.info(f"Saved visualizations for {mri_file}")
            
        except Exception as e:
            logger.error(f"Error processing {mri_file}: {str(e)}")
    
    # Create side-by-side comparison of all scans
    if comparison_data:
        create_comparison_visualization(comparison_data)
    
    return True


def create_comparison_visualization(comparison_data):
    """
    Create a side-by-side comparison of original and standardized volumes.
    """
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a figure with subplots for each volume type
    num_volumes = len(comparison_data)
    fig, axes = plt.subplots(num_volumes, 2, figsize=(15, 5 * num_volumes))
    
    if num_volumes == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each volume
    for i, (filename, data) in enumerate(comparison_data.items()):
        # Plot original middle axial slice
        orig_slice_idx = data['original'].shape[2] // 2
        axes[i, 0].imshow(data['original'][:, :, orig_slice_idx].T, cmap='gray', origin='lower')
        axes[i, 0].set_title(f"Original: {data['info']['dimension_type']} - {data['info']['shape']}")
        axes[i, 0].axis('off')
        
        # Plot standardized middle axial slice
        std_slice_idx = data['standard'].shape[2] // 2
        axes[i, 1].imshow(data['standard'][:, :, std_slice_idx].T, cmap='gray', origin='lower')
        axes[i, 1].set_title(f"Standardized: 176x208x176")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "dimension_comparison.png"))
    plt.close(fig)
    
    logger.info(f"Created side-by-side comparison visualization")


def main():
    """
    Main function to demonstrate standardization of different sized scans.
    """
    logger.info("Starting standardization demonstration")
    
    # Check that we have test data
    if not os.path.exists(TEST_DATA_DIR):
        logger.error(f"Test data directory {TEST_DATA_DIR} does not exist")
        return
    
    # Standardize and visualize the different sized scans
    standardize_and_visualize()
    
    logger.info("Completed standardization demonstration")


if __name__ == "__main__":
    main() 