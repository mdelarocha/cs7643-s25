#!/usr/bin/env python
"""
Script to download the OASIS dataset using the consolidated download utilities.
"""

import os
import logging
import argparse
from src.utils.download import download_metadata, download_oasis_data, download_sample_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download OASIS dataset")
    parser.add_argument("--sample", action="store_true", 
                        help="Download only a sample of the dataset (5 subjects)")
    parser.add_argument("--masked_only", action="store_true",
                        help="Download only masked MRI files (recommended)")
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Maximum number of subjects to download")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Directory to save the downloaded files")
    parser.add_argument("--metadata_only", action="store_true",
                        help="Download only the metadata file")
    return parser.parse_args()

def main():
    """Main function to download the dataset."""
    args = parse_args()
    
    logger.info("Starting OASIS dataset download")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download metadata
    metadata_path = download_metadata()
    if not metadata_path:
        logger.error("Failed to download metadata file")
        return 1
    
    # If metadata only, exit
    if args.metadata_only:
        logger.info("Downloaded metadata file successfully")
        return 0
    
    # Download sample or full dataset
    if args.sample:
        metadata_path, total_files = download_sample_data(
            n_subjects=5,
            data_dir=args.output_dir,
            masked_only=args.masked_only
        )
        logger.info(f"Downloaded sample dataset: {total_files} files")
    else:
        total_files = download_oasis_data(
            data_dir=args.output_dir,
            masked_only=args.masked_only,
            max_subjects=args.max_subjects
        )
        logger.info(f"Downloaded dataset: {total_files} files")
    
    if total_files > 0:
        logger.info("Dataset download completed successfully")
        return 0
    else:
        logger.warning("No new files were downloaded")
        return 0

if __name__ == "__main__":
    exit(main()) 