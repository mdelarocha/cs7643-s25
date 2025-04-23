#!/usr/bin/env python
"""
Unified runner script for the Alzheimer's detection baseline pipeline.
This script provides command-line options for running different parts
of the baseline pipeline, from data processing to model training and evaluation.
"""

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

from src.pipelines.baseline_pipeline import run_baseline_pipeline, run_simple_baseline
from src.utils.dataloader import load_oasis_metadata

def setup_logging(log_file=None):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"baseline_run_{timestamp}.log")
    elif not os.path.isabs(log_file):
        # If a relative path is provided, make it relative to the logs directory
        log_file = os.path.join(logs_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Alzheimer's detection baseline pipeline")
    
    parser.add_argument('--metadata_path', type=str, default="data/oasis-cross-sectional.csv",
                        help="Path to the metadata CSV file")
    parser.add_argument('--data_dir', type=str, default="data/raw",
                        help="Directory containing MRI files")
    parser.add_argument('--output_dir', type=str, default="outputs",
                        help="Directory to save results")
    parser.add_argument('--log_file', type=str, default=None,
                        help="Log file path (default: baseline_run_TIMESTAMP.log)")
    
    # Pipeline options
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['logistic_regression', 'random_forest'],
                        choices=['logistic_regression', 'random_forest', 'svm', 'knn', 'kmeans'],
                        help="Model types to train")
    parser.add_argument('--feature_types', type=str, nargs='+',
                        default=['statistical'],
                        choices=['statistical', 'textural'],
                        help="Feature types to extract")
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="Proportion of data for testing")
    parser.add_argument('--val_size', type=float, default=0.1,
                        help="Proportion of data for validation")
    parser.add_argument('--n_features', type=int, default=20,
                        help="Number of features to select")
    
    # Mode options
    parser.add_argument('--simple', action='store_true',
                        help="Run pipeline with simplified options")
    
    return parser.parse_args()

def main():
    """Run the baseline pipeline with provided options."""
    args = parse_args()
    logger = setup_logging(args.log_file)
    
    logger.info("Starting baseline pipeline")
    logger.info(f"Metadata: {args.metadata_path}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run pipeline
    try:
        if args.simple:
            logger.info("Running simplified baseline pipeline")
            models, results = run_simple_baseline(
                metadata_path=args.metadata_path,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                model_types=args.model_types,
                feature_types=args.feature_types
            )
        else:
            logger.info("Running full baseline pipeline")
            models, results = run_baseline_pipeline(
                metadata_path=args.metadata_path,
                data_dir=args.data_dir,
                model_types=args.model_types,
                feature_types=args.feature_types,
                output_dir=args.output_dir,
                test_size=args.test_size,
                val_size=args.val_size,
                n_features=args.n_features
            )
        
        if models is not None and results is not None:
            logger.info("Pipeline completed successfully")
            return 0
        else:
            logger.error("Pipeline failed")
            return 1
    
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 