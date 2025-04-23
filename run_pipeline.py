#!/usr/bin/env python
"""
Unified runner script for the Alzheimer's detection pipeline.
This script provides a simplified interface to the entire pipeline.
"""

import os
import sys
import logging
from datetime import datetime

from src.pipelines.run_baseline import main as run_baseline_main

def setup_logging(log_file=None):
    """Set up logging configuration."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"pipeline_run_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Run the complete pipeline."""
    logger = setup_logging()
    logger.info("Starting Alzheimer's detection pipeline")
    
    # Run baseline pipeline
    logger.info("Running baseline models pipeline")
    status = run_baseline_main()
    
    if status != 0:
        logger.error("Pipeline failed")
        return 1
    
    logger.info("Pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 