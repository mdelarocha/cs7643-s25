# Testing Documentation

This directory contains test scripts and test data for the OASIS brain MRI analysis project.

## Directory Structure

```
tests/
├── __init__.py              # Makes tests a proper package
├── README.md                # This file
├── data/                    # Test data directory
│   ├── oasis_sample/        # Raw sample data from OASIS bucket
│   ├── processed/           # Processed data, visualizations, and splits
│   └── standardized/        # Standardized scans with different dimensions
├── test_oasis_processing.py # Main integration test script
├── test_download.py         # Script for downloading OASIS samples and testing visualization
├── test_standardization.py  # Script for demonstrating standardization of different sized scans 
└── test_utils.py            # Unit tests for utility functions
```

## Test Data

The test data is stored in the `data` directory:

- `data/oasis_sample`: Contains raw MRI files downloaded from the OASIS bucket
- `data/processed`: Contains processed MRI data, including:
  - Numpy arrays of extracted core slices
  - Visualizations of brain slices
  - Train-test split files
  - Test metadata file
- `data/standardized`: Contains standardized scans from different dimensions (176x208 vs 256x256)

## Running Tests

### Running the download and visualization test

This test downloads a few sample files from the OASIS bucket and visualizes them:

```bash
python -m tests.test_download
```

### Running the main processing test

This test downloads sample data from the OASIS bucket, processes it, extracts core slices, and creates a train-test split:

```bash
python -m tests.test_oasis_processing
```

### Running the standardization test

This test demonstrates how to standardize MRI scans with different dimensions (176x208 vs 256x256):

```bash
python -m tests.test_standardization
```

### Running unit tests

To run the utility unit tests:

```bash
python -m tests.test_utils
```

### Running all tests

To run all tests using unittest:

```bash
python -m unittest discover tests
```

## Test Components

1. **Download Test**: Downloads sample data from the OASIS bucket and visualizes it.
   - Downloads a few sample files (.nii, .nii.gz, or .gif)
   - Visualizes the downloaded NIfTI files

2. **OASIS Processing Test**: Tests the entire pipeline:
   - Downloading data from the OASIS bucket
   - Extracting core slices (140 slices from the center)
   - Visualizing brain slices
   - Creating a train-test split

3. **Standardization Test**: Demonstrates handling different scan dimensions:
   - Shows how to standardize between 176x208 and 256x256 sized scans
   - Visualizes original and standardized scans for comparison
   - Creates core slices from standardized volumes

4. **Utility Tests**: Unit tests for specific utility functions:
   - Preprocessing functions (normalization, slice extraction)
   - Train-test split functions
   
## Adding New Tests

To add new tests:

1. Create a new test file with a `test_` prefix
2. Import the necessary modules and functions
3. Add test functions using the `unittest` framework
4. Run the tests to ensure they pass

Remember to update this README when adding significant new test components. 