# Brain Power: Deep Learning for Early Alzheimer's Detection

## Project Overview

This repository contains code for our deep learning approach to early Alzheimer's disease detection using brain MRI scans. We aim to develop and compare various machine learning and deep learning techniques for identifying early signs of Alzheimer's disease from the OASIS-1 dataset.

### Problem Statement

Alzheimer's disease is a degenerative brain disorder leading to cognitive decline and memory loss. Given its life-altering impact, early diagnosis continues to be a challenge. Our approach uses deep learning methods to identify early signs of Alzheimer's from MRI brain scans in the OASIS-1 dataset.

### Expected Impact

Early identification through our approach could give patients access to timely interventions, potentially delaying symptom progression and extending quality of life.

## Dataset

We're using the OASIS-1 (Open Access Series of Imaging Studies) dataset:
- 416 subjects aged 18-96
- ~80,000 MRI scans acquired using a 1.5T scanner
- T1-weighted MRI scans
- Clinical Dementia Rating (CDR) scores for classification:
  - 0 = Normal
  - 0.5 = Very mild dementia
  - 1 = Mild dementia
  - 2 = Moderate dementia

## Repository Structure

```
├── README.md                   # Repository documentation
├── requirements.txt            # Python dependencies
├── src/                        # Source code
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── gcs_utils.py        # Google Cloud Storage utilities
│   │   ├── preprocessing.py    # MRI preprocessing utilities
│   │   ├── dataloader.py       # Dataset loading utilities
│   │   ├── train_test_split.py # Data splitting utilities
│   │   ├── evaluation.py       # Model evaluation utilities
│   │   └── visualization.py    # Visualization utilities
│   ├── models/                 # Model implementations
│   │   ├── __init__.py
│   │   ├── baseline/           # Traditional ML models
│   │   └── deep_learning/      # Deep learning models
│   └── test_download.py        # Script to test GCS download
├── data/                       # Data directory
│   ├── raw/                    # Raw MRI data
│   ├── processed/              # Processed data
│   └── oasis-cross-sectional.csv  # Metadata
├── notebooks/                  # Jupyter notebooks for exploration
├── test_data/                  # Test data directory
├── team_members/               # Individual team member experiments
│   ├── miguel/                 # Miguel's experiments
│   ├── vetrivel/               # Vetrivel's experiments
│   ├── lucy/                   # Lucy's experiments
│   └── kaitlin/                # Kaitlin's experiments
└── proposal/                   # Project proposal documents
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Access to the OASIS-1 dataset
- Google Cloud SDK (for GCS bucket access)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mdelarocha/cs7643-s25.git
cd cs7643-s25
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure GCS access to the dataset bucket:
```bash
# Set environment variables for GCS access
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

### Dataset Access

The OASIS-1 dataset is stored in a Google Cloud Storage bucket. You can download a sample using:

```bash
python src/test_download.py
```

## Team Members

- Miguel de la Rocha (malr7@gatech.edu)
- Vetrivel Kanakasabai (vkanakasabai3@gatech.edu)
- Lucy Mendez (lmendez33@gatech.edu)
- Kaitlin Timmer (ktimmer3@gatech.edu)

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.
