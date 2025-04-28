# Alzheimer's Detection Project

This project aims to detect Alzheimer's disease using MRI data.

## Project Structure

The project is organized primarily by team member contributions.

```
project_root/
├── data/
│   ├── raw/          # Raw MRI scans (e.g., NIfTI, ANALYZE)
│   └── oasis-cross-sectional.csv # Metadata file
├── notebooks/        # Jupyter notebooks for exploration, visualization
├── outputs/          # Directory for pipeline outputs (models, results, plots)
│   └── ...           # Timestamped output directories from runs
├── team_members/     # Code organized by team member
│   └── miguel/       # Code developed by Miguel
│       ├── src/        # Source code (pipelines, models, features)
│       └── README.md # Details specific to Miguel's code
├── tests/            # Unit and integration tests (if applicable per member)
├── requirements.txt  # Project dependencies (consider member-specific if needed)
└── README.md         # This file
```

## Contributions

This section outlines the primary contributions of each team member. Refer to the individual `README.md` files within each `team_members/<name>/` directory for more specific details (if available).

*   **Miguel de la Rocha:** Handled data loading and extraction for OASIS-1 dataset, MRI data preprocessing, and exploration. Implemented and analyzed traditional machine learning models (Logistic Regression, Random Forest, KNN, K-Means) via the baseline pipeline framework (`src/pipelines/run_baseline.py`). Contributed to the final report. See `team_members/miguel/` for detailed code and usage.

*   **Vetrivel Kanakasabai:** Explored MRI scan slices and handled MRI data preprocessing. Implemented, trained, and analyzed the ConvMixer deep learning model (`ConvMixer.ipynb`, `data_check.ipynb`). Contributed to the final report. See `team_members/vetrivel/`.

*   **Lucy Mendez:** Focused on preprocessing, data augmentation, and exploration of MRI slices. Implemented, trained, and analyzed the EfficientNet deep learning model (`efficientnetv2_model_3classes.py`), including performance visualizations. Handled project team meeting logistics. Contributed to the final report. See `team_members/lucy/`.

*   **Kaitlin Timmer:** Worked on visualization of MRI scans, data augmentation, and preprocessing via Otsu's method. Implemented, trained, and analyzed the ResNeXt deep learning model. Key code contributions include data handling (`data.py`, `data_exploration.py`). Contributed to the final report. See `team_members/kaitlin/`.
