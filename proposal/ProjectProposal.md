# Team Name: Brain Power

# Project Title: Deep Learning for Early Alzheimer’s

# Detection Using Brain MRI

## Project Summary

Problem Statement: Alzheimer’s disease is a degenerative brain disorder leading to cognitive
decline and memory loss. Given its life-altering impact, early diagnosis still continues to be a
challenge. We plan to use deep learning methods that can spot early signs of Alzheimer’s disease
from MRI brain scans in the OASIS-1 dataset.

Background/motivation: According to the Alzheimer’s Association, Alzheimer’s disease affects
millions of people worldwide, with numbers projected to more than double by 2050. Alzheimer’s is
the fifth-leading cause of death among people age 65 and older in 2021 [1]. MRI scans can provide
structural changes indicative of neurodegeneration, but conventional radiologist interpretation is
both labor-intensive and often subjective. Deep learning models can help expedite the processing
by analyzing thousands of MRI scans with consistent precision, improving both the speed and
accuracy of early Alzheimer’s diagnosis.

Expected impact: Early identification through our approach could give patients access to timely
interventions, potentially delaying symptom progression and extending quality of life.

## Proposed Approach

We will use the following approach to predict whether MRI scans of the OASIS-1 dataset can
determine if a brain is healthy or demented. We will compare multiple classification techniques to
identify the best method to detect early Alzheimer’s. For our models, we will use existing Python
libraries and machine learning frameworks, such as Pytorch. In addition, leveraging preexisting
CNN architectures such as Resnet50.

Basic Pre-processing

- Intensity normalization is recommended to standardize MRI signal variations
- Image resizing or resampling may be required depending on the model input resolution
- Each test subject’s scans will be contained entirely within the training or testing set, not
    spread across both. This will prevent turning the task onto subject memorization.
- We will organize the pre-processed data and separate the classes (mild-demented, demented,
    non-demented) based on the Clinical Dementia Rating (CDR) scores.

Traditional Machine Learning Implementation

We will implement baseline classifiers like logistic regression, K-Nearest Neighbors (KNN), and
K-Means using scikit-learn. For these models, we will attempt to extract meaningful features from
the pre-processed data such as pixel intensities, statistical measure such as mean and standard
deviation, and texture-based features such as edge information or contrast measures.


Deep Learning Implementation

We will leverage several modern CNN and Vision Transformer (ViTs) architectures such as ResNet
and EfficientNet. This will include summaries and commentary on each of their performances
against the task. An extension we are considering is testing modern, foundational large-language
models (LLMs), such as LLaMA, as a reference for modern approaches.

Data Augmentation Implementation

If needed, we will implement some basic augmentation techniques for MRI data, including:

- Random flips or rotations: Mirroring or turning images in different directions
- Small translations: Slighting shifting images in different directions
- Scaling variations: Resizing images by small random factors

Experimental Methodology

Exploratory data analysis, such as histogram analysis to visualize the distribution of pixel intensities
to understand image brightness and contrast. We will be testing and experimenting with different
model architectures to evaluate their performance against the MRI classification task. Performance
metrics dimensions would include accuracy, sensitivity, and specificity analyzed via a confusion
matrix.

## Resources and Related Work

1. The paper [2] discusses the comparison of different machine learning algorithms and deep
    learning techniques to the early identification of Alzheimer’s disease. Their performance and
    accuracy are compared. The paper also discusses the challenges and limitations of these
    methods in terms of data quality, model interpretability, and generalization.
2. The paper [3] uses magnetic resonance imaging and PET scans to train machine learning
    models such as the support vector machine, random forests, and deep learning methods such
    as convolutional neural network to identify early signs of Alzheimer’s. The authors discuss
    selecting relevant image features, imbalanced datasets, image data variability, and large-scale
    datasets to create robust models.
3. The paper [4] integrates out-of-distribution detection methods into AI models for Alzheimer’s
    diagnosis to identify cases that deviate from the normal training distribution. This method
    helped to reduce false positives and negatives, improving the model’s ability to generalize
    better.
4. The authors of the paper [5] use deep learning techniques with MRI data to diagnose dementia
    specifically in the aging population. It discusses the proper selection of MRI features and
    neural network architecture to detect various forms of dementia, including Alzheimer’s disease.
5. Several machine learning algorithms such as support vector machine, decision trees, and deep
    learning methods were used to identify Alzheimer’s patients from healthy patients. This paper
    emphasizes feature selection and pre-processing techniques to improve model generalization
    and compares different model accuracy, sensitivity, and specificity [6].


6. The paper [7] specifically focuses on a 19-layer CNN to extract features from MRI scans
    to identify Alzheimer’s patients. The performance metrics of the model were compared for
    different classes of dementia.
7. The author [8] proposed that a convolutional neural network in combination with transfer
    learning using pre-trained models provides significant improvements in Alzheimer’s detection
    accuracy and reduces the need for large amounts of data.
8. These papers [9, 10] use different CNN architectures to effectively distinguish Alzheimer’s
    patients from healthy individuals according to MRI scans.
9. Research [11] aims to create a generalizable diagnostic system that combines various diagnos-
    tic methods such as clinical data, neuroimaging, and genetic data to improve accuracy and
    robustness.

## Dataset

Dataset Name:OASIS-
Link:Accessible fromhttps://www.kaggle.com/datasets/ninadaithal/imagesoasis/dataor
https://sites.wustl.edu/oasisbrains/home/oasis-1/

- Dataset Description:OASIS-1 (Open Access Series of Imaging Studies) is a cross-sectional
    brain MRI dataset containing scans from young, middle-aged, non-demented, and demented
    older adults.
- Size: The dataset includes 416 subjects, totalling 80,000 MRI scans acquired using a 1.5T
    scanner. Each subject has multiple T1-weighted MRI scans to ensure reliability. A reliability
    data set is included containing 20 non-demented subjects imaged on a subsequent visit within
    90 days of their initial session.
- Scope:Designed for neuroimaging research, particularly in Alzheimer’s disease classification,
    brain structure analysis, and cognitive decline studies.
- Relevant Characteristics: The dataset includes CDR scores, allowing researchers to dif-
    ferentiate between healthy controls, mild cognitive impairment, and dementia patients. Ad-
    ditional metadata includes age, gender, education level, and estimated intracranial volume.

## Team Members

1. Miguel de la Rocha; malr7@gatech.edu
2. Vetrivel Kanakasabai; vkanakasabai3@gatech.edu
3. Lucy Mendez; lmendez33@gatech.edu
4. Kaitlin Timmer; ktimmer3@gatech.edu


## References

```
[1] Alzheimer’s Association. (n.d.).Alzheimer’s disease facts and figures. Retrieved [your access
date], fromhttps://www.alz.org/alzheimers-dementia/facts-figures
```
```
[2] Chakraborty, M., Naoal, N., Momen, S., & Mohammed, N. (2024). ANALYZE-AD: A com-
parative analysis of novel AI approaches for early Alzheimer’s detection.Array, 22, 100352.
https://doi.org/10.1016/j.array.2024.
```
```
[3] Khan, Y. F., Kaushik, B., & Koundal, D. (2023). Machine learning models for Alzheimer’s
disease detection using medical images. InCognitive technologies(pp. 165–182).
```
```
[4] Paleczny, A., Parab, S., & Zhang, M. (2025). Enhancing automated and early detection of
Alzheimer’s disease using out-of-distribution detection.
```
```
[5] Ntampakis, N., Diamantaras, K., Chouvarda, I., Argyriou, V., & Sarigianndis, P. (2025).
Enhanced deep learning methodologies and MRI selection techniques for dementia diagnosis
in the elderly population.
```
```
[6] Baglat, P., Salehi, A. W., Gupta, A., & Gupta, G. (2025). Multiple machine learning models
for detection of Alzheimer’s disease using OASIS dataset.
```
```
[7] Garg, G., Singh, R., Prabha, C., & Agarwal, A. (2025). An in-depth study of Alzheimer’s
detection: Leveraging OASIS MRI with a 19-layer CNN.
```
```
[8] Author unknown. An MRI Scans-Based Alzheimer’s Disease Detection via Convolutional Neu-
ral Network and Transfer Learning.
```
```
[9] Author unknown. MRI Deep Learning-Based Solution for Alzheimer’s Disease Prediction.
```
[10] Author unknown. A novel CNN architecture for accurate early detection and classification of
Alzheimer’s disease using MRI data.

[11] Author unknown. Early diagnosis of Alzheimer’s disease using machine learning: a
multi-diagnostic, generalizable approach.


