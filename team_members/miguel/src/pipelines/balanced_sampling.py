"""
Sampling techniques for handling class imbalance in Alzheimer's disease classification.
"""

import numpy as np
import pandas as pd
import logging
from collections import Counter
from sklearn.utils import resample

# Import imblearn classes for various resampling techniques
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_class_distribution(y):
    """
    Get the distribution of classes in the target variable.
    
    Args:
        y (numpy.ndarray): Target labels
        
    Returns:
        dict: Class distribution with counts and percentages (JSON serializable)
    """
    if y is None:
        logger.warning("Input array y is None for get_class_distribution")
        return {}
        
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    distribution = {}
    if total > 0:
        for i, (cls, count) in enumerate(zip(unique, counts)):
            # Convert numpy types to standard python types for JSON serialization
            distribution[int(cls)] = { # Convert class label (key) to int
                'count': int(count), # Convert count to int
                'percentage': float((count / total) * 100) # Convert percentage to float
            }
    
    logger.info(f"Class distribution: {distribution}")
    return distribution

def calculate_class_weights(y):
    """
    Calculate class weights inversely proportional to class frequencies.
    
    Args:
        y (numpy.ndarray): Target labels
        
    Returns:
        dict: Class weights dictionary
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    class_weights = {}
    for cls, count in zip(unique, counts):
        # Weight inversely proportional to frequency
        class_weights[cls] = total / (len(unique) * count)
    
    logger.info(f"Calculated class weights: {class_weights}")
    return class_weights

def manual_resample(X, y, strategy='oversample', random_state=42):
    """
    Manually resample the dataset using simple techniques.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        strategy (str): 'oversample', 'undersample', or 'combine'
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    classes, counts = np.unique(y, return_counts=True)
    
    # Convert to DataFrame for easier manipulation
    data = pd.DataFrame(X)
    data['target'] = y
    
    if strategy == 'oversample':
        # Find the majority class count
        max_count = max(counts)
        resampled_dfs = []
        
        for cls in classes:
            cls_data = data[data['target'] == cls]
            
            # If this is a minority class, oversample it
            if len(cls_data) < max_count:
                resampled = resample(
                    cls_data,
                    replace=True,
                    n_samples=max_count,
                    random_state=random_state
                )
                resampled_dfs.append(resampled)
            else:
                resampled_dfs.append(cls_data)
        
        # Combine all resampled classes
        resampled_data = pd.concat(resampled_dfs)
        
    elif strategy == 'undersample':
        # Find the minority class count
        min_count = min(counts)
        resampled_dfs = []
        
        for cls in classes:
            cls_data = data[data['target'] == cls]
            
            # If this is a majority class, undersample it
            if len(cls_data) > min_count:
                resampled = resample(
                    cls_data,
                    replace=False,
                    n_samples=min_count,
                    random_state=random_state
                )
                resampled_dfs.append(resampled)
            else:
                resampled_dfs.append(cls_data)
        
        # Combine all resampled classes
        resampled_data = pd.concat(resampled_dfs)
        
    elif strategy == 'combine':
        # Use a combination of over and under sampling
        # First identify the median class size
        median_count = np.median(counts)
        resampled_dfs = []
        
        for cls in classes:
            cls_data = data[data['target'] == cls]
            cls_count = len(cls_data)
            
            if cls_count < median_count:
                # Oversample minority classes
                resampled = resample(
                    cls_data,
                    replace=True,
                    n_samples=int(median_count),
                    random_state=random_state
                )
                resampled_dfs.append(resampled)
            elif cls_count > median_count * 2:
                # Undersample severely overrepresented classes
                resampled = resample(
                    cls_data,
                    replace=False,
                    n_samples=int(median_count * 1.5),
                    random_state=random_state
                )
                resampled_dfs.append(resampled)
            else:
                # Keep moderately represented classes as is
                resampled_dfs.append(cls_data)
        
        # Combine all resampled classes
        resampled_data = pd.concat(resampled_dfs)
    else:
        logger.warning(f"Unknown resampling strategy: {strategy}. Using original data.")
        return X, y
    
    # Shuffle the resampled data
    resampled_data = resampled_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Extract features and target
    X_resampled = resampled_data.drop('target', axis=1).values
    y_resampled = resampled_data['target'].values
    
    # Log resampling results
    original_dist = dict(zip(classes, counts))
    new_dist = dict(Counter(y_resampled))
    logger.info(f"Original distribution: {original_dist}")
    logger.info(f"Resampled distribution: {new_dist}")
    
    return X_resampled, y_resampled

def apply_smote(X, y, k_neighbors=5, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced data.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        k_neighbors (int): Number of nearest neighbors to use
        sampling_strategy (str or dict): Sampling strategy
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled) or original data if imblearn not available
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("imblearn package not available. Install with: pip install imbalanced-learn")
        return X, y
    
    try:
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Log resampling results
        original_dist = dict(Counter(y))
        new_dist = dict(Counter(y_resampled))
        logger.info(f"Original distribution: {original_dist}")
        logger.info(f"SMOTE resampled distribution: {new_dist}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"Error applying SMOTE: {str(e)}")
        return X, y

def apply_smote_tomek(X, y, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE+Tomek (combined over and under-sampling) to handle imbalanced data.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        sampling_strategy (str or dict): Sampling strategy
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled) or original data if imblearn not available
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("imblearn package not available. Install with: pip install imbalanced-learn")
        return X, y
    
    try:
        # Apply SMOTE+Tomek
        smote_tomek = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        # Log resampling results
        original_dist = dict(Counter(y))
        new_dist = dict(Counter(y_resampled))
        logger.info(f"Original distribution: {original_dist}")
        logger.info(f"SMOTE+Tomek resampled distribution: {new_dist}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"Error applying SMOTE+Tomek: {str(e)}")
        return X, y

def apply_adasyn(X, y, sampling_strategy='auto', random_state=42):
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to handle imbalanced data.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        sampling_strategy (str or dict): Sampling strategy
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled) or original data if imblearn not available
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("imblearn package not available. Install with: pip install imbalanced-learn")
        return X, y
    
    try:
        # Apply ADASYN
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        # Log resampling results
        original_dist = dict(Counter(y))
        new_dist = dict(Counter(y_resampled))
        logger.info(f"Original distribution: {original_dist}")
        logger.info(f"ADASYN resampled distribution: {new_dist}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"Error applying ADASYN: {str(e)}")
        return X, y

def get_sampling_function(method):
    """
    Get the appropriate sampling function based on method name.
    
    Args:
        method (str): Sampling method name
        
    Returns:
        function: Sampling function
    """
    methods = {
        'none': lambda X, y, **kwargs: (X, y),
        'oversample': manual_resample,
        'undersample': lambda X, y, **kwargs: manual_resample(X, y, strategy='undersample', **kwargs),
        'combine': lambda X, y, **kwargs: manual_resample(X, y, strategy='combine', **kwargs),
        'smote': apply_smote,
        'smote_tomek': apply_smote_tomek,
        'adasyn': apply_adasyn
    }
    
    if method in methods:
        return methods[method]
    else:
        logger.warning(f"Unknown sampling method: {method}. Using original data.")
        return methods['none']

def create_custom_sampling_strategy(y, target_ratios=None):
    """
    Create a custom sampling strategy dictionary based on target ratios.
    
    Args:
        y (numpy.ndarray): Target labels
        target_ratios (dict, optional): Dictionary of class to target ratio
        
    Returns:
        dict: Sampling strategy dictionary
    """
    if target_ratios is None:
        return 'auto'
    
    # Get original class distribution
    unique, counts = np.unique(y, return_counts=True)
    
    # Find the majority class
    majority_class = unique[np.argmax(counts)]
    majority_count = counts.max()
    
    # Create sampling strategy
    strategy = {}
    for cls in unique:
        if cls in target_ratios:
            # Convert numpy integer types to Python int to avoid serialization issues
            cls_key = int(cls) if isinstance(cls, np.integer) else cls
            
            # Calculate target count based on ratio to majority class
            target_count = int(majority_count * target_ratios[cls])
            current_count = counts[unique == cls][0]
            
            # Only include in strategy if we need to increase samples
            if current_count < target_count:
                strategy[cls_key] = target_count
    
    if not strategy:
        logger.warning("No sampling strategy created, using 'auto'")
        return 'auto'
    
    logger.info(f"Created custom sampling strategy: {strategy}")
    return strategy 