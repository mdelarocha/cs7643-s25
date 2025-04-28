import logging
import numpy as np
from typing import List, Tuple, Optional

from .statistical import extract_statistical_features_batch, dict_list_to_array
from .textural import extract_textural_features_batch

logger = logging.getLogger(__name__)

def extract_features(volumes: Optional[List[np.ndarray]], feature_types: List[str]) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Extract specified features from MRI volumes and combine them.

    Args:
        volumes (list): List of MRI volumes (e.g., numpy arrays).
        feature_types (list): List of feature types to extract (e.g., ['statistical', 'textural']).

    Returns:
        tuple: (X, feature_names)
               X: Combined feature matrix (np.ndarray) or None if extraction fails.
               feature_names: List of names for the extracted features or None.
    """
    if not volumes:
        logger.error("No volumes provided for feature extraction.")
        return None, None

    if not feature_types:
        logger.warning("No feature types specified. No features will be extracted.")
        return np.array([]).reshape(len(volumes), 0), [] # Return empty array and list if no types specified

    all_features = []
    all_feature_names = []
    initial_num_samples = len(volumes)

    try:
        # --- Statistical Features ---
        if 'statistical' in feature_types:
            logger.info("Extracting statistical features...")
            statistical_features_list = extract_statistical_features_batch(volumes)
            if statistical_features_list:
                # Check if list length matches input volumes before converting
                if len(statistical_features_list) != initial_num_samples:
                     logger.warning(f"Statistical feature list length ({len(statistical_features_list)}) doesn't match input volumes ({initial_num_samples}). Skipping.")
                else:
                    X_stat, feat_names_stat = dict_list_to_array(statistical_features_list)
                    if X_stat is not None and X_stat.size > 0 and feat_names_stat:
                        if X_stat.shape[0] != initial_num_samples:
                             logger.warning(f"Statistical feature array shape ({X_stat.shape}) doesn't match input volumes ({initial_num_samples}). Skipping.")
                        else:
                            all_features.append(X_stat)
                            all_feature_names.extend(feat_names_stat)
                            logger.debug(f"Added {X_stat.shape[1]} statistical features.")
                    else:
                        logger.warning("Statistical feature extraction (dict_list_to_array) yielded no data or feature names.")
            else:
                 logger.warning("Statistical feature extraction (batch) returned None or empty list.")

        # --- Textural Features ---
        if 'textural' in feature_types:
            logger.info("Extracting textural features...")
            textural_features_list = extract_textural_features_batch(volumes)
            if textural_features_list:
                 if len(textural_features_list) != initial_num_samples:
                     logger.warning(f"Textural feature list length ({len(textural_features_list)}) doesn't match input volumes ({initial_num_samples}). Skipping.")
                 else:
                    X_text, feat_names_text = dict_list_to_array(textural_features_list)
                    if X_text is not None and X_text.size > 0 and feat_names_text:
                        if X_text.shape[0] != initial_num_samples:
                            logger.warning(f"Textural feature array shape ({X_text.shape}) doesn't match input volumes ({initial_num_samples}). Skipping.")
                        else:
                            all_features.append(X_text)
                            all_feature_names.extend(feat_names_text)
                            logger.debug(f"Added {X_text.shape[1]} textural features.")
                    else:
                        logger.warning("Textural feature extraction (dict_list_to_array) yielded no data or feature names.")
            else:
                logger.warning("Textural feature extraction (batch) returned None or empty list.")

        # --- Combine Features ---
        if all_features:
            # Double-check sample count consistency before stacking
            if not all(f.shape[0] == initial_num_samples for f in all_features):
                 logger.error("Mismatch in number of samples between successfully extracted feature types.")
                 # Log shapes for debugging
                 for i, f in enumerate(all_features):
                     # This mapping might be imperfect if features failed
                     logger.error(f"Shape for feature set {i}: {f.shape}")
                 return None, None # Critical error, cannot combine

            # Horizontally stack the feature matrices
            X_combined = np.hstack(all_features)
            logger.info(f"Successfully extracted and combined features. Final shape: {X_combined.shape}")
            return X_combined, all_feature_names
        elif not feature_types:
             # Handles case where feature_types was initially empty
             logger.warning("No feature types were specified for extraction.")
             return np.array([]).reshape(initial_num_samples, 0), []
        else:
            # Handles case where features were specified but all failed
            logger.error("No features were successfully extracted for any specified type.")
            return None, None

    except Exception as e:
        logger.exception(f"An error occurred during feature extraction: {e}")
        return None, None 