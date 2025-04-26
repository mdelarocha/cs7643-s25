"""
Dimensionality reduction and feature selection module.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

def pca_reduction(X, n_components=0.95):
    """
    Reduce dimensionality using PCA.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        n_components (int or float, optional): Number of components or variance to retain.
        
    Returns:
        tuple: (X_reduced, pca_model) - Reduced features and PCA model.
    """
    if X is None or X.size == 0:
        logger.error("Cannot perform PCA on empty data")
        return None, None
    
    try:
        # Initialize PCA
        pca = PCA(n_components=n_components)
        
        # Fit and transform
        X_reduced = pca.fit_transform(X)
        
        # Log information about dimensionality reduction
        if isinstance(n_components, float):
            logger.info(f"PCA: Reduced from {X.shape[1]} to {X_reduced.shape[1]} features, "
                      f"retaining {n_components:.2%} variance")
        else:
            explained_var = np.sum(pca.explained_variance_ratio_)
            logger.info(f"PCA: Reduced from {X.shape[1]} to {X_reduced.shape[1]} features, "
                      f"retaining {explained_var:.2%} variance")
        
        return X_reduced, pca
    
    except Exception as e:
        logger.error(f"Error in PCA reduction: {str(e)}")
        return X, None

def svd_reduction(X, n_components=100):
    """
    Reduce dimensionality using truncated SVD (useful for sparse matrices).
    
    Args:
        X (numpy.ndarray): Feature matrix.
        n_components (int, optional): Number of components to retain.
        
    Returns:
        tuple: (X_reduced, svd_model) - Reduced features and SVD model.
    """
    if X is None or X.size == 0:
        logger.error("Cannot perform SVD on empty data")
        return None, None
    
    try:
        # Initialize SVD
        svd = TruncatedSVD(n_components=min(n_components, X.shape[1]-1))
        
        # Fit and transform
        X_reduced = svd.fit_transform(X)
        
        # Log information
        explained_var = np.sum(svd.explained_variance_ratio_)
        logger.info(f"SVD: Reduced from {X.shape[1]} to {X_reduced.shape[1]} features, "
                  f"retaining {explained_var:.2%} variance")
        
        return X_reduced, svd
    
    except Exception as e:
        logger.error(f"Error in SVD reduction: {str(e)}")
        return X, None

def variance_threshold_selection(X, threshold=0.01):
    """
    Select features based on variance threshold.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        threshold (float, optional): Variance threshold.
        
    Returns:
        tuple: (X_selected, selector, selected_indices) - Selected features, selector model, and indices.
    """
    if X is None or X.size == 0:
        logger.error("Cannot perform variance selection on empty data")
        return None, None, None
    
    try:
        # Initialize selector
        selector = VarianceThreshold(threshold=threshold)
        
        # Fit and transform
        X_selected = selector.fit_transform(X)
        
        # Get indices of selected features
        selected_indices = np.where(selector.get_support())[0]
        
        logger.info(f"Variance threshold: Selected {X_selected.shape[1]} features from {X.shape[1]} "
                  f"(threshold = {threshold})")
        
        return X_selected, selector, selected_indices
    
    except Exception as e:
        logger.error(f"Error in variance threshold selection: {str(e)}")
        return X, None, np.arange(X.shape[1])

def univariate_feature_selection(X, y, k=10, method='f_classif'):
    """
    Select features using univariate statistical tests.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        k (int, optional): Number of features to select.
        method (str, optional): Selection method ('f_classif' or 'mutual_info_classif').
        
    Returns:
        tuple: (X_selected, selector, selected_indices) - Selected features, selector model, and indices.
    """
    if X is None or y is None or X.size == 0 or y.size == 0:
        logger.error("Cannot perform univariate selection on empty data")
        return None, None, None
    
    try:
        # Choose scoring function
        if method == 'mutual_info_classif':
            score_func = mutual_info_classif
        else:
            score_func = f_classif
        
        # Initialize selector
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=score_func, k=k)
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Get indices of selected features
        selected_indices = np.where(selector.get_support())[0]
        
        logger.info(f"Univariate selection ({method}): Selected {X_selected.shape[1]} features from {X.shape[1]}")
        
        return X_selected, selector, selected_indices
    
    except Exception as e:
        logger.error(f"Error in univariate feature selection: {str(e)}")
        return X, None, np.arange(X.shape[1])

def model_based_selection(X, y, model_type='logistic', max_features=10):
    """
    Select features using model-based selection (L1 penalty or tree-based).
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        model_type (str, optional): Model type ('logistic' or 'random_forest').
        max_features (int, optional): Maximum number of features to select.
        
    Returns:
        tuple: (X_selected, selector, selected_indices) - Selected features, selector model, and indices.
    """
    if X is None or y is None or X.size == 0 or y.size == 0:
        logger.error("Cannot perform model-based selection on empty data")
        return None, None, None
    
    try:
        # Choose model
        if model_type == 'logistic':
            model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Initialize selector
        selector = SelectFromModel(model, threshold='mean', max_features=max_features)
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Get indices of selected features
        selected_indices = np.where(selector.get_support())[0]
        
        logger.info(f"Model-based selection ({model_type}): Selected {X_selected.shape[1]} features from {X.shape[1]}")
        
        return X_selected, selector, selected_indices
    
    except Exception as e:
        logger.error(f"Error in model-based feature selection: {str(e)}")
        return X, None, np.arange(X.shape[1])

def recursive_feature_elimination(X, y, model_type='logistic', n_features=10):
    """
    Select features using recursive feature elimination.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        model_type (str, optional): Model type ('logistic' or 'random_forest').
        n_features (int, optional): Number of features to select.
        
    Returns:
        tuple: (X_selected, selector, selected_indices) - Selected features, selector model, and indices.
    """
    if X is None or y is None or X.size == 0 or y.size == 0:
        logger.error("Cannot perform RFE on empty data")
        return None, None, None
    
    try:
        # Choose model
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Initialize selector
        n_features = min(n_features, X.shape[1])
        selector = RFE(estimator=model, n_features_to_select=n_features)
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Get indices of selected features
        selected_indices = np.where(selector.get_support())[0]
        
        logger.info(f"Recursive Feature Elimination ({model_type}): Selected {X_selected.shape[1]} features from {X.shape[1]}")
        
        return X_selected, selector, selected_indices
    
    except Exception as e:
        logger.error(f"Error in recursive feature elimination: {str(e)}")
        return X, None, np.arange(X.shape[1])

def feature_selection_pipeline(X, y, feature_names=None, methods='all', n_features=10):
    """
    Run a feature selection pipeline with multiple methods and return results.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        feature_names (list, optional): List of feature names.
        methods (str or list, optional): Methods to use ('all' or list of method names).
        n_features (int, optional): Number of features to select per method.
        
    Returns:
        dict: Feature selection results.
    """
    if X is None or y is None or X.size == 0 or y.size == 0:
        logger.error("Cannot perform feature selection on empty data")
        return {}
    
    # If no feature names provided, create generic ones
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    results = {
        'X_original': X,
        'feature_names': feature_names,
        'methods': {}
    }
    
    # Available methods
    available_methods = {
        'variance': variance_threshold_selection,
        'f_classif': lambda X, y: univariate_feature_selection(X, y, k=n_features, method='f_classif'),
        'mutual_info': lambda X, y: univariate_feature_selection(X, y, k=n_features, method='mutual_info_classif'),
        'logistic_l1': lambda X, y: model_based_selection(X, y, model_type='logistic', max_features=n_features),
        'random_forest': lambda X, y: model_based_selection(X, y, model_type='random_forest', max_features=n_features),
        'rfe_logistic': lambda X, y: recursive_feature_elimination(X, y, model_type='logistic', n_features=n_features),
        'rfe_random_forest': lambda X, y: recursive_feature_elimination(X, y, model_type='random_forest', n_features=n_features)
    }
    
    # Determine which methods to use
    if methods == 'all':
        methods_to_use = list(available_methods.keys())
    else:
        methods_to_use = methods
    
    # Run each selected method
    for method in methods_to_use:
        if method in available_methods:
            logger.info(f"Running feature selection method: {method}")
            
            try:
                X_selected, selector, selected_indices = available_methods[method](X, y)
                
                # Make sure selected_indices is not None
                if selected_indices is not None:
                    # Get names of selected features
                    selected_features = [feature_names[i] for i in selected_indices]
                    
                    # Store results
                    results['methods'][method] = {
                        'X_selected': X_selected,
                        'selector': selector,
                        'selected_indices': selected_indices,
                        'selected_features': selected_features
                    }
                
            except Exception as e:
                logger.error(f"Error in feature selection method {method}: {str(e)}")
    
    return results

def get_combined_feature_importance(results, top_k=20):
    """
    Combine feature importance from different selection methods.
    
    Args:
        results (dict): Feature selection results from feature_selection_pipeline.
        top_k (int, optional): Number of top features to return.
        
    Returns:
        pandas.DataFrame: Combined feature importance.
    """
    if not results or 'methods' not in results or len(results['methods']) == 0:
        logger.error("Invalid or empty results dictionary")
        return pd.DataFrame(columns=['count'])
    
    # Get all feature names
    feature_names = results.get('feature_names', [])
    if len(feature_names) == 0:
        return pd.DataFrame(columns=['count'])
    
    # Create a dataframe to count selections across methods
    importance_df = pd.DataFrame(index=feature_names)
    importance_df['count'] = 0
    
    # Count how many times each feature was selected
    for method, method_results in results['methods'].items():
        if 'selected_features' in method_results:
            selected_features = method_results['selected_features']
            
            # Increment count for each selected feature
            for feature in selected_features:
                if feature in importance_df.index:
                    importance_df.loc[feature, 'count'] += 1
    
    # Sort by count (descending)
    importance_df = importance_df.sort_values('count', ascending=False)
    
    # Return top K features (or all if fewer than top_k)
    return importance_df.head(min(top_k, len(importance_df)))

def select_features_from_combined(X, feature_names, top_features):
    """
    Select features based on a list of top feature names.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        feature_names (list): List of all feature names.
        top_features (list): List of names of features to select.
        
    Returns:
        tuple: (X_selected, selected_indices) - Selected features and indices.
    """
    if X is None or X.size == 0:
        logger.error("Cannot select features from empty data")
        return None, None
    
    if not top_features or len(top_features) == 0:
        logger.warning("No top features provided, returning all features")
        return X, list(range(len(feature_names)))
    
    # Get indices of top features
    selected_indices = []
    for feature in top_features:
        try:
            if feature in feature_names:
                selected_indices.append(feature_names.index(feature))
        except Exception as e:
            logger.warning(f"Error finding index for feature {feature}: {str(e)}")
    
    # If no indices were found, return all features
    if not selected_indices:
        logger.warning("No matching features found, returning all features")
        return X, list(range(len(feature_names)))
    
    # Convert selected_indices to numpy array
    selected_indices = np.array(selected_indices)
    
    # Select columns from X
    X_selected = X[:, selected_indices]
    
    # Ensure X_selected is a 2D numpy array
    if len(X_selected.shape) == 1:
        X_selected = X_selected.reshape(-1, 1)
    
    logger.info(f"Selected {X_selected.shape[1]} features based on combined importance")
    
    return X_selected, selected_indices

# New combined function (adapted from baseline_pipeline.py)
def select_top_k_features_combined(X_train: np.ndarray, y_train: np.ndarray, X_test: Optional[np.ndarray],
                                  X_val: Optional[np.ndarray],
                                  feature_names: List[str], n_features: int,
                                  methods: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[str], Optional[pd.DataFrame]]:
    """
    Performs feature selection using a pipeline, combines results, and selects top k features.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        X_val: Validation features (optional).
        feature_names: List of original feature names.
        n_features: Number of top features to select.
        methods: List of selection methods to use in the pipeline (e.g., ['f_classif', 'random_forest']).

    Returns:
        Tuple containing:
            - X_train_selected: Selected training features.
            - X_test_selected: Selected test features.
            - X_val_selected: Selected validation features (or None).
            - selected_feature_names: Names of the selected features.
            - importance_df: DataFrame with combined feature importance scores.
        Returns original features if selection fails or is skipped.
    """
    if X_train is None or y_train is None or X_test is None or not feature_names:
        logger.error("Cannot select features with None inputs for train/test data or feature names.")
        return X_train, X_test, X_val, feature_names, None

    # Skip selection if n_features is invalid or not needed
    if n_features <= 0 or n_features >= X_train.shape[1]:
        logger.info(f"Skipping feature selection: n_features ({n_features}) out of range or equals total features ({X_train.shape[1]}).")
        return X_train, X_test, X_val, feature_names, None

    if methods is None:
        methods = ['f_classif', 'random_forest'] # Default methods

    X_train_selected, X_test_selected, X_val_selected = X_train, X_test, X_val
    selected_feature_names = feature_names
    final_importance_df = None

    try:
        logger.info(f"Running feature selection pipeline with methods: {methods}, target features: {n_features}")
        selection_results = feature_selection_pipeline(X_train, y_train, feature_names, methods, n_features)

        if not selection_results or 'methods' not in selection_results or not selection_results['methods']:
            logger.warning("Feature selection pipeline did not return valid results. Using all features.")
            return X_train, X_test, X_val, feature_names, None

        importance_df = get_combined_feature_importance(selection_results, top_k=n_features * 2) # Get more than k initially
        final_importance_df = importance_df # Store for return

        if importance_df is None or importance_df.empty:
            logger.warning("Could not obtain combined feature importance. Using all features.")
            return X_train, X_test, X_val, feature_names, final_importance_df

        # Select exactly n_features if possible
        top_features = importance_df.index.tolist()[:n_features]

        if not top_features:
            logger.warning("No top features identified from importance data. Using all features.")
            return X_train, X_test, X_val, feature_names, final_importance_df

        # Apply selection to train, test, and val sets
        logger.info(f"Applying selection for top {len(top_features)} features: {top_features}")
        X_train_sel, sel_indices_train = select_features_from_combined(X_train, feature_names, top_features)
        X_test_sel, sel_indices_test = select_features_from_combined(X_test, feature_names, top_features)

        # Validate selection results
        if (X_train_sel is None or X_test_sel is None or
            X_train_sel.shape[1] != len(top_features) or X_test_sel.shape[1] != len(top_features)):
            logger.warning("Feature selection failed during application or resulted in unexpected shape. Using all features.")
            return X_train, X_test, X_val, feature_names, final_importance_df

        X_train_selected = X_train_sel
        X_test_selected = X_test_sel
        selected_feature_names = top_features

        # Apply to validation set if it exists
        if X_val is not None:
            X_val_sel, _ = select_features_from_combined(X_val, feature_names, top_features)
            if X_val_sel is not None and X_val_sel.shape[1] == len(top_features):
                X_val_selected = X_val_sel
                logger.info(f"Applied feature selection to validation set. Shape: {X_val_selected.shape}")
            else:
                logger.warning("Failed to apply consistent feature selection to validation set. Setting validation set to None.")
                X_val_selected = None # Invalidate validation set if selection failed

        logger.info(f"Successfully selected {len(selected_feature_names)} features.")

    except Exception as e:
        logger.exception(f"Error during combined feature selection: {e}")
        # Return original features on error
        return X_train, X_test, X_val, feature_names, final_importance_df

    return X_train_selected, X_test_selected, X_val_selected, selected_feature_names, final_importance_df 