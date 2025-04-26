import os
import pickle
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_pipeline_artifacts(trained_data: Dict[str, Any], output_dir: str):
    """
    Save trained models and associated scalers to disk.

    Creates 'models' and 'scalers' subdirectories within the output_dir.

    Args:
        trained_data (dict): Dictionary potentially containing 'models' and 'scalers' dicts.
                              Example: {'models': {'lr': model_obj}, 'scalers': {'lr': scaler_obj}}
        output_dir (str): Directory to save the artifacts.
    """
    if not trained_data:
        logger.warning("No trained data provided to save artifacts.")
        return

    models = trained_data.get('models', {})
    scalers = trained_data.get('scalers', {})

    if not models and not scalers:
        logger.warning("Trained data dictionary contains no models or scalers to save.")
        return

    try:
        # Ensure base output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save models
        if models:
            models_dir = os.path.join(output_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            logger.info(f"Saving models to {models_dir}...")
            for model_type, model in models.items():
                if model is not None:
                    model_file = os.path.join(models_dir, f"{model_type}.pkl")
                    try:
                        with open(model_file, 'wb') as f:
                            pickle.dump(model, f)
                        logger.info(f"  Saved {model_type} model to {model_file}")
                    except Exception as e:
                        logger.error(f"  Error saving {model_type} model to {model_file}: {e}")
                else:
                     logger.warning(f"  Skipping save for {model_type} model as it is None.")

        # Save scalers
        if scalers:
            scalers_dir = os.path.join(output_dir, "scalers")
            os.makedirs(scalers_dir, exist_ok=True)
            logger.info(f"Saving scalers to {scalers_dir}...")
            for scaler_type, scaler in scalers.items():
                 if scaler is not None:
                    scaler_file = os.path.join(scalers_dir, f"{scaler_type}_scaler.pkl")
                    try:
                        with open(scaler_file, 'wb') as f:
                            pickle.dump(scaler, f)
                        logger.info(f"  Saved {scaler_type} scaler to {scaler_file}")
                    except Exception as e:
                        logger.error(f"  Error saving {scaler_type} scaler to {scaler_file}: {e}")
                 else:
                      logger.warning(f"  Skipping save for {scaler_type} scaler as it is None.")

    except Exception as e:
        logger.exception(f"An error occurred while saving pipeline artifacts to {output_dir}: {e}")


def load_pipeline_artifacts(input_dir: str) -> Dict[str, Any]:
    """
    Load trained models and scalers from disk.

    Looks for 'models' and 'scalers' subdirectories within the input_dir.

    Args:
        input_dir (str): Directory containing saved artifacts.

    Returns:
        dict: Dictionary containing loaded 'models' and 'scalers'.
              Example: {'models': {'lr': model_obj}, 'scalers': {'lr': scaler_obj}}
              Returns empty dicts if subdirectories or files are not found.
    """
    loaded_data = {'models': {}, 'scalers': {}}
    if not input_dir or not os.path.isdir(input_dir):
        logger.error(f"Input directory for loading artifacts not found or invalid: '{input_dir}'")
        return loaded_data

    # Load models
    models_dir = os.path.join(input_dir, "models")
    if os.path.isdir(models_dir):
        logger.info(f"Loading models from {models_dir}...")
        for filename in os.listdir(models_dir):
            if filename.endswith(".pkl"):
                # Assumes filename format is {model_type}.pkl
                model_type = filename.replace(".pkl", "")
                model_path = os.path.join(models_dir, filename)
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    loaded_data['models'][model_type] = model
                    logger.info(f"  Loaded {model_type} model from {model_path}")
                except Exception as e:
                    logger.error(f"  Error loading {model_type} model from {model_path}: {e}")
    else:
        logger.warning(f"Models subdirectory not found in {input_dir}. No models loaded.")

    # Load scalers
    scalers_dir = os.path.join(input_dir, "scalers")
    if os.path.isdir(scalers_dir):
        logger.info(f"Loading scalers from {scalers_dir}...")
        for filename in os.listdir(scalers_dir):
            if filename.endswith("_scaler.pkl"):
                 # Assumes filename format is {scaler_type}_scaler.pkl
                 scaler_type = filename.replace("_scaler.pkl", "")
                 scaler_path = os.path.join(scalers_dir, filename)
                 try:
                     with open(scaler_path, 'rb') as f:
                         scaler = pickle.load(f)
                     # Ensure the loaded scaler corresponds to a loaded model if necessary?
                     # For now, just load all found scalers.
                     loaded_data['scalers'][scaler_type] = scaler
                     logger.info(f"  Loaded {scaler_type} scaler from {scaler_path}")
                 except Exception as e:
                     logger.error(f"  Error loading {scaler_type} scaler from {scaler_path}: {e}")
    else:
         logger.warning(f"Scalers subdirectory not found in {input_dir}. No scalers loaded.")

    return loaded_data 