import numpy as np
import logging

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """
    Recursively convert numpy data types within an object (dict, list, scalar) 
    to their standard Python equivalents for JSON serialization.

    Handles np.integer, np.floating (including NaN/Inf), np.ndarray, np.bool_,
    np.string_, np.unicode_.

    Args:
        obj: The object (potentially nested) containing numpy types.

    Returns:
        The object with numpy types converted to Python equivalents.
        Returns obj directly if no conversion is needed or applicable.
        NaN is converted to None.
        Infinity is converted to string ('inf' or '-inf').
    
    Raises:
        TypeError: If an unhandled type is encountered (delegated to json default handler).
    """
    # Check for basic Python types first for efficiency
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    # Numpy specific types
    if isinstance(obj, np.integer): 
        return int(obj)
    elif isinstance(obj, np.floating): 
        if np.isnan(obj):
            return None # JSON null for NaN
        elif np.isinf(obj):
            # Represent infinity as string 'inf' or '-inf' compatible with JSON
            return str(obj) 
        else:
            return float(obj)
    elif isinstance(obj, np.ndarray): 
        # Recursively convert elements within the array
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, np.bool_): 
        return bool(obj)
    elif isinstance(obj, (np.bytes_, np.str_)):
        return str(obj)
    
    # Recursive handling for Python containers
    elif isinstance(obj, dict): 
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): 
        return [convert_numpy_types(i) for i in obj]
    
    # If type is not handled, return the object itself. 
    # Let the caller (e.g., json.dump) handle potential TypeErrors for unsupported types.
    # Using logger.debug can be helpful during development if needed:
    # logger.debug(f"Type {type(obj)} not explicitly handled by convert_numpy_types, returning object as is.")
    return obj 