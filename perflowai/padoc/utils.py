"""
This module contains utility functions for PADoc.
"""

import logging
from typing import Any
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy objects into JSON-serializable types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    return obj