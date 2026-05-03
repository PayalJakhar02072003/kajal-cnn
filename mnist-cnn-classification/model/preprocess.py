"""NumPy-only MNIST input prep — safe to import on Streamlit Cloud (no TensorFlow)."""

from __future__ import annotations

import numpy as np


def normalize_images(x: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] and add channel dimension if needed."""
    x = x.astype("float32") / 255.0
    if x.ndim == 3:
        x = np.expand_dims(x, -1)
    return x


def preprocess_pil_grayscale(arr: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    Convert a 2D float array in [0,255] or [0,1] to model input (1, 28, 28, 1).
    MNIST digits are light on dark; many scans are dark on light — invert helps.
    """
    if arr.max() <= 1.0:
        x = (arr * 255).astype(np.float32)
    else:
        x = arr.astype(np.float32)
    if invert:
        x = 255.0 - x
    x = x / 255.0
    x = np.expand_dims(np.expand_dims(x, 0), -1)
    return x
