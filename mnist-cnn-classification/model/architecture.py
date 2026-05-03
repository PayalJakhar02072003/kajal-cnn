"""CNN architecture and preprocessing for MNIST (28×28 grayscale)."""

from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model() -> keras.Model:
    """Small CNN suitable for MNIST."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.25),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


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
