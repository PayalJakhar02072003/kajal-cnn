"""Load MNIST classifier: ONNX on Cloud (no TF), else Keras when training locally."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st


class _OnnxClassifier:
    def __init__(self, session: Any, input_name: str) -> None:
        self._sess = session
        self._input_name = input_name

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        (out,) = self._sess.run(None, {self._input_name: x})
        return out


@st.cache_resource
def load_mnist_model(saved_model_dir: str):
    """
    Prefer ``mnist_cnn.onnx`` (onnxruntime, any supported Python).
    Fall back to ``mnist_cnn.keras`` if TensorFlow is installed (local dev).
    """
    d = Path(saved_model_dir)
    onnx_path = d / "mnist_cnn.onnx"
    keras_path = d / "mnist_cnn.keras"

    if onnx_path.is_file():
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        input_name = sess.get_inputs()[0].name
        return _OnnxClassifier(sess, input_name)

    if keras_path.is_file():
        from tensorflow import keras

        return keras.models.load_model(keras_path)

    return None
