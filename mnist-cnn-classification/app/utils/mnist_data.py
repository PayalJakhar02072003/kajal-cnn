"""Cached MNIST test split — no TensorFlow (works on Streamlit Cloud Python 3.14+)."""

from __future__ import annotations

import io
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import streamlit as st

# Same archive Keras uses; only stdlib + numpy required at runtime.
_MNIST_NPZ = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


@st.cache_data
def load_mnist_test():
    try:
        with urlopen(_MNIST_NPZ, timeout=120) as resp:
            raw = resp.read()
    except URLError as e:
        raise RuntimeError(f"Could not download MNIST test set: {e}") from e
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        x_test = np.asarray(z["x_test"])
        y_test = np.asarray(z["y_test"])
    return x_test, y_test
