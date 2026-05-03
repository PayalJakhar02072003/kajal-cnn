"""Load Keras model once per Streamlit session."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from tensorflow import keras


@st.cache_resource
def load_mnist_model(weights_path: str):
    path = Path(weights_path)
    if not path.is_file():
        return None
    return keras.models.load_model(path)
