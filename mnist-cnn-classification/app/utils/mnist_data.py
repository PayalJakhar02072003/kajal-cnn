"""Cached MNIST slices for the Streamlit app."""

from __future__ import annotations

import streamlit as st
from tensorflow import keras


@st.cache_data
def load_mnist_test():
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_test, y_test
