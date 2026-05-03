"""Predict using digits from the built-in MNIST test set."""

from __future__ import annotations

import numpy as np
import streamlit as st

from app.utils.mnist_data import load_mnist_test
from model.architecture import normalize_images


def render_dataset_tab(model) -> None:
    st.caption("Uses Keras `mnist.load_data()` — no upload needed.")
    if st.button("Load random test image", type="primary"):
        st.session_state["mnist_seed"] = st.session_state.get("mnist_seed", 0) + 1

    seed = st.session_state.get("mnist_seed", 0)
    rng = np.random.default_rng(seed)
    x_test, y_test = load_mnist_test()
    idx = int(rng.integers(0, len(x_test)))
    img = x_test[idx]
    label = int(y_test[idx])

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption=f"True label: {label}", width=280)
    x = normalize_images(img[np.newaxis, ...])
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    with col2:
        st.metric("Predicted digit", pred)
        st.metric("Match", "Yes" if pred == label else "No")
        st.bar_chart(
            {str(i): float(probs[i]) for i in range(10)},
            height=320,
        )
