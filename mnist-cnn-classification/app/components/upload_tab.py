"""Predict from an uploaded image (resized to 28×28 grayscale)."""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from app.utils.digit_likelihood import digit_likelihood_assessment, softmax_entropy
from model.preprocess import preprocess_pil_grayscale


def render_upload_tab(model) -> None:
    st.caption(
        "Upload a digit on a plain background. Toggle invert if predictions look wrong. "
        "Non-digit images are flagged using confidence + entropy (heuristic, not perfect)."
    )
    invert = st.checkbox("Invert colors (try if digit is dark on light)", value=True)
    f = st.file_uploader("PNG or JPEG", type=["png", "jpg", "jpeg"])
    if not f:
        return
    pil = Image.open(f).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(pil, dtype=np.float32)
    st.image(pil, caption="Resized to 28×28 (grayscale)", width=280)

    x = preprocess_pil_grayscale(arr, invert=invert)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    likely_digit, explain_md = digit_likelihood_assessment(probs)
    H = softmax_entropy(probs)
    max_p = float(np.max(probs))

    if likely_digit:
        st.success("**Looks like a digit** — prediction below is meaningful for this model.")
    else:
        st.warning(
            "**This does not appear to be a digit** (for this MNIST-style detector). "
            "The bar chart is shown for curiosity only."
        )
    st.caption(explain_md)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted digit", pred if likely_digit else "—", help="Ignored when marked non-digit")
    with c2:
        st.metric("Top probability", f"{max_p:.1%}")
    with c3:
        st.metric("Entropy", f"{H:.2f}", help="Lower ≈ more peaked; very high ≈ not digit-like")

    st.bar_chart({str(i): float(probs[i]) for i in range(10)}, height=320)
