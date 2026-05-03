"""
MNIST CNN — Streamlit entrypoint.

PowerShell (from folder `mnist-cnn-classification`):
  python -m venv .venv
  .\\.venv\\Scripts\\Activate.ps1
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
  python -m model.train
  python -m model.export_onnx
  streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root must import `model` and `app`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.components.dataset_tab import render_dataset_tab
from app.components.upload_tab import render_upload_tab
from app.utils.model_cache import load_mnist_model

SAVED_MODEL_DIR = _ROOT / "saved_model"


def main() -> None:
    st.set_page_config(
        page_title="MNIST CNN Classifier",
        page_icon="🔢",
        layout="wide",
    )
    st.title("MNIST digit classification (CNN)")
    st.markdown(
        "Convolutional neural network on **28×28** grayscale digits "
        "(60k train / 10k test). Deploy with **Streamlit**."
    )

    model = load_mnist_model(str(SAVED_MODEL_DIR))
    if model is None:
        st.error(
            f"No model in `{SAVED_MODEL_DIR}`.\n\n"
            "For **Streamlit Cloud**, commit **`mnist_cnn.onnx`** (export after training).\n"
            "Locally: `python -m model.train` then `python -m model.export_onnx`."
        )
        st.stop()

    tab_a, tab_b = st.tabs(["MNIST test samples", "Upload your image"])
    with tab_a:
        render_dataset_tab(model)
    with tab_b:
        render_upload_tab(model)


if __name__ == "__main__":
    main()
