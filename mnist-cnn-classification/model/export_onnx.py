"""
Export ``saved_model/mnist_cnn.keras`` → ``saved_model/mnist_cnn.onnx`` for Streamlit Cloud.

Requires: ``pip install -r requirements-dev.txt`` (TensorFlow + tf2onnx for export.)

PowerShell:
  cd mnist-cnn-classification
  python -m model.export_onnx
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERAS = ROOT / "saved_model" / "mnist_cnn.keras"
ONNX_OUT = ROOT / "saved_model" / "mnist_cnn.onnx"


def main() -> None:
    if not KERAS.is_file():
        raise SystemExit(f"Missing {KERAS}; run: python -m model.train")
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--keras",
        str(KERAS),
        "--output",
        str(ONNX_OUT),
        "--opset",
        "13",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Wrote {ONNX_OUT}")


if __name__ == "__main__":
    main()
