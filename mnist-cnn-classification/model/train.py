"""
Train the MNIST CNN and save weights to saved_model/mnist_cnn.keras

PowerShell:
  cd mnist-cnn-classification
  python -m model.train
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensorflow import keras

from model.architecture import build_model, normalize_images


def main() -> None:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = build_model()
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=12,
        validation_split=0.1,
        verbose=1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    out_dir = ROOT / "saved_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mnist_cnn.keras"
    model.save(path)
    print(f"Saved model to {path}")
    print("For Streamlit Cloud (no TensorFlow at runtime), export ONNX:")
    print("  python -m model.export_onnx")


if __name__ == "__main__":
    main()
