"""Heuristic checks for MNIST-like digit uploads (not a substitute for OOD training)."""

from __future__ import annotations

import numpy as np

# Typical MNIST-like crops: high max probability, moderate–low entropy.
# Random natural images often look more “flat” under softmax or stay weakly confident.
_MIN_TOP1_PROB = 0.48
_MAX_ENTROPY_NAT = 1.92
# If the model hesitates between classes, treat as not a clear digit.
_MIN_TOP1_TOP2_MARGIN = 0.12


def softmax_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def digit_likelihood_assessment(probs: np.ndarray) -> tuple[bool, str]:
    """
    Returns (is_likely_digit, short_explanation).

    False means: show “this does not look like a digit” (or ambiguous / low confidence).
    """
    p = np.asarray(probs, dtype=np.float64)
    max_p = float(np.max(p))
    H = softmax_entropy(p)
    order = np.argsort(p)[::-1]
    margin = float(p[order[0]] - p[order[1]]) if p.size > 1 else max_p

    if max_p < _MIN_TOP1_PROB:
        return False, (
            f"Top class confidence is only **{max_p:.0%}**. "
            "The model is not convinced this is a single clear digit."
        )
    if H > _MAX_ENTROPY_NAT:
        return False, (
            f"Class probabilities are spread out (entropy **{H:.2f}**). "
            "That usually happens with random photos, textures, or non-digit shapes."
        )
    if margin < _MIN_TOP1_TOP2_MARGIN and max_p < 0.62:
        return False, (
            "The two most likely digits are **too close**; the image may not be a single digit."
        )
    return True, (
        f"Confidence **{max_p:.0%}**, entropy **{H:.2f}** — consistent with a handwritten digit."
    )
