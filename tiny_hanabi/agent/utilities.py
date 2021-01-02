"""Utility functions.
"""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax function"""
    pref = np.exp(logits - logits.max())
    return pref / pref.sum()
