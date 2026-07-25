"""Adapter: feed a spatial-mean RGB trace to the toolbox methods via (N,1,1,3)."""
import numpy as np
from .toolbox import load_methods

PANEL = ["POS", "CHROM", "GREEN"]
PRIMARY = "POS"

_METHODS = None


def _methods():
    global _METHODS
    if _METHODS is None:
        _METHODS = load_methods()
    return _METHODS


def extract_bvp(rgb_trace, fps, method) -> np.ndarray:
    rgb = np.asarray(rgb_trace, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb_trace must be (N,3); got {rgb.shape}")
    frames = rgb.reshape(rgb.shape[0], 1, 1, 3)   # 1x1 spatial -> method's mean is a no-op
    fn = _methods()[method]
    return np.asarray(fn(frames, float(fps))).reshape(-1)
