"""Heart-rate estimation from a (cleaned) BVP waveform."""
import numpy as np
from scipy import signal as sps
from .toolbox import load_post_process


def summary_hr(bvp, fps, low=0.6, high=3.3) -> float:
    pp = load_post_process()
    return float(pp._calculate_fft_hr(np.asarray(bvp, float), fs=float(fps),
                                      low_pass=low, high_pass=high))


def hr_timeline(bvp, fps, win_s=10.0, stride_s=1.0):
    bvp = np.asarray(bvp, float)
    win = int(round(win_s * fps))
    stride = max(1, int(round(stride_s * fps)))
    out = []
    if len(bvp) < win:
        return out
    for start in range(0, len(bvp) - win + 1, stride):
        seg = bvp[start:start + win]
        out.append((start / float(fps), summary_hr(seg, fps)))
    return out


def detect_ibis(bvp, fps, high=3.3) -> np.ndarray:
    bvp = np.asarray(bvp, float)
    min_dist = max(1, int(fps / high))   # no two beats closer than 1/high s
    peaks, _ = sps.find_peaks(bvp, distance=min_dist)
    return np.diff(peaks) / float(fps) * 1000.0
