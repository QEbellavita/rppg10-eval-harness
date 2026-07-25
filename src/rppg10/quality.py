"""Signal-quality + agreement-based confidence.

NOTE: an earlier claim that "no ground truth is available" was wrong ;  each subject
ships a synchronized 1000 Hz ECG (`Subject_N_ECG.npy`). This module's confidence is
still computed purely from signal SNR + cross-method/ROI agreement (no GT peeking), but
the confidence is now *validated* against ECG truth ;  see `rppg10.score`.
"""
import numpy as np
from .toolbox import load_post_process


def snr_db(bvp, fps, low=0.6, high=3.3) -> float:
    pp = load_post_process()
    x = np.asarray(bvp, float)
    ref_hr = pp._calculate_fft_hr(x, fs=float(fps), low_pass=low, high_pass=high)
    return float(pp._calculate_SNR(x, ref_hr, fs=float(fps), low_pass=low, high_pass=high))


def spread_bpm(hrs) -> float:
    vals = np.asarray([h for h in hrs if np.isfinite(h)], dtype=np.float64)
    return float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0


def confidence(snr, cross_method_spread, cross_roi_spread) -> float:
    s = 1.0 / (1.0 + np.exp(-float(snr) / 3.0))
    cm = np.exp(-float(cross_method_spread) / 10.0)
    cr = np.exp(-float(cross_roi_spread) / 10.0)
    return float(0.5 * s + 0.25 * cm + 0.25 * cr)
