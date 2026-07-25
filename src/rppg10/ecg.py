"""ECG ground-truth heart rate for Dataset_rPPG-10.

Each subject ships a synchronized single-lead ECG as ``Subject_N_ECG.npy`` sampled
at 1000 Hz over the same 600 s as the videos. We detect R-peaks (Pan-Tompkins-style
band-pass -> derivative -> square -> moving-window integration) and build a reference
HR timeline on the *same* 10 s-window / 1 s-stride grid as the rPPG ``hr_timeline``,
so the two align on ``t_s`` with no resampling.
"""
import numpy as np
from scipy import signal as sps

ECG_FS = 1000.0  # Hz ;  verified: 600,000 samples / 600 s for every subject


def load_ecg(path) -> np.ndarray:
    """Load a Subject_N_ECG.npy as a 1-D float array."""
    return np.load(path).astype(float).ravel()


def detect_r_peaks(ecg, fs: float = ECG_FS) -> np.ndarray:
    """Return R-peak sample indices via a Pan-Tompkins-style pipeline."""
    ecg = np.asarray(ecg, float)
    if ecg.size < int(fs):
        return np.empty(0, dtype=int)
    sos = sps.butter(2, [5.0, 15.0], btype="band", fs=fs, output="sos")
    filt = sps.sosfiltfilt(sos, ecg)
    squared = np.ediff1d(filt, to_begin=0.0) ** 2
    win = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    # Refractory: no two beats closer than 1/ (180 bpm) = 0.33 s.
    min_dist = max(1, int(0.33 * fs))
    height = np.percentile(integrated, 75)
    peaks, _ = sps.find_peaks(integrated, height=height, distance=min_dist)
    return peaks


def reference_hr_timeline(ecg, fs: float = ECG_FS, win_s: float = 10.0,
                          stride_s: float = 1.0):
    """List of (t_s, hr_bpm) over sliding windows; NaN where <2 R-peaks in a window.

    Per-window HR uses the median RR interval of the peaks inside the window, which
    is robust to the occasional missed/extra detection.
    """
    ecg = np.asarray(ecg, float)
    peaks_t = detect_r_peaks(ecg, fs) / float(fs)
    total_s = len(ecg) / float(fs)
    out = []
    start = 0.0
    while start + win_s <= total_s + 1e-6:
        in_win = peaks_t[(peaks_t >= start) & (peaks_t < start + win_s)]
        if in_win.size >= 2:
            hr = 60.0 / float(np.median(np.diff(in_win)))
        else:
            hr = float("nan")
        out.append((round(start, 6), hr))
        start += stride_s
    return out
