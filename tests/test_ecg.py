import numpy as np
import pytest
from synthetic import make_ecg
from rppg10.ecg import detect_r_peaks, reference_hr_timeline, load_ecg


def test_detect_r_peaks_count_matches_rate():
    ecg, fs, bpm = make_ecg(duration_s=60.0, fs=1000.0, bpm=60.0)
    peaks = detect_r_peaks(ecg, fs)
    # ~60 beats in 60 s (first beat at t=1s) -> ~59 peaks
    assert 55 <= len(peaks) <= 62
    rr_ms = np.diff(peaks) / fs * 1000.0
    assert abs(np.median(rr_ms) - 1000.0) <= 20.0


def test_reference_hr_recovery_within_2bpm_with_wander():
    ecg, fs, bpm = make_ecg(duration_s=120.0, fs=1000.0, bpm=78.0, wander=True)
    tl = reference_hr_timeline(ecg, fs, win_s=10.0, stride_s=1.0)
    bpms = np.array([hr for _, hr in tl if not np.isnan(hr)])
    assert abs(np.median(bpms) - bpm) <= 2.0
    assert np.all((bpms > 40) & (bpms < 180))


def test_timeline_grid_matches_rppg_window_count():
    # 600 s at 10 s window / 1 s stride must yield 591 rows, identical to the
    # rPPG hr_timeline grid so the two can be aligned on t_s without resampling.
    ecg, fs, bpm = make_ecg(duration_s=600.0, fs=1000.0, bpm=72.0, noise=0.01)
    tl = reference_hr_timeline(ecg, fs, win_s=10.0, stride_s=1.0)
    assert len(tl) == 591
    assert tl[0][0] == 0.0 and tl[-1][0] == 590.0


def test_window_with_too_few_peaks_is_nan():
    flat = np.zeros(int(20 * 1000.0))   # no QRS at all
    tl = reference_hr_timeline(flat, 1000.0, win_s=10.0, stride_s=1.0)
    assert all(np.isnan(hr) for _, hr in tl)


def test_load_ecg_reads_npy(tmp_path):
    arr = np.arange(5000, dtype=np.float64)
    p = tmp_path / "Subject_99_ECG.npy"
    np.save(p, arr)
    loaded = load_ecg(p)
    assert loaded.shape == (5000,)
    assert np.array_equal(loaded, arr)
