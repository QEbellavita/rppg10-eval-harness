import numpy as np
from synthetic import make_pulse_frames
from rppg10.methods import extract_bvp, PRIMARY
from rppg10.postprocess import clean_bvp
from rppg10.hr import summary_hr, hr_timeline, detect_ibis


def test_golden_hr_recovery_within_2bpm():
    frames, fps, bpm = make_pulse_frames(n_frames=1800, fps=30.0, bpm=72.0)
    trace = frames.reshape(frames.shape[0], -1, 3).mean(axis=1)  # (N,3) spatial mean
    bvp = clean_bvp(extract_bvp(trace, fps, PRIMARY), fps)
    assert abs(summary_hr(bvp, fps) - bpm) <= 2.0


def test_timeline_shape_and_plausibility():
    frames, fps, bpm = make_pulse_frames(n_frames=1800, fps=30.0, bpm=66.0)
    trace = frames.reshape(frames.shape[0], -1, 3).mean(axis=1)
    bvp = clean_bvp(extract_bvp(trace, fps, PRIMARY), fps)
    tl = hr_timeline(bvp, fps, win_s=10.0, stride_s=1.0)
    assert len(tl) > 30
    bpms = np.array([b for _, b in tl])
    assert np.all((bpms > 40) & (bpms < 180))
    assert abs(np.median(bpms) - bpm) <= 4.0


def test_detect_ibis_count_matches_rate():
    fps, n, bpm = 30.0, 1800, 60.0
    t = np.arange(n) / fps
    clean = np.sin(2 * np.pi * (bpm / 60.0) * t)
    ibis = detect_ibis(clean, fps)
    # ~60 beats in 60 s -> ~59 intervals, each ~1000 ms
    assert 55 <= len(ibis) <= 63
    assert abs(np.median(ibis) - 1000.0) <= 40.0
