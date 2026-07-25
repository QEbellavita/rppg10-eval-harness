import numpy as np
from rppg10.quality import snr_db, spread_bpm, confidence


def test_clean_signal_has_higher_snr_than_noise():
    fps, n = 30.0, 1800
    t = np.arange(n) / fps
    clean = np.sin(2 * np.pi * 1.2 * t)
    noise = np.random.default_rng(0).normal(size=n)
    assert snr_db(clean, fps) > snr_db(noise, fps)


def test_spread_bpm():
    assert spread_bpm([70.0, 70.0, 70.0]) == 0.0
    assert spread_bpm([60.0, 80.0]) > 0.0
    assert spread_bpm([72.0]) == 0.0          # < 2 finite values
    assert spread_bpm([72.0, np.nan]) == 0.0  # ignores NaN


def test_confidence_bounds_and_monotonicity():
    c = confidence(snr=5.0, cross_method_spread=1.0, cross_roi_spread=1.0)
    assert 0.0 <= c <= 1.0
    # higher SNR -> higher confidence
    assert confidence(8.0, 1.0, 1.0) > confidence(-2.0, 1.0, 1.0)
    # lower spread -> higher confidence
    assert confidence(5.0, 1.0, 1.0) > confidence(5.0, 20.0, 20.0)
