import numpy as np
import pytest
from rppg10.postprocess import clean_bvp, smoothness_detrend


def _power_at(sig, fps, freq, half_bw=0.1):
    spec = np.abs(np.fft.rfft(sig - np.mean(sig)))
    f = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    band = (f >= freq - half_bw) & (f <= freq + half_bw)
    return spec[band].sum()


def test_bandpass_keeps_inband_rejects_outofband():
    fps, n = 30.0, 1800
    t = np.arange(n) / fps
    inband = np.sin(2 * np.pi * 1.2 * t)      # 72 bpm, kept
    drift = 3.0 * np.sin(2 * np.pi * 0.15 * t)  # below band, rejected
    hifreq = np.sin(2 * np.pi * 6.0 * t)       # above band, rejected
    out = clean_bvp(inband + drift + hifreq, fps)
    assert _power_at(out, fps, 1.2) > 5 * _power_at(out, fps, 0.15)
    assert _power_at(out, fps, 1.2) > 5 * _power_at(out, fps, 6.0)


def test_output_same_length():
    fps, n = 30.0, 600
    sig = np.random.default_rng(0).normal(size=n)
    assert clean_bvp(sig, fps).shape[0] == n


def _dense_ref(sig_1d, n, lam):
    """Dense O(N³) reference matching the original toolbox math."""
    from scipy.sparse import spdiags as _spdiags
    e = np.ones(n)
    D = _spdiags(np.array([e, -2.0 * e, e]), np.array([0, 1, 2]), n - 2, n).toarray()
    H = np.eye(n)
    return (H - np.linalg.inv(H + lam ** 2 * D.T @ D)) @ sig_1d


@pytest.mark.parametrize("shape", [(400,), (400, 1)])
def test_smoothness_detrend_matches_toolbox(shape):
    """smoothness_detrend must be numerically equivalent to the dense O(N³) toolbox.

    Parametrised over both 1-D (clean_bvp path) and (N,1) column shape
    (POS_WANG path) to lock the shape+value contract in one assertion.
    """
    rng = np.random.default_rng(42)
    n = shape[0]
    sig = rng.standard_normal(shape)
    lam = 100.0

    ref_1d = _dense_ref(sig.ravel(), n, lam)
    ref = ref_1d.reshape(shape)

    result = smoothness_detrend(sig, lam)
    assert result.shape == shape
    assert np.max(np.abs(result - ref)) < 1e-8, (
        f"shape={shape} max abs diff {np.max(np.abs(result - ref)):.2e} exceeds 1e-8"
    )


def test_smoothness_detrend_is_fast_on_long_signal():
    """smoothness_detrend on an 18000-sample signal must return shape (18000,) quickly."""
    import time
    sig = np.random.default_rng(0).standard_normal(18000)
    t0 = time.monotonic()
    result = smoothness_detrend(sig, 100.0)
    elapsed = time.monotonic() - t0
    assert result.shape == (18000,)
    assert elapsed < 5.0, f"took {elapsed:.2f}s ;  expected < 5s"


