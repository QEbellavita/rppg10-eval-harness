import numpy as np
import pandas as pd
import pytest
from rppg10.score import compute_metrics, align, score_clip


def test_metrics_perfect_agreement():
    ref = np.array([60.0, 70.0, 80.0, 90.0])
    m = compute_metrics(ref.copy(), ref)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["bias"] == 0.0
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_metrics_constant_offset():
    ref = np.array([60.0, 70.0, 80.0, 90.0])
    est = ref + 5.0
    m = compute_metrics(est, ref)
    assert m["mae"] == pytest.approx(5.0)
    assert m["rmse"] == pytest.approx(5.0)
    assert m["bias"] == pytest.approx(5.0)        # est - ref
    assert m["pearson_r"] == pytest.approx(1.0)


def test_align_inner_joins_on_time_and_drops_nan():
    rppg = [(0.0, 60.0), (1.0, 61.0), (2.0, np.nan), (3.0, 63.0)]
    ecg = [(0.0, 59.0), (1.0, np.nan), (2.0, 80.0), (3.0, 62.0)]
    t, est, ref = align(rppg, ecg)
    # window 0 ok; 1 dropped (ecg nan); 2 dropped (rppg nan); 3 ok
    assert list(t) == [0.0, 3.0]
    assert list(est) == [60.0, 63.0]
    assert list(ref) == [59.0, 62.0]


def test_score_clip_reports_coverage_and_gated_metrics():
    t = np.arange(10, dtype=float)
    rppg = pd.DataFrame({
        "t_s": t,
        "hr_fft_bpm": np.r_[np.full(5, 70.0), np.full(5, 100.0)],
        "confidence": np.r_[np.full(5, 0.9), np.full(5, 0.2)],
    })
    # ECG truth is 70 everywhere -> high-confidence windows agree, low-conf ones don't
    ecg = [(float(i), 70.0) for i in range(10)]
    res = score_clip(rppg, ecg, conf_gate=0.5)
    assert res["n_windows"] == 10
    assert res["coverage"] == pytest.approx(1.0)
    assert res["mae"] == pytest.approx(15.0)          # mean(|0|*5, |30|*5)
    assert res["mae_gated"] == pytest.approx(0.0)     # only the 0.9-conf windows
    assert res["n_gated"] == 5
