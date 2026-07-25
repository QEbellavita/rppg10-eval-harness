import numpy as np
from rppg10.hrv import hrv_metrics


def test_known_ibis_exact_values():
    ibis = np.array([800.0, 850.0, 800.0, 860.0])  # ms
    m = hrv_metrics(ibis)
    diffs = np.diff(ibis)                              # [50,-50,60]
    assert np.isclose(m["hrv_rmssd"], np.sqrt(np.mean(diffs ** 2)))
    assert np.isclose(m["hrv_sdnn"], np.std(ibis, ddof=1))
    assert np.isclose(m["hrv_pnn50"], np.mean(np.abs(diffs) > 50.0))  # 1/3
    assert np.isclose(m["mean_ibi_ms"], np.mean(ibis))


def test_too_few_intervals_returns_nan():
    m = hrv_metrics(np.array([800.0]))
    assert all(np.isnan(m[k]) for k in ("hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "mean_ibi_ms"))
