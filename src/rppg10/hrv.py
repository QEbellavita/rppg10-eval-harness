"""Time-domain HRV metrics from inter-beat intervals (ms)."""
import numpy as np

_NAN = float("nan")


def hrv_metrics(ibis_ms) -> dict:
    ibis = np.asarray(ibis_ms, dtype=np.float64)
    if ibis.size < 2:
        return {"hrv_rmssd": _NAN, "hrv_sdnn": _NAN, "hrv_pnn50": _NAN, "mean_ibi_ms": _NAN}
    diffs = np.diff(ibis)
    return {
        "hrv_rmssd": float(np.sqrt(np.mean(diffs ** 2))),
        "hrv_sdnn": float(np.std(ibis, ddof=1)),
        "hrv_pnn50": float(np.mean(np.abs(diffs) > 50.0)),
        "mean_ibi_ms": float(np.mean(ibis)),
    }
