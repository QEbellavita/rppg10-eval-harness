"""Assemble local extraction results and a non-identifying ingest example."""
import numpy as np
from .video_io import decode_rgb_trace, EmptyVideoError
from .methods import extract_bvp, PANEL, PRIMARY
from .postprocess import clean_bvp
from .hr import summary_hr, hr_timeline, detect_ibis
from .hrv import hrv_metrics
from .quality import snr_db, spread_bpm


def process_video(path, subject_id, roi, meta=None, methods=PANEL, roi_mode="full") -> dict:
    base = {
        "subject_id": subject_id, "roi": roi,
        "fitzpatrick": getattr(meta, "fitzpatrick", None),
        "arrhythmia": getattr(meta, "arrhythmia", None),
        "skin_obstruction": getattr(meta, "skin_obstruction", None),
        "age": getattr(meta, "age", None), "sex": getattr(meta, "sex", None),
    }
    _failed_shape = {"per_method": {}, "primary_hr": None, "hr_timeline": [],
                     "hrv": {}, "cross_method_spread_bpm": None}
    try:
        trace = decode_rgb_trace(path, roi_mode=roi_mode)
    except EmptyVideoError:
        return {**base, "status": "failed", "reason": "empty_video",
                "fps": None, "n_frames": 0, "duration_s": 0.0, **_failed_shape}
    except Exception as e:
        return {**base, "status": "failed", "reason": f"error: {type(e).__name__}: {e}",
                "fps": None, "n_frames": 0, "duration_s": 0.0, **_failed_shape}

    fps = trace.fps
    try:
        per_method = {}
        for name in methods:
            bvp = clean_bvp(extract_bvp(trace.rgb, fps, name), fps)
            per_method[name] = {
                "hr_bpm": summary_hr(bvp, fps),
                "snr_db": snr_db(bvp, fps),
                "bvp": bvp,
                "bvp_is_primary": (name == PRIMARY),
            }
        primary_bvp = per_method[PRIMARY]["bvp"]
        ibis = detect_ibis(primary_bvp, fps)
        return {
            **base, "status": "ok", "reason": None,
            "fps": fps, "n_frames": trace.n_frames, "duration_s": trace.n_frames / fps,
            "per_method": per_method,
            "primary_hr": per_method[PRIMARY]["hr_bpm"],
            "hr_timeline": hr_timeline(primary_bvp, fps),
            "hrv": hrv_metrics(ibis),
            "cross_method_spread_bpm": spread_bpm([m["hr_bpm"] for m in per_method.values()]),
        }
    except Exception as e:
        return {**base, "status": "failed", "reason": f"error: {type(e).__name__}: {e}",
                "fps": fps, "n_frames": trace.n_frames,
                "duration_s": trace.n_frames / fps, **_failed_shape}


def ingest_stub(result) -> dict:
    """Return a deliberately non-identifying camera-signal example.

    Subject identifiers and health or demographic attributes are excluded even when
    they exist in the local extraction record.
    """
    return {
        "source": "camera",
        "provider": "rppg-camera",
        "value": result.get("primary_hr"),
        "metadata": {
            "roi": result["roi"],
            "method": PRIMARY,
            "snr_db": (result["per_method"].get(PRIMARY, {}) or {}).get("snr_db"),
        },
    }
