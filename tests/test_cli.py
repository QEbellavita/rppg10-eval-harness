"""Tests for merge-hardening fixes in cli.py.

Fix 1 – Sanitize non-finite floats out of JSON outputs.
Fix 2 – Fail fast if the primary method is missing from the methods panel.
Fix 3 – Support a .zip dataset path.
"""
import json
import shutil

import cv2
import numpy as np
import pytest

from synthetic import make_pulse_frames
from rppg10 import cli
from rppg10.methods import PRIMARY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_video(path, frames, fps):
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def _make_subject_dir(root, sid, n_frames=300, bpm=72.0, roi_tag="Forehead"):
    d = root / f"Subject_{sid}"
    d.mkdir(parents=True, exist_ok=True)
    frames, fps, _ = make_pulse_frames(n_frames=n_frames, fps=30.0, bpm=bpm)
    _write_video(d / f"Subject_{sid}_{roi_tag}_.avi", frames, fps)
    return fps


# ===========================================================================
# FIX 1 – Sanitize non-finite floats (NaN / ±inf) before json.dump
# ===========================================================================

def test_sanitize_json_replaces_nonfinite_with_none():
    """_sanitize_json walks nested dicts/lists and maps non-finite floats to None."""
    from rppg10.cli import _sanitize_json

    dirty = {
        "a": float("nan"),
        "b": float("-inf"),
        "c": float("inf"),
        "d": 42.0,
        "e": None,
        "nested": {"x": float("nan"), "y": 1},
        "lst": [float("nan"), 0.0, float("inf")],
    }
    clean = _sanitize_json(dirty)

    assert clean["a"] is None
    assert clean["b"] is None
    assert clean["c"] is None
    assert clean["d"] == 42.0
    assert clean["e"] is None
    assert clean["nested"]["x"] is None
    assert clean["nested"]["y"] == 1
    assert clean["lst"][0] is None
    assert clean["lst"][1] == 0.0
    assert clean["lst"][2] is None


def test_nonfinite_floats_produce_valid_json_in_manifest_and_stub(tmp_path, monkeypatch):
    """When hrv_metrics returns NaN (< 2 IBIs), manifest.json and ingest stubs must
    still be parseable by the strict stdlib JSON parser; NaN fields must come through as None."""
    import rppg10.records as _rec_mod

    # Force hrv_metrics to always return NaN values (simulates < 2 IBI scenario).
    # Must patch on the module that imported it (records.py) not the source module.
    _NAN = float("nan")
    monkeypatch.setattr(
        _rec_mod, "hrv_metrics",
        lambda ibis_ms: {"hrv_rmssd": _NAN, "hrv_sdnn": _NAN, "hrv_pnn50": _NAN, "mean_ibi_ms": _NAN},
    )

    ds = tmp_path / "ds"
    _make_subject_dir(ds, sid=1, n_frames=300)

    out = tmp_path / "out"
    cli.run(ds, out, workers=1, xlsx=None, private_rows=True)

    # manifest.json must be parseable by the strict stdlib parser (no 'NaN' literals)
    man_text = (out / "manifest.json").read_text()
    man = json.loads(man_text)          # raises ValueError if NaN literal present

    ok_rows = [r for r in man if r["status"] == "ok"]
    assert ok_rows, "Expected at least one ok row"
    for row in ok_rows:
        for field in ("hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "mean_ibi_ms"):
            val = row[field]
            assert val is None, (
                f"field {field!r} should be None (was NaN), got {val!r}"
            )

    # ingest stubs must also be parseable
    for stub_path in (out / "ingest").glob("*.json"):
        stub = json.loads(stub_path.read_text())
        assert isinstance(stub, dict)


# ===========================================================================
# FIX 2 – Fail fast when PRIMARY is absent from the methods panel
# ===========================================================================

def test_run_raises_value_error_when_primary_method_missing(tmp_path):
    """run() raises ValueError immediately (before any processing) if PRIMARY
    (currently 'POS') is not included in the methods list."""
    ds = tmp_path / "ds"
    ds.mkdir()
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=PRIMARY):
        cli.run(ds, out, methods=["CHROM", "GREEN"], workers=1, xlsx=None)


def test_run_default_panel_does_not_raise(tmp_path):
    """Default PANEL contains PRIMARY → no ValueError; dataset with 1 subject runs cleanly."""
    ds = tmp_path / "ds"
    _make_subject_dir(ds, sid=1, n_frames=300)
    out = tmp_path / "out"
    # must not raise ValueError for missing primary (POS is in PANEL)
    summary = cli.run(ds, out, workers=1, xlsx=None)
    assert summary["processed"] >= 1
    assert (out / "run_summary.json").exists()
    assert not (out / "summary.csv").exists()
    assert not (out / "manifest.json").exists()
    assert not (out / "ingest").exists()
    assert not (out / "hr_timeline").exists()


# ===========================================================================
# FIX 3 – Support a .zip dataset path
# ===========================================================================

def test_run_accepts_zip_dataset(tmp_path):
    """run() accepts a .zip file path and extracts it, discovering subjects inside.

    Tests both the 'wrapped' layout (Dataset_rPPG-10/Subject_*) and verifies the
    xlsx auto-discovery path still works from the resolved root.
    """
    # Build a tiny 1-subject dataset under a Dataset_rPPG-10/ wrapper (real zip layout)
    raw = tmp_path / "raw"
    _make_subject_dir(raw / "Dataset_rPPG-10", sid=1, n_frames=300)

    # Create the zip archive
    zip_base = str(tmp_path / "dataset")
    shutil.make_archive(zip_base, "zip", str(raw))
    zip_path = tmp_path / "dataset.zip"
    assert zip_path.exists()

    out = tmp_path / "out"
    summary = cli.run(zip_path, out, workers=1, xlsx=None)
    assert summary["ok"] == 1, f"Expected ok=1 from zip input; got {summary}"
    assert summary["failed"] == 0
