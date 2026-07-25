import os
import json
import zipfile
import numpy as np
import cv2
import pytest
from concurrent.futures import ProcessPoolExecutor
from synthetic import make_pulse_frames
from rppg10 import cli

ZIP = os.environ.get(
    "RPPG10_DATASET_ZIP", os.path.expanduser("~/Downloads/Dataset_rPPG-10.zip"))


def _write_video(path, frames, fps):
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    assert writer.isOpened(), f"VideoWriter failed to open for {path} ;  codec missing?"
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def _worker_detrend_identity(_):
    """Module-level worker: load_methods() must apply the monkeypatch inside the worker."""
    from rppg10 import toolbox
    toolbox.load_methods()
    import unsupervised_methods.utils as u
    from evaluation import post_process as pp
    return (u.detrend.__name__, u.detrend.__module__,
            pp._detrend.__name__, pp._detrend.__module__)


def test_detrend_patch_propagates_to_worker():
    """Deterministic proof the O(N) detrend monkeypatch reaches a ProcessPoolExecutor
    worker (the failure mode that would resurrect the ~16h/OOM run if it regressed).

    Does NOT rely on timing: checks __name__/__module__ of the patched functions
    in the spawned worker process.  Fails immediately if the patch stops propagating.
    """
    with ProcessPoolExecutor(max_workers=1) as ex:
        u_name, u_mod, pp_name, pp_mod = ex.submit(_worker_detrend_identity, 0).result()
    assert (u_name, u_mod) == ("smoothness_detrend", "rppg10.postprocess")
    assert (pp_name, pp_mod) == ("smoothness_detrend", "rppg10.postprocess")


def test_synthetic_dataset_two_subjects(tmp_path):
    ds = tmp_path / "Dataset"
    for sid, bpm in [(1, 72.0), (2, 90.0)]:
        d = ds / f"Subject_{sid}"; d.mkdir(parents=True)
        for roi in ("Forehead", "Cheek1", "Cheek2"):
            frames, fps, _ = make_pulse_frames(n_frames=1200, fps=30.0, bpm=bpm, seed=sid)
            _write_video(d / f"Subject_{sid}_{roi}_.avi", frames, fps)
    out = tmp_path / "out"
    s = cli.run(ds, out, workers=1, xlsx=None, private_rows=True)
    assert s["ok"] == 6 and s["failed"] == 0
    man = json.loads((out / "manifest.json").read_text())
    # each subject's recovered HR is near its injected rate and ROIs agree
    for sid, bpm in [(1, 72.0), (2, 90.0)]:
        hrs = [r["hr_bpm"] for r in man if r["subject_id"] == sid]
        assert np.all(np.abs(np.array(hrs) - bpm) <= 4.0)
        assert all(r["cross_roi_spread_bpm"] <= 5.0 for r in man if r["subject_id"] == sid)


def test_long_signal_patched_detrend_in_worker(tmp_path):
    """End-to-end correctness smoke: a 6000-frame single-ROI video processed via
    workers=2 must recover HR within ±4 bpm.

    NOTE: the 60s bound below is a non-discriminating sanity timeout, NOT a patch
    guard ;  at N=6000 an unpatched O(N³) detrend only adds ~4s and would not trip
    this limit.  See test_detrend_patch_propagates_to_worker for the real
    deterministic patch-propagation guard.
    """
    import time
    ds = tmp_path / "Dataset"
    d = ds / "Subject_1"; d.mkdir(parents=True)
    frames, fps, _ = make_pulse_frames(n_frames=6000, fps=30.0, bpm=72.0, seed=1)
    _write_video(d / "Subject_1_Forehead_.avi", frames, fps)
    out = tmp_path / "out"

    t0 = time.monotonic()
    s = cli.run(ds, out, workers=2, xlsx=None, private_rows=True)
    elapsed = time.monotonic() - t0

    assert s["ok"] == 1 and s["failed"] == 0
    man = json.loads((out / "manifest.json").read_text())
    row = next(r for r in man if r["subject_id"] == 1 and r["status"] == "ok")
    assert abs(row["hr_bpm"] - 72.0) <= 4.0, (
        f"HR {row['hr_bpm']:.1f} bpm outside 72 ± 4"
    )
    assert elapsed < 60.0, f"Sanity timeout exceeded: {elapsed:.1f}s"  # non-discriminating


@pytest.mark.skipif(not os.path.exists(ZIP), reason="dataset zip not present")
def test_real_data_smoke_subject1_and_empty_subject4(tmp_path):
    root = tmp_path / "extract"
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            if ("/Subject_1/" in name or "/Subject_4/" in name
                    or name.endswith("Subject Data.xlsx")):
                z.extract(name, root)
    ds = root / "Dataset_rPPG-10"
    out = tmp_path / "out"
    s = cli.run(ds, out, workers=1, private_rows=True)
    man = json.loads((out / "manifest.json").read_text())
    s1 = [r for r in man if r["subject_id"] == 1 and r["status"] == "ok"]
    assert s1, "Subject_1 produced no ok rows"
    assert all(40.0 <= r["hr_bpm"] <= 180.0 for r in s1)
    s4 = [r for r in man if r["subject_id"] == 4]
    assert s4 and all(r["status"] == "failed" for r in s4)  # empty videos -> graceful
