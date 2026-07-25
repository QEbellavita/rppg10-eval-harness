import numpy as np
import cv2
import json
from synthetic import make_pulse_frames
from rppg10.records import process_video, ingest_stub
from rppg10 import cli


def _write_video(path, frames, fps):
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def test_process_video_ok(tmp_path):
    frames, fps, bpm = make_pulse_frames(n_frames=1800, fps=30.0, bpm=72.0)
    p = tmp_path / "Subject_1_Forehead_.avi"
    _write_video(p, frames, fps)
    res = process_video(p, subject_id=1, roi="forehead")
    assert res["status"] == "ok"
    assert abs(res["primary_hr"] - bpm) <= 3.0
    assert "POS" in res["per_method"]
    assert res["per_method"]["POS"]["bvp"].ndim == 1
    assert len(res["hr_timeline"]) > 30


def test_process_video_empty(tmp_path):
    p = tmp_path / "Subject_4_Forehead_.avi"
    p.write_bytes(b"")
    res = process_video(p, subject_id=4, roi="forehead")
    assert res["status"] == "failed"
    assert res["reason"] == "empty_video"


def test_ingest_stub_shape(tmp_path):
    frames, fps, bpm = make_pulse_frames(n_frames=900, fps=30.0, bpm=72.0)
    p = tmp_path / "Subject_1_Cheek1_.avi"
    _write_video(p, frames, fps)
    res = process_video(p, subject_id=1, roi="cheek1")
    stub = ingest_stub(res)
    assert stub["source"] == "camera"
    assert stub["provider"] == "rppg-camera"
    assert stub["metadata"]["roi"] == "cheek1"
    assert stub["value"] == res["primary_hr"]
    assert "subject_id" not in stub["metadata"]
    assert "fitzpatrick" not in stub["metadata"]
    assert "arrhythmia" not in stub["metadata"]


def test_run_end_to_end_and_resume(tmp_path):
    ds = tmp_path / "ds"
    for sid, roi in [(1, "Forehead"), (1, "Cheek1"), (1, "Cheek2")]:
        d = ds / f"Subject_{sid}"
        d.mkdir(parents=True, exist_ok=True)
        frames, fps, _ = make_pulse_frames(n_frames=900, fps=30.0, bpm=72.0)
        _write_video(d / f"Subject_{sid}_{roi}_.avi", frames, fps)
    # add the empty Subject_4 case
    d4 = ds / "Subject_4"; d4.mkdir()
    (d4 / "Subject_4_Forehead_.avi").write_bytes(b"")

    out = tmp_path / "out"
    summary = cli.run(ds, out, workers=1, xlsx=None, private_rows=True)
    assert summary["processed"] == 4 and summary["ok"] == 3 and summary["failed"] == 1
    assert (out / "summary.csv").exists()
    assert (out / "manifest.json").exists()
    stubs = list((out / "ingest").glob("*.json"))
    assert len(stubs) == 3
    # confidence filled at subject level for the 3 ok ROIs
    man = json.loads((out / "manifest.json").read_text())
    ok_rows = [r for r in man if r["status"] == "ok"]
    assert all("confidence" in r and 0 <= r["confidence"] <= 1 for r in ok_rows)
    # snr_by_fitzpatrick artifact must exist
    assert (out / "snr_by_fitzpatrick.csv").exists()
    # resume: second run reprocesses nothing
    summary2 = cli.run(ds, out, workers=1, xlsx=None, private_rows=True)
    assert summary2["skipped"] == 3


def test_run_parallel_workers_pickles(tmp_path):
    """workers>1 uses ProcessPoolExecutor; the worker must be picklable (no closures)."""
    ds = tmp_path / "ds"
    for sid in (1, 2):
        d = ds / f"Subject_{sid}"; d.mkdir(parents=True)
        frames, fps, _ = make_pulse_frames(n_frames=300, fps=30.0, bpm=72.0, seed=sid)
        _write_video(d / f"Subject_{sid}_Forehead_.avi", frames, fps)
    out = tmp_path / "out"
    summary = cli.run(ds, out, workers=2, xlsx=None)
    assert summary["ok"] == 2 and summary["failed"] == 0


def test_process_video_catches_unexpected_error(tmp_path, monkeypatch):
    """Any non-EmptyVideoError exception inside extraction returns a failed record (no raise)."""
    frames, fps, _ = make_pulse_frames(n_frames=900, fps=30.0, bpm=72.0)
    p = tmp_path / "Subject_1_Forehead_.avi"
    _write_video(p, frames, fps)

    import rppg10.records as rec
    monkeypatch.setattr(rec, "extract_bvp", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    res = process_video(p, subject_id=1, roi="forehead")
    assert res["status"] == "failed"
    assert "boom" in res["reason"] or "RuntimeError" in res["reason"]
    assert res["per_method"] == {}
    assert res["primary_hr"] is None


def test_partial_resume_cross_roi_spread(tmp_path):
    """cross_roi_spread on resumed run includes prior-run ROIs, not just current batch."""
    ds = tmp_path / "ds"
    d1 = ds / "Subject_1"; d1.mkdir(parents=True)

    # Run 1: only Forehead at bpm=60 (well separated from cheeks to make spreads distinct)
    frames_f, fps, _ = make_pulse_frames(n_frames=900, fps=30.0, bpm=60.0, seed=10)
    _write_video(d1 / "Subject_1_Forehead_.avi", frames_f, fps)

    out = tmp_path / "out"
    cli.run(ds, out, workers=1, xlsx=None, private_rows=True)

    # Add Cheek1 (bpm=90) and Cheek2 (bpm=120) then run 2 (resume)
    # std([90,120],ddof=1)≈21.2; std([60,90,120],ddof=1)=30.0 → diff≈8.8 > 3bpm tolerance
    frames_c1, fps, _ = make_pulse_frames(n_frames=900, fps=30.0, bpm=90.0, seed=11)
    frames_c2, fps, _ = make_pulse_frames(n_frames=900, fps=30.0, bpm=120.0, seed=12)
    _write_video(d1 / "Subject_1_Cheek1_.avi", frames_c1, fps)
    _write_video(d1 / "Subject_1_Cheek2_.avi", frames_c2, fps)

    cli.run(ds, out, workers=1, xlsx=None, private_rows=True)

    # Load manifest and check cheek rows' cross_roi_spread reflects all 3 ROIs
    man = json.loads((out / "manifest.json").read_text())
    cheek_rows = [r for r in man if r["roi"] in ("cheek1", "cheek2") and r["status"] == "ok"]
    assert len(cheek_rows) == 2

    # Expected: std([recovered_forehead, recovered_cheek1, recovered_cheek2], ddof=1)
    cheek_hrs = [r["hr_bpm"] for r in cheek_rows]
    forehead_row = next(r for r in man if r["roi"] == "forehead" and r["status"] == "ok")
    expected_spread = np.std([forehead_row["hr_bpm"]] + cheek_hrs, ddof=1)
    for r in cheek_rows:
        assert abs(r["cross_roi_spread_bpm"] - expected_spread) < 3.0
