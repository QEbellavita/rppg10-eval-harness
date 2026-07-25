import json
import numpy as np
import pandas as pd
from synthetic import make_ecg
from rppg10.score import run_scoring


def _setup(tmp_path, true_bpm=72.0):
    """Build a minimal out/ + dataset/ with one subject (2 ROIs) of agreeing data."""
    out = tmp_path / "out"
    (out / "hr_timeline").mkdir(parents=True)
    ds = tmp_path / "ds"
    (ds / "Subject_1").mkdir(parents=True)

    # rPPG timelines (591-row grid) that sit right on the truth
    t = np.arange(0.0, 591.0, 1.0)
    for roi in ("forehead", "cheek1"):
        pd.DataFrame({"t_s": t,
                      "hr_fft_bpm": np.full(len(t), true_bpm),
                      "confidence": np.full(len(t), 0.8)}
                     ).to_parquet(out / "hr_timeline" / f"subject1_{roi}.parquet")

    pd.DataFrame([
        {"subject_id": 1, "roi": "forehead", "status": "ok", "fitzpatrick": "II",
         "arrhythmia": False, "sex": "M", "age": 23.0},
        {"subject_id": 1, "roi": "cheek1", "status": "ok", "fitzpatrick": "II",
         "arrhythmia": False, "sex": "M", "age": 23.0},
    ]).to_csv(out / "summary.csv", index=False)

    ecg, fs, _ = make_ecg(duration_s=600.0, fs=1000.0, bpm=true_bpm, noise=0.01)
    np.save(ds / "Subject_1" / "Subject_1_ECG.npy", ecg)
    return ds, out


def test_run_scoring_writes_artifacts_and_low_mae(tmp_path):
    ds, out = _setup(tmp_path, true_bpm=72.0)
    summary = run_scoring(str(ds), str(out), conf_gate=0.5)

    assert not (out / "accuracy_by_clip.csv").exists()

    assert (out / "accuracy_by_roi.csv").exists()
    assert (out / "accuracy_by_fitzpatrick.csv").exists()

    js = json.loads((out / "accuracy_summary.json").read_text())
    assert js["overall"]["mae"] < 2.0
    assert js["overall"]["n_clips"] == 2
    assert summary["clips_scored"] == 2


def test_run_scoring_private_rows_are_explicit(tmp_path):
    ds, out = _setup(tmp_path, true_bpm=72.0)
    run_scoring(str(ds), str(out), conf_gate=0.5, write_private_rows=True)

    by_clip = pd.read_csv(out / "accuracy_by_clip.csv")
    assert len(by_clip) == 2
    assert (by_clip["mae"] < 2.0).all()


def test_run_scoring_excludes_boolean_arrhythmia(tmp_path):
    """arrhythmia is stored as boolean True/False in summary.csv; the excl-arrhythmia
    aggregate must drop True rows (regression: a string 'N' filter left it null)."""
    ds, out = _setup(tmp_path, true_bpm=72.0)
    # mark the cheek1 clip as an arrhythmia subject
    s = pd.read_csv(out / "summary.csv")
    s.loc[s["roi"] == "cheek1", "arrhythmia"] = True
    s.to_csv(out / "summary.csv", index=False)

    run_scoring(str(ds), str(out), conf_gate=0.5)
    js = json.loads((out / "accuracy_summary.json").read_text())
    assert js["overall_excl_arrhythmia"] is not None
    assert js["overall_excl_arrhythmia"]["n_clips"] == 1     # only forehead remains
    assert js["overall_gated"] is not None
    assert js["forehead"]["mae"] < 2.0
