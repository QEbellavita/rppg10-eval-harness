"""CLI runner: discover videos, process (parallel), write artifacts. No DB/API writes."""
import math
import os
import re
import json
import tempfile
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from .records import process_video, ingest_stub
from .methods import PANEL, PRIMARY
from .quality import spread_bpm, confidence
from .metadata import load_metadata
from .toolbox import toolbox_commit

ROI_FILES = {"forehead": "Forehead", "cheek1": "Cheek1", "cheek2": "Cheek2"}


def discover(dataset):
    """Yield (subject_id, roi, path) for every expected ROI file that exists."""
    entries = sorted(os.listdir(dataset), key=lambda e: int(m.group(1)) if (m := re.match(r"Subject_(\d+)$", e)) else float("inf"))
    for entry in entries:
        m = re.match(r"Subject_(\d+)$", entry)
        if not m:
            continue
        sid = int(m.group(1))
        for roi, tag in ROI_FILES.items():
            p = os.path.join(dataset, entry, f"Subject_{sid}_{tag}_.avi")
            if os.path.exists(p):
                yield sid, roi, p


def _stub_path(out_dir, sid, roi):
    return os.path.join(out_dir, "ingest", f"subject{sid}_{roi}.json")


def _write_artifacts(out_dir, res):
    sid, roi = res["subject_id"], res["roi"]
    if res["status"] == "ok":
        wdir = os.path.join(out_dir, "waveforms"); os.makedirs(wdir, exist_ok=True)
        for name, m in res["per_method"].items():
            bvp = m["bvp"]
            t = np.arange(len(bvp)) / res["fps"]
            pd.DataFrame({"t_s": t, "bvp": bvp}).to_parquet(
                os.path.join(wdir, f"subject{sid}_{roi}_{name}.parquet"))
        tdir = os.path.join(out_dir, "hr_timeline"); os.makedirs(tdir, exist_ok=True)
        tl = res["hr_timeline"]
        pd.DataFrame({"t_s": [t for t, _ in tl],
                      "hr_fft_bpm": [b for _, b in tl],
                      "confidence": [res.get("confidence")] * len(tl)}).to_parquet(
            os.path.join(tdir, f"subject{sid}_{roi}.parquet"))
        qdir = os.path.join(out_dir, "ingest"); os.makedirs(qdir, exist_ok=True)
        with open(_stub_path(out_dir, sid, roi), "w") as f:
            json.dump(_sanitize_json(ingest_stub(res)), f, indent=2,
                      allow_nan=False, default=_json_safe)


def _json_safe(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _sanitize_json(obj):
    """Recursively replace non-finite floats (NaN / ±inf) with None.

    Covers both Python ``float`` and numpy floating-point scalars so that
    ``json.dump(..., allow_nan=False)`` never raises after sanitization.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not math.isfinite(float(obj)):
        return None
    return obj


def _manifest_row(res):
    pm = res.get("per_method", {})
    primary = pm.get(PRIMARY, {})
    return {
        "subject_id": res["subject_id"], "roi": res["roi"], "status": res["status"],
        "reason": res.get("reason"), "fps": res.get("fps"),
        "n_frames": res.get("n_frames"), "duration_s": res.get("duration_s"),
        "hr_bpm": res.get("primary_hr"),
        "snr_db": primary.get("snr_db"),
        "hrv_rmssd": res.get("hrv", {}).get("hrv_rmssd"),
        "hrv_sdnn": res.get("hrv", {}).get("hrv_sdnn"),
        "hrv_pnn50": res.get("hrv", {}).get("hrv_pnn50"),
        "mean_ibi_ms": res.get("hrv", {}).get("mean_ibi_ms"),
        "cross_method_spread_bpm": res.get("cross_method_spread_bpm"),
        "cross_roi_spread_bpm": res.get("cross_roi_spread_bpm"),
        "confidence": res.get("confidence"),
        "fitzpatrick": res.get("fitzpatrick"), "arrhythmia": res.get("arrhythmia"),
        "skin_obstruction": res.get("skin_obstruction"),
        "age": res.get("age"), "sex": res.get("sex"),
    }


def _worker(task):
    """Module-level (picklable) worker for ProcessPoolExecutor."""
    sid, roi, path, subj_meta, methods, roi_mode = task
    return process_video(path, sid, roi, meta=subj_meta, methods=methods, roi_mode=roi_mode)


def run(dataset, out_dir, methods=PANEL, workers=4, resume=True, xlsx=None,
        roi_mode="full", private_rows=False) -> dict:
    dataset, out_dir = str(dataset), str(out_dir)

    # FIX 2: fail fast when the primary method is absent from the panel
    if PRIMARY not in methods:
        raise ValueError(
            f"primary method {PRIMARY!r} must be included in methods; got {methods}"
        )

    # FIX 3: transparent .zip support ;  extract to a temp dir and resolve the dataset root
    if dataset.endswith(".zip"):
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(dataset) as zf:
            zf.extractall(tmpdir)
        # Handle the common "wrapped" layout: Dataset_rPPG-10/Subject_* inside the zip.
        # If the temp dir contains exactly one sub-directory and no Subject_* dirs directly,
        # descend into that sub-directory.
        top_entries = os.listdir(tmpdir)
        has_subjects = any(re.match(r"Subject_\d+$", e) for e in top_entries)
        subdirs = [e for e in top_entries if os.path.isdir(os.path.join(tmpdir, e))]
        if not has_subjects and len(subdirs) == 1:
            tmpdir = os.path.join(tmpdir, subdirs[0])
        dataset = tmpdir

    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    if xlsx is None:
        cand = os.path.join(dataset, "Subject Data.xlsx")
        xlsx = cand if os.path.exists(cand) else None
    if xlsx:
        try:
            meta = load_metadata(xlsx)
        except Exception:
            meta = {}

    man_path = os.path.join(out_dir, "manifest.json")
    if not private_rows:
        resume = False

    tasks = list(discover(dataset))
    todo, skipped = [], 0
    for sid, roi, path in tasks:
        if resume and os.path.exists(_stub_path(out_dir, sid, roi)):
            skipped += 1
        else:
            todo.append((sid, roi, path))

    # build picklable arg tuples (closures can't cross the process boundary)
    work = [(sid, roi, path, meta.get(sid), methods, roi_mode) for (sid, roi, path) in todo]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_worker, work))
    else:
        results = [_worker(w) for w in work]

    # load prior-run OK HRs so cross-ROI spread is correct on partial resume
    current_keys = {(r["subject_id"], r["roi"]) for r in results}
    prior_hr: dict[int, list] = {}
    if resume and os.path.exists(man_path):
        with open(man_path) as fh:
            for row in json.load(fh):
                if row["status"] == "ok" and (row["subject_id"], row["roi"]) not in current_keys:
                    prior_hr.setdefault(row["subject_id"], []).append(row["hr_bpm"])

    # fill cross-ROI spread + confidence per subject (ok results + prior-run ok HRs)
    by_subject = {}
    for r in results:
        by_subject.setdefault(r["subject_id"], []).append(r)
    for sid, rs in by_subject.items():
        ok = [r for r in rs if r["status"] == "ok"]
        cr = spread_bpm([r["primary_hr"] for r in ok] + prior_hr.get(sid, []))
        for r in ok:
            r["cross_roi_spread_bpm"] = cr
            primary_snr = r["per_method"][PRIMARY]["snr_db"]
            r["confidence"] = confidence(primary_snr, r["cross_method_spread_bpm"], cr)

    if private_rows:
        for r in results:
            _write_artifacts(out_dir, r)

    # merge with any pre-existing manifest rows (resume) before writing
    rows = [_manifest_row(r) for r in results]
    if private_rows:
        if resume and os.path.exists(man_path):
            with open(man_path) as fh:
                prev = {(x["subject_id"], x["roi"]): x for x in json.load(fh)}
            for row in rows:
                prev[(row["subject_id"], row["roi"])] = row
            rows = list(prev.values())
        rows.sort(key=lambda x: (x["subject_id"], x["roi"]))
        with open(man_path, "w") as f:
            json.dump(_sanitize_json(rows), f, indent=2, allow_nan=False, default=_json_safe)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

        # This aggregate is safe to publish, but requires private local rows to compute.
        df = pd.DataFrame(rows)
        ok_df = df[df["status"] == "ok"] if not df.empty else pd.DataFrame()
        if not ok_df.empty:
            (ok_df.groupby("fitzpatrick")["snr_db"]
                  .agg(["count", "mean", "median", "std"])
                  .to_csv(os.path.join(out_dir, "snr_by_fitzpatrick.csv")))

    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump({"toolbox_commit": toolbox_commit(), "methods": list(methods),
                   "primary": PRIMARY, "band_hz": [0.6, 3.3],
                   "hr_window_s": 10.0, "hr_stride_s": 1.0,
                   "roi_mode": roi_mode,
                   "dataset": os.path.basename(os.path.normpath(dataset)),
                   "private_row_output": bool(private_rows)}, f, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "failed")
    public_summary = {"processed": len(results), "ok": ok,
                      "failed": failed, "skipped": skipped}
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(public_summary, f, indent=2)
    return public_summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="rPPG-10 pulse & HR extractor (Spec A)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default=",".join(PANEL))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--xlsx", default=None)
    ap.add_argument("--roi-mode", choices=["full", "detect"], default="full",
                    help="'full' = whole-frame mean (pre-cropped ROI inputs, e.g. rPPG-10); "
                         "'detect' = Haar face-detect + skin crop (full-frame webcam inputs)")
    ap.add_argument(
        "--private-row-output", action="store_true",
        help="write subject-linked local artifacts; never publish this output directory")
    a = ap.parse_args(argv)
    summary = run(a.dataset, a.out, methods=a.methods.split(","),
                  workers=a.workers, resume=not a.no_resume, xlsx=a.xlsx,
                  roi_mode=a.roi_mode, private_rows=a.private_row_output)
    print(json.dumps(summary, indent=2))


def main_score(argv=None):
    """Score a finished extractor run against the dataset's ECG ground truth."""
    from .score import run_scoring
    ap = argparse.ArgumentParser(
        description="rPPG-10 accuracy scoring vs ECG ground truth")
    ap.add_argument("--dataset", required=True, help="dir with Subject_N/Subject_N_ECG.npy")
    ap.add_argument("--out", required=True, help="extractor out/ dir (read + write)")
    ap.add_argument("--conf-gate", type=float, default=0.5)
    ap.add_argument(
        "--private-row-output", action="store_true",
        help="also write accuracy_by_clip.csv; never publish that file")
    a = ap.parse_args(argv)
    summary = run_scoring(a.dataset, a.out, conf_gate=a.conf_gate,
                          write_private_rows=a.private_row_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
