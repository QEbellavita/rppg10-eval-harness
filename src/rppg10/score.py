"""Score extracted rPPG HR against the ECG ground-truth timeline.

The rPPG ``hr_timeline`` (t_s, hr_fft_bpm, confidence) and the ECG
``reference_hr_timeline`` (t_s, hr) share the same 10 s/1 s grid, so scoring is an
inner join on ``t_s`` with NaN windows dropped on either side. We report the standard
HR-accuracy metrics (MAE, RMSE, bias, Pearson r), the coverage (fraction of windows
where both signals were valid), and a confidence-gated variant ;  the honest read is
"how good is the HR when the engine says it has signal", reported alongside the raw.
"""
import os
import json
import math
import numpy as np
import pandas as pd

from .ecg import load_ecg, reference_hr_timeline, ECG_FS


def compute_metrics(est, ref) -> dict:
    """MAE/RMSE/bias(est-ref)/Pearson r over paired arrays (already NaN-free)."""
    est = np.asarray(est, float)
    ref = np.asarray(ref, float)
    n = int(est.size)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "pearson_r": float("nan"), "n": 0}
    err = est - ref
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if n >= 2 and np.std(est) > 0 and np.std(ref) > 0:
        pearson_r = float(np.corrcoef(est, ref)[0, 1])
    else:
        pearson_r = float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "pearson_r": pearson_r, "n": n}


def align(rppg_tl, ecg_tl):
    """Inner-join two (t_s, hr) timelines on t_s, dropping windows that are NaN on
    either side. Returns (t, est, ref) float arrays."""
    ecg_map = {round(float(t), 6): float(hr) for t, hr in ecg_tl}
    t_out, est_out, ref_out = [], [], []
    for t, hr in rppg_tl:
        key = round(float(t), 6)
        if key not in ecg_map:
            continue
        ref = ecg_map[key]
        est = float(hr)
        if np.isnan(est) or np.isnan(ref):
            continue
        t_out.append(key)
        est_out.append(est)
        ref_out.append(ref)
    return np.array(t_out), np.array(est_out), np.array(ref_out)


def _rppg_pairs(rppg_df):
    return list(zip(rppg_df["t_s"].to_numpy(float), rppg_df["hr_fft_bpm"].to_numpy(float)))


def score_clip(rppg_df, ecg_tl, conf_gate=None) -> dict:
    """Score one subject×ROI clip.

    rppg_df: DataFrame with t_s, hr_fft_bpm, confidence.
    ecg_tl:  list of (t_s, hr) reference windows.
    Returns metrics over all valid windows plus a confidence-gated variant.
    """
    n_windows = int(len(rppg_df))
    t, est, ref = align(_rppg_pairs(rppg_df), ecg_tl)
    res = compute_metrics(est, ref)
    res["n_windows"] = n_windows
    res["n_valid"] = int(est.size)
    res["coverage"] = float(est.size / n_windows) if n_windows else float("nan")

    if conf_gate is not None and "confidence" in rppg_df:
        conf_map = {round(float(tt), 6): float(c)
                    for tt, c in zip(rppg_df["t_s"], rppg_df["confidence"])}
        keep = np.array([conf_map.get(round(float(tt), 6), 0.0) >= conf_gate for tt in t])
        gm = compute_metrics(est[keep], ref[keep])
        res["mae_gated"] = gm["mae"]
        res["rmse_gated"] = gm["rmse"]
        res["bias_gated"] = gm["bias"]
        res["n_gated"] = gm["n"]
    return res


# --------------------------------------------------------------------------- #
# Orchestration: score a finished extractor run against the dataset's ECG GT.  #
# --------------------------------------------------------------------------- #

def _pooled(rows, est_key="_est", ref_key="_ref"):
    """Pool window-level (est, ref) pairs across clips and compute one metric set."""
    est = np.concatenate([r[est_key] for r in rows]) if rows else np.array([])
    ref = np.concatenate([r[ref_key] for r in rows]) if rows else np.array([])
    m = compute_metrics(est, ref)
    m["n_clips"] = sum(1 for r in rows if r[est_key].size > 0)
    return m


def _gate_mask(rppg_df, t_aligned, conf_gate):
    """Boolean mask over the aligned windows where rPPG confidence >= conf_gate."""
    conf_map = {round(float(tt), 6): float(c)
                for tt, c in zip(rppg_df["t_s"], rppg_df.get("confidence", []))}
    return np.array([conf_map.get(round(float(tt), 6), 0.0) >= conf_gate
                     for tt in t_aligned], dtype=bool)


def _nan_to_none(d):
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in d.items()}


def run_scoring(dataset, out_dir, conf_gate=0.5, write_private_rows=False) -> dict:
    """Score every OK clip in ``out_dir`` against the per-subject ECG ground truth.

    Reads: out/summary.csv (metadata + status), out/hr_timeline/subject{N}_{roi}.parquet,
           dataset/Subject_{N}/Subject_{N}_ECG.npy.
    Writes aggregate ROI, skin-tone, and summary artifacts. The subject-linked
    ``accuracy_by_clip.csv`` is written only when ``write_private_rows=True``.
    """
    out_dir, dataset = str(out_dir), str(dataset)
    summary = pd.read_csv(os.path.join(out_dir, "summary.csv"))
    ok = summary[summary["status"] == "ok"].copy()

    ecg_cache = {}

    def ecg_tl_for(sid):
        if sid not in ecg_cache:
            p = os.path.join(dataset, f"Subject_{sid}", f"Subject_{sid}_ECG.npy")
            if not os.path.exists(p):
                ecg_cache[sid] = None
            else:
                ecg_cache[sid] = reference_hr_timeline(load_ecg(p), ECG_FS)
        return ecg_cache[sid]

    rows = []
    for _, r in ok.iterrows():
        sid, roi = int(r["subject_id"]), r["roi"]
        tl_path = os.path.join(out_dir, "hr_timeline", f"subject{sid}_{roi}.parquet")
        ecg_tl = ecg_tl_for(sid)
        if not os.path.exists(tl_path) or ecg_tl is None:
            continue
        rppg_df = pd.read_parquet(tl_path)
        sc = score_clip(rppg_df, ecg_tl, conf_gate=conf_gate)
        t, est, ref = align(_rppg_pairs(rppg_df), ecg_tl)
        gmask = _gate_mask(rppg_df, t, conf_gate)
        row = {
            "subject_id": sid, "roi": roi,
            "fitzpatrick": r.get("fitzpatrick"), "arrhythmia": r.get("arrhythmia"),
            "sex": r.get("sex"), "age": r.get("age"),
            "mae": sc["mae"], "rmse": sc["rmse"], "bias": sc["bias"],
            "pearson_r": sc["pearson_r"], "n_valid": sc["n_valid"],
            "coverage": sc["coverage"],
            "mae_gated": sc.get("mae_gated"), "n_gated": sc.get("n_gated"),
            "_est": est, "_ref": ref,
            "_est_g": est[gmask], "_ref_g": ref[gmask],
        }
        rows.append(row)

    # Build the local row table for aggregation. Do not persist it by default.
    clip_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                            for r in rows])
    clip_df.sort_values(["subject_id", "roi"], inplace=True)
    if write_private_rows:
        clip_df.to_csv(os.path.join(out_dir, "accuracy_by_clip.csv"), index=False)

    # by-ROI (pooled across subjects)
    roi_recs = []
    for roi in sorted(clip_df["roi"].unique()) if not clip_df.empty else []:
        grp = [r for r in rows if r["roi"] == roi]
        m = _pooled(grp)
        roi_recs.append({"roi": roi, **{k: m[k] for k in
                        ("mae", "rmse", "bias", "pearson_r", "n", "n_clips")}})
    pd.DataFrame(roi_recs).to_csv(os.path.join(out_dir, "accuracy_by_roi.csv"), index=False)

    # by-Fitzpatrick (pooled)
    fitz_recs = []
    fitz_vals = sorted(clip_df["fitzpatrick"].dropna().unique()) if not clip_df.empty else []
    for fz in fitz_vals:
        grp = [r for r in rows if r["fitzpatrick"] == fz]
        m = _pooled(grp)
        fitz_recs.append({"fitzpatrick": fz, **{k: m[k] for k in
                         ("mae", "rmse", "bias", "pearson_r", "n", "n_clips")}})
    pd.DataFrame(fitz_recs).to_csv(os.path.join(out_dir, "accuracy_by_fitzpatrick.csv"),
                                   index=False)

    # summary JSON: overall pooled, confidence-gated, forehead-only headline,
    # best ROI, arrhythmia-excluded variant. arrhythmia is a boolean in summary.csv.
    overall = _pooled(rows)
    overall_gated = _pooled(rows, est_key="_est_g", ref_key="_ref_g")
    non_arr = [r for r in rows if not bool(r["arrhythmia"])]
    forehead_rows = [r for r in rows if r["roi"] == "forehead"]
    forehead = _pooled(forehead_rows)
    forehead_gated = _pooled(forehead_rows, est_key="_est_g", ref_key="_ref_g")
    best_roi = min(roi_recs, key=lambda x: x["mae"])["roi"] if roi_recs else None
    js = {
        "dataset": "Dataset_rPPG-10",
        "ground_truth": "ECG @ 1000 Hz, R-peak median-RR, 10s/1s windows",
        "rppg_method": "rPPG-Toolbox POS (primary), FFT HR, band 0.6-3.3 Hz",
        "conf_gate": conf_gate,
        "overall": _nan_to_none(overall),
        "overall_gated": _nan_to_none(overall_gated),
        "overall_excl_arrhythmia": _nan_to_none(_pooled(non_arr)) if non_arr else None,
        "forehead": _nan_to_none(forehead),
        "forehead_gated": _nan_to_none(forehead_gated),
        "best_roi": best_roi,
        "by_roi": [_nan_to_none(x) for x in roi_recs],
        "by_fitzpatrick": [_nan_to_none(x) for x in fitz_recs],
    }
    with open(os.path.join(out_dir, "accuracy_summary.json"), "w") as f:
        json.dump(js, f, indent=2)

    return {"clips_scored": len(rows),
            "subjects": int(clip_df["subject_id"].nunique()) if not clip_df.empty else 0,
            "overall_mae": overall["mae"]}
