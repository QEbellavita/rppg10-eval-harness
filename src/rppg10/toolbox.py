"""Bridge to the existing rPPG-Toolbox install (reuse, do not reimplement)."""
import os
import sys
import subprocess

DEFAULT_TOOLBOX_PATH = os.path.expanduser("~/rppg-toolbox")

_PATCHED = False  # guard: patch at most once per process


def toolbox_path() -> str:
    return os.environ.get("RPPG_TOOLBOX_PATH", DEFAULT_TOOLBOX_PATH)


def _ensure_on_path() -> None:
    p = toolbox_path()
    if not os.path.isdir(os.path.join(p, "unsupervised_methods")):
        raise RuntimeError(
            f"rPPG-Toolbox not found at {p!r}; set RPPG_TOOLBOX_PATH to its checkout."
        )
    if p not in sys.path:
        sys.path.insert(0, p)


def _patch_slow_detrend() -> None:
    """Idempotent monkeypatch: replace both O(N³) toolbox detrend functions
    with the O(N) banded-solve equivalent from rppg10.postprocess.

    Imported lazily to avoid a circular import (postprocess does NOT import
    from toolbox after clean_bvp was updated to call smoothness_detrend directly).

    Safe under ProcessPoolExecutor: each worker re-imports the module and calls
    load_methods(), which runs this patch in the worker process.
    """
    global _PATCHED
    if _PATCHED:
        return
    from .postprocess import smoothness_detrend  # lazy, one-directional
    import unsupervised_methods.utils as _um_utils
    import evaluation.post_process as _pp
    _um_utils.detrend = smoothness_detrend
    _pp._detrend = smoothness_detrend
    _PATCHED = True


def load_methods():
    _ensure_on_path()
    from unsupervised_methods.methods.POS_WANG import POS_WANG
    from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
    from unsupervised_methods.methods.GREEN import GREEN
    from unsupervised_methods.methods.ICA_POH import ICA_POH
    _patch_slow_detrend()
    return {
        "POS": lambda frames, fps: POS_WANG(frames, fps),
        "CHROM": lambda frames, fps: CHROME_DEHAAN(frames, fps),
        "GREEN": lambda frames, fps: GREEN(frames),
        "ICA": lambda frames, fps: ICA_POH(frames, fps),
    }


def load_post_process():
    _ensure_on_path()
    from evaluation import post_process
    _patch_slow_detrend()
    return post_process


def toolbox_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", toolbox_path(), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
