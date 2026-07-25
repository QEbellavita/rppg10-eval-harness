import numpy as np
from rppg10.methods import extract_bvp, PANEL, PRIMARY
from rppg10.toolbox import load_methods


def test_panel_constants():
    assert PRIMARY == "POS"
    assert PANEL == ["POS", "CHROM", "GREEN"]


def test_reshape_passthrough_equivalence():
    """method(full HxW frames) must equal method(spatial-mean trace as (N,1,1,3))."""
    rng = np.random.default_rng(7)
    frames = rng.random((400, 8, 8, 3)) * 255.0          # full frames
    trace = frames.reshape(frames.shape[0], -1, 3).mean(axis=1)  # (400,3) spatial mean
    methods = load_methods()
    for key in PANEL:
        bvp_full = np.asarray(methods[key](frames, 30.0)).reshape(-1)
        bvp_trace = extract_bvp(trace, 30.0, key)
        assert bvp_full.shape == bvp_trace.shape
        assert np.allclose(bvp_full, bvp_trace, atol=1e-6), f"{key} mismatch"


def test_extract_bvp_returns_1d():
    trace = (np.random.default_rng(3).random((300, 3)) * 255.0)
    bvp = extract_bvp(trace, 30.0, "POS")
    assert bvp.ndim == 1 and bvp.shape[0] == 300
