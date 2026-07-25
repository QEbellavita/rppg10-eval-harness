import numpy as np
from rppg10 import toolbox


def test_load_methods_exposes_panel_and_is_callable():
    methods = toolbox.load_methods()
    for key in ("POS", "CHROM", "GREEN", "ICA"):
        assert key in methods
    # a flat (N,1,1,3) trace must return a 1-D BVP from each panel method
    frames = (np.random.default_rng(1).random((256, 1, 1, 3)) * 255)
    for key in ("POS", "CHROM", "GREEN"):
        bvp = np.asarray(methods[key](frames, 30.0)).reshape(-1)
        assert bvp.ndim == 1 and bvp.shape[0] > 0


def test_post_process_has_expected_functions():
    pp = toolbox.load_post_process()
    for fn in ("_detrend", "_calculate_fft_hr", "_calculate_peak_hr", "_calculate_SNR"):
        assert hasattr(pp, fn)


def test_toolbox_commit_is_a_string():
    assert isinstance(toolbox.toolbox_commit(), str)
