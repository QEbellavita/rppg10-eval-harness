import numpy as np
import pytest
from rppg10.video_io import decode_rgb_trace, EmptyVideoError


def test_decode_returns_rgb_trace_with_right_shape(written_video):
    path, fps, _ = written_video
    trace = decode_rgb_trace(path)
    assert trace.rgb.shape[1] == 3
    assert trace.n_frames > 800
    assert abs(trace.fps - fps) < 1.0
    assert trace.rgb.dtype == np.float64
    # green channel (index 1) should oscillate -> std clearly above sensor noise floor
    assert trace.rgb[:, 1].std() > 1.0
    # channel-order guard: R baseline (120) > B baseline (110); fails if BGR not flipped
    assert trace.rgb[:, 0].mean() > trace.rgb[:, 2].mean()


def test_detect_mode_falls_back_to_full_when_no_face(written_video):
    """On a synthetic uniform clip (no detectable face), 'detect' must not crash and
    falls back to the whole frame ;  so its trace matches 'full' mode exactly."""
    path, _, _ = written_video
    full = decode_rgb_trace(path, roi_mode="full")
    det = decode_rgb_trace(path, roi_mode="detect")
    assert det.n_frames == full.n_frames
    # no face ever found -> identical whole-frame means
    assert np.allclose(det.rgb, full.rgb)


def test_empty_video_raises(empty_video):
    with pytest.raises(EmptyVideoError):
        decode_rgb_trace(empty_video)


def test_zero_frame_video_raises(zero_frame_video):
    with pytest.raises(EmptyVideoError):
        decode_rgb_trace(zero_frame_video)
