import cv2
import pytest
from synthetic import make_pulse_frames


@pytest.fixture
def pulse_frames():
    return make_pulse_frames()


@pytest.fixture
def written_video(tmp_path, pulse_frames):
    """Write the synthetic frames to a lossless .avi so decode preserves the signal."""
    frames, fps, bpm = pulse_frames
    path = tmp_path / "synthetic.avi"
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    assert writer.isOpened(), "FFV1 writer unavailable"
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))   # store as BGR
    writer.release()
    return path, fps, bpm


@pytest.fixture
def empty_video(tmp_path):
    path = tmp_path / "empty.avi"
    path.write_bytes(b"")
    return path


@pytest.fixture
def zero_frame_video(tmp_path):
    """Valid AVI header, zero frames ;  exercises the `not means` guard in decode_rgb_trace."""
    path = tmp_path / "zero_frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (64, 64))
    assert writer.isOpened(), "FFV1 writer unavailable"
    writer.release()   # close without writing any frames
    return path
