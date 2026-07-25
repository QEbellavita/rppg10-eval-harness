"""Stream-decode a video into a spatial-mean RGB trace (N,3), RGB order.

Two ROI modes:
  - "full"   : spatial-mean over the WHOLE frame. Correct when the input is already a
               cropped skin patch (e.g. Dataset_rPPG-10 Forehead/Cheek files). Default,
               so Spec A behaviour is unchanged.
  - "detect" : Haar-detect the face (re-detected periodically), crop to a central skin
               sub-region of the face box, and spatial-mean that. Needed for full-frame
               webcam datasets where the whole frame is dominated by
               non-skin pixels (hair, clothing, background) and the pulse drowns out.
               Falls back to the whole frame if no face is ever found.
"""
from dataclasses import dataclass
import numpy as np
import cv2


class EmptyVideoError(Exception):
    """Raised for 0-byte, unreadable, or 0-frame videos (e.g. Subject_4)."""


@dataclass
class RgbTrace:
    rgb: np.ndarray   # (N,3) float64, channel order R,G,B
    fps: float
    n_frames: int


_CASCADE = None


def _face_cascade():
    """Lazily load the frontal-face Haar cascade (once per process)."""
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def _detect_face(frame):
    """Return the largest face box (x, y, w, h) or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # largest by area ;  the subject's face, not a background false positive
    return max(faces, key=lambda b: b[2] * b[3])


def _skin_region(frame, box):
    """Crop a central forehead+cheek skin band inside the face box.

    Avoids the background corners of the box, the eyes/brows at the very top, and the
    mouth/jaw at the bottom (those move and add motion noise). Vertical 10-70 %,
    horizontal 15-85 % of the box.
    """
    x, y, w, h = box
    y0, y1 = y + int(0.10 * h), y + int(0.70 * h)
    x0, x1 = x + int(0.15 * w), x + int(0.85 * w)
    H, W = frame.shape[:2]
    y0, y1 = max(0, y0), min(H, y1)
    x0, x1 = max(0, x0), min(W, x1)
    if y1 <= y0 or x1 <= x0:
        return frame
    return frame[y0:y1, x0:x1]


def decode_rgb_trace(path, roi_mode="full", redetect_every=30) -> RgbTrace:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise EmptyVideoError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    detect = roi_mode == "detect"
    box = None        # last good face box, reused between re-detects
    means = []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if detect:
                if box is None or (i % redetect_every == 0):
                    new = _detect_face(frame)
                    if new is not None:
                        box = new
                region = _skin_region(frame, box) if box is not None else frame
            else:
                region = frame
            bgr_mean = region.reshape(-1, 3).mean(axis=0)   # (3,) in B,G,R
            means.append(bgr_mean[[2, 1, 0]])               # -> R,G,B (owned copy)
            i += 1
    finally:
        cap.release()
    if not means:
        raise EmptyVideoError(f"no frames decoded: {path}")
    return RgbTrace(rgb=np.asarray(means, dtype=np.float64),
                    fps=float(fps), n_frames=len(means))
