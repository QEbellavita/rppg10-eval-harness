import numpy as np

# Per-channel pulse weights: green strongest, as in real skin rPPG. The cross-channel
# RATIO (chrominance) is what POS/CHROM project onto; a green-only pulse would be
# partly cancelled as common-mode, so R and B carry scaled copies.
_CHANNEL_WEIGHTS = np.array([0.33, 1.0, 0.5])  # R, G, B


def make_pulse_frames(n_frames=900, fps=30.0, bpm=72.0, size=64, amp=6.0, seed=0):
    """In-memory (N,size,size,3) uint8 video with a sinusoidal pulse across channels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    pulse = amp * np.sin(2 * np.pi * (bpm / 60.0) * t)          # (N,)
    base = np.array([120.0, 130.0, 110.0])                     # R,G,B baseline
    per_channel = pulse[:, None] * _CHANNEL_WEIGHTS[None, :]    # (N,3)
    frames = np.empty((n_frames, size, size, 3), dtype=np.float64)
    for i in range(n_frames):
        frame = np.broadcast_to(base + per_channel[i], (size, size, 3)).copy()
        frame += rng.normal(0, 0.5, frame.shape)               # mild sensor noise
        frames[i] = frame
    return np.clip(frames, 0, 255).astype(np.uint8), fps, bpm


def make_ecg(duration_s=120.0, fs=1000.0, bpm=72.0, wander=True, noise=0.02, seed=0):
    """Synthetic single-lead ECG: sharp QRS-like spikes at a fixed RR interval, with
    optional low-frequency baseline wander + Gaussian noise. Returns (ecg, fs, bpm)."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    ecg = np.zeros(n)
    rr = 60.0 / bpm
    # QRS modelled as a narrow Gaussian (~10 ms) at each beat instant
    beat_times = np.arange(rr, duration_s, rr)
    width = 0.010  # s
    for bt in beat_times:
        ecg += np.exp(-0.5 * ((t - bt) / width) ** 2)
    if wander:
        ecg += 0.3 * np.sin(2 * np.pi * 0.15 * t)   # respiration-rate baseline drift
    ecg += rng.normal(0, noise, n)
    return ecg, fs, bpm

