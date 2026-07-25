"""Detrend + zero-phase Butterworth band-pass for the BVP.

smoothness_detrend is an O(N) banded-solve equivalent of the toolbox's
O(N³) dense-inverse detrend. It is mathematically identical to
    detrended = (I − (I + λ²DᵀD)⁻¹) x
but solves (I + λ²DᵀD) z = x via scipy.sparse.linalg.spsolve instead of
materialising an N×N matrix and calling np.linalg.inv.
"""
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

_LAMBDA = 100  # detrend smoothing parameter (toolbox convention)


def smoothness_detrend(input_signal, lambda_value=100.0):
    """O(N) banded-solve equivalent of the toolbox smoothness-prior detrend.

    Drop-in replacement for both ``evaluation.post_process._detrend`` and
    ``unsupervised_methods.utils.detrend``.  Shape-preserving: if the caller
    passes an (N, 1) matrix (as POS_WANG does), the return is also (N, 1).
    """
    arr = np.asarray(input_signal, dtype=float)
    x = arr.ravel()
    n = x.shape[0]
    if n < 3:
        return arr  # nothing to detrend; float-cast intentional (dtype consistency)
    ident = sparse.identity(n, format="csc")
    e = np.ones(n)
    D = sparse.spdiags(
        [e, -2.0 * e, e], [0, 1, 2], n - 2, n, format="csc"
    )
    A = (ident + (lambda_value ** 2) * (D.T @ D)).tocsc()
    z = spsolve(A, x)
    detrended = x - z
    return detrended.reshape(arr.shape)  # preserve input shape


def clean_bvp(bvp, fps, low=0.6, high=3.3) -> np.ndarray:
    x = np.asarray(bvp, dtype=np.float64)
    x = smoothness_detrend(x, _LAMBDA)
    nyq = 0.5 * float(fps)
    b, a = signal.butter(2, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)
