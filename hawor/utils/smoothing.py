"""Smoothing utilities for hand joint trajectories."""

import torch
from scipy.signal import savgol_filter


def smooth_joints(joints, window_length=7, polyorder=3):
    """Apply Savitzky-Golay filter along the temporal axis to reduce jitter.

    Args:
        joints: Tensor of shape (T, 21, 3).
        window_length: Must be odd and <= T.
        polyorder: Polynomial order (must be < window_length).

    Returns:
        Smoothed torch.Tensor, same shape and dtype.
    """
    T = joints.shape[0]
    if T < window_length:
        window_length = T if T % 2 == 1 else T - 1
    if window_length < polyorder + 1:
        return joints
    np_joints = joints.numpy()
    smoothed = savgol_filter(np_joints, window_length, polyorder, axis=0, mode="interp")
    return torch.from_numpy(smoothed).to(joints.dtype)


def _one_euro_alpha(cutoff, dt):
    import math
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro_smooth_joints(joints, fps=30.0, mincutoff=1.0, beta=0.3, dcutoff=1.0):
    """One-Euro filter along the temporal axis (Casiez et al. 2012).

    Adaptive low-pass: smooths slow motion hard (removes jitter) while letting
    fast motion through with low lag. Expects a *contiguous* (no gaps) run.

    Args:
        joints: Tensor of shape (T, 21, 3), assumed temporally contiguous.
    Returns:
        Smoothed torch.Tensor, same shape and dtype.
    """
    import numpy as np
    T = joints.shape[0]
    if T < 3:
        return joints
    x = joints.numpy().astype(np.float64)          # (T, 21, 3)
    out = x.copy()
    dt = 1.0 / fps
    dx_prev = np.zeros_like(x[0])
    for t in range(1, T):
        dx = (x[t] - out[t - 1]) / dt
        a_d = _one_euro_alpha(dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = mincutoff + beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * cutoff)) / dt)
        out[t] = a * x[t] + (1 - a) * out[t - 1]
        dx_prev = dx_hat
    return torch.from_numpy(out).to(joints.dtype)
