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
