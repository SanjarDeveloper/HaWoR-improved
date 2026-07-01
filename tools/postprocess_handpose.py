"""Post-process a HaWoR handpose JSON to reduce jitter (One-Euro filter).

Operates directly on the exported MediaPipe JSON -- no model / GPU needed.
Smooths both image landmarks and world landmarks per hand, per coordinate,
only across contiguous detected runs (never invents data across gaps).

Usage:
  python tools/postprocess_handpose.py <in.json> <out.json> [--mincutoff 1.0] [--beta 0.3]

One-Euro filter (Casiez et al. 2012): adaptive low-pass -- smooths slow motion
hard (kills jitter) but lets fast motion through (low lag). fps taken from
frame rate 30 unless overridden.
"""
import sys
import json
import argparse
import numpy as np


def _smoothing_alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(series, present, fps=30.0, mincutoff=1.0, beta=0.3, dcutoff=1.0):
    """One-Euro filter a (T,) series; only within contiguous present runs."""
    out = series.copy()
    dt = 1.0 / fps
    T = len(series)
    i = 0
    while i < T:
        if not present[i]:
            i += 1
            continue
        j = i
        while j < T and present[j]:
            j += 1
        x_prev = series[i]
        dx_prev = 0.0
        out[i] = x_prev
        for t in range(i + 1, j):
            dx = (series[t] - x_prev) / dt
            a_d = _smoothing_alpha(dcutoff, dt)
            dx_hat = a_d * dx + (1 - a_d) * dx_prev
            cutoff = mincutoff + beta * abs(dx_hat)
            a = _smoothing_alpha(cutoff, dt)
            x_hat = a * series[t] + (1 - a) * x_prev
            out[t] = x_hat
            x_prev, dx_prev = x_hat, dx_hat
        i = j
    return out


def smooth_landmark_block(frames, key, present_by_frame, fps, mincutoff, beta):
    """Smooth frames[*][key][hand_slot] in place for one hand slot mapping.
    present_by_frame: dict frame_index -> hand_slot index for this category."""
    # gather (T,21,3)
    T = len(present_by_frame)
    idxs = sorted(present_by_frame)
    # Build dense arrays keyed by frame_index
    maxf = max(f["frame_index"] for f in frames) + 1
    arr = np.full((maxf, 21, 3), np.nan)
    slot_of = {}
    for f in frames:
        fi = f["frame_index"]
        if fi in present_by_frame:
            slot = present_by_frame[fi]
            block = f.get(key)
            if block is None:
                continue
            arr[fi] = [[p["x"], p["y"], p["z"]] for p in block[slot]]
            slot_of[fi] = slot
    present = ~np.isnan(arr[:, 0, 0])
    for j in range(21):
        for c in range(3):
            arr[:, j, c] = one_euro(arr[:, j, c], present, fps, mincutoff, beta)
    # write back
    for f in frames:
        fi = f["frame_index"]
        if fi in slot_of:
            block = f[key][slot_of[fi]]
            for j in range(21):
                block[j]["x"] = float(arr[fi, j, 0])
                block[j]["y"] = float(arr[fi, j, 1])
                block[j]["z"] = float(arr[fi, j, 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--mincutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    args = ap.parse_args()

    data = json.load(open(args.inp))
    frames = data["frames"]

    for cat in ("Left", "Right"):
        present_by_frame = {}
        for f in frames:
            for slot, hand in enumerate(f.get("handedness", [])):
                if hand.get("category_name") == cat:
                    present_by_frame[f["frame_index"]] = slot
        if not present_by_frame:
            continue
        for key in ("hand_landmarks", "hand_world_landmarks"):
            if any(key in f for f in frames):
                smooth_landmark_block(frames, key, present_by_frame,
                                      args.fps, args.mincutoff, args.beta)

    json.dump(data, open(args.out, "w"))
    print(f"wrote {args.out}  (mincutoff={args.mincutoff} beta={args.beta})")


if __name__ == "__main__":
    main()
