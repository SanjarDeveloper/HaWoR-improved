"""Post-process a HaWoR handpose JSON to improve stability -- no model/GPU.

Pipeline (per hand, applied to hand_landmarks + hand_world_landmarks):
  1. gap-fill : linearly interpolate short interior gaps (<= max_gap frames)
  2. one-euro : adaptive low-pass smoothing (kills jitter, low lag)
  3. bone-canon (world only): enforce constant (median) bone lengths while
                keeping joint directions -> stabilises hand geometry (bone_CV~0)

Usage:
  python tools/postprocess_handpose.py <in.json> <out.json>
      [--fps 30] [--mincutoff 1.0] [--beta 0.3] [--maxgap 10] [--no-bone]
"""
import sys
import json
import argparse
import numpy as np

# MediaPipe 21-joint hand kinematic tree: child -> parent
PARENT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7,
          9: 0, 10: 9, 11: 10, 12: 11, 13: 0, 14: 13, 15: 14, 16: 15,
          17: 0, 18: 17, 19: 18, 20: 19}
TREE_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def _one_euro_alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(series, present, fps, mincutoff, beta, dcutoff=1.0):
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
        x_prev, dx_prev = series[i], 0.0
        for t in range(i + 1, j):
            dx = (series[t] - x_prev) / dt
            a_d = _one_euro_alpha(dcutoff, dt)
            dx_hat = a_d * dx + (1 - a_d) * dx_prev
            a = _one_euro_alpha(mincutoff + beta * abs(dx_hat), dt)
            x_prev = a * series[t] + (1 - a) * x_prev
            out[t] = x_prev
            dx_prev = dx_hat
        i = j
    return out


def fill_gaps(arr, present, max_gap):
    """Linearly interpolate interior gaps up to max_gap. Returns new present mask."""
    T = len(present)
    present = present.copy()
    i = 0
    while i < T:
        if present[i]:
            i += 1
            continue
        j = i
        while j < T and not present[j]:
            j += 1
        if i > 0 and j < T and (j - i) <= max_gap:      # interior + short
            for t in range(i, j):
                w = (t - (i - 1)) / (j - (i - 1))
                arr[t] = (1 - w) * arr[i - 1] + w * arr[j]
                present[t] = True
        i = j
    return present


def canon_bones(world, present):
    """Enforce median bone lengths, keeping bone directions. world: (T,21,3)."""
    lens = {c: [] for c in TREE_ORDER}
    for t in np.where(present)[0]:
        for c in TREE_ORDER:
            lens[c].append(np.linalg.norm(world[t, c] - world[t, PARENT[c]]))
    canon = {c: (np.median(v) if v else 0.0) for c, v in lens.items()}
    for t in np.where(present)[0]:
        for c in TREE_ORDER:
            p = PARENT[c]
            vec = world[t, c] - world[t, p]
            n = np.linalg.norm(vec)
            if n > 1e-9:
                world[t, c] = world[t, p] + vec / n * canon[c]
    return world


def _extract(frames, slot_by_frame, key, maxf):
    arr = np.full((maxf, 21, 3), np.nan)
    for f in frames:
        fi = f["frame_index"]
        if fi in slot_by_frame and key in f:
            blk = f[key][slot_by_frame[fi]]
            arr[fi] = [[p["x"], p["y"], p["z"]] for p in blk]
    return arr


def _writeback(frames, slot_by_frame, key, arr, present):
    for f in frames:
        fi = f["frame_index"]
        if fi not in slot_by_frame or key not in f:
            continue
        if not present[fi]:
            continue
        blk = f[key][slot_by_frame[fi]]
        for j in range(21):
            blk[j]["x"], blk[j]["y"], blk[j]["z"] = (
                float(arr[fi, j, 0]), float(arr[fi, j, 1]), float(arr[fi, j, 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--mincutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--maxgap", type=int, default=10)
    ap.add_argument("--no-bone", action="store_true")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    frames = data["frames"]
    maxf = max(f["frame_index"] for f in frames) + 1

    for cat in ("Left", "Right"):
        slot_by_frame = {}
        for f in frames:
            for slot, h in enumerate(f.get("handedness", [])):
                if h.get("category_name") == cat:
                    slot_by_frame[f["frame_index"]] = slot
        if not slot_by_frame:
            continue
        for key in ("hand_landmarks", "hand_world_landmarks"):
            if not any(key in f for f in frames):
                continue
            arr = _extract(frames, slot_by_frame, key, maxf)
            present = ~np.isnan(arr[:, 0, 0])
            present = fill_gaps(arr, present, args.maxgap)
            if not args.no_smooth:
                for j in range(21):
                    for c in range(3):
                        arr[:, j, c] = one_euro(arr[:, j, c], present,
                                                args.fps, args.mincutoff, args.beta)
            if key == "hand_world_landmarks" and not args.no_bone:
                arr = canon_bones(arr, present)
            _writeback(frames, slot_by_frame, key, arr, present)
        # note: filled frames need handedness entries to be counted; they already
        # have a slot only if detected. Interpolated-only frames stay as-is in JSON
        # structure (landmarks updated where a slot exists).

    json.dump(data, open(args.out, "w"))
    print(f"wrote {args.out} (mincutoff={args.mincutoff} beta={args.beta} "
          f"maxgap={args.maxgap} bone={not args.no_bone})")


if __name__ == "__main__":
    main()
