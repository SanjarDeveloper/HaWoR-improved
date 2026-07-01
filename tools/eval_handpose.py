"""Evaluate HaWoR handpose JSON output: stability / quality metrics.

No ground truth needed. Reports, per hand (Left/Right):
  - coverage: fraction of frames the hand is detected
  - gaps: number of missing runs and the longest gap (frames)
  - jitter_2d: mean per-joint acceleration in pixels/frame^2 (lower = smoother)
  - jitter_3d: mean per-joint acceleration in mm/frame^2 (from world landmarks)
  - bone_cv: mean coefficient of variation of bone lengths over time
             (lower = more anatomically stable; a rigid hand keeps constant bones)

Usage:  python tools/eval_handpose.py <output.json>
"""
import sys
import json
import numpy as np


def _hand_arrays(data, category):
    """Return (T,21,3) img landmarks and (T,21,3) world landmarks with NaN for
    frames where `category` ('Left'/'Right') is absent."""
    T = data["frame_count"]
    img = np.full((T, 21, 3), np.nan)
    world = np.full((T, 21, 3), np.nan)
    for fr in data["frames"]:
        t = fr["frame_index"]
        for h, hand in enumerate(fr.get("handedness", [])):
            if hand.get("category_name") != category:
                continue
            lm = fr["hand_landmarks"][h]
            img[t] = [[p["x"], p["y"], p["z"]] for p in lm]
            wl = fr.get("hand_world_landmarks")
            if wl:
                world[t] = [[p["x"], p["y"], p["z"]] for p in wl[h]]
    return img, world


def _coverage_gaps(present):
    T = len(present)
    cov = present.sum() / T
    gaps = []
    i = 0
    while i < T:
        if not present[i]:
            j = i
            while j < T and not present[j]:
                j += 1
            gaps.append(j - i)
            i = j
        else:
            i += 1
    return cov, len(gaps), (max(gaps) if gaps else 0)


def _jitter(pts):
    """Mean magnitude of 2nd temporal difference over contiguous present runs."""
    present = ~np.isnan(pts[:, 0, 0])
    accs = []
    T = len(pts)
    i = 0
    while i < T:
        if present[i]:
            j = i
            while j < T and present[j]:
                j += 1
            run = pts[i:j]  # (L,21,C)
            if len(run) >= 3:
                acc = np.diff(run, n=2, axis=0)  # (L-2,21,C)
                accs.append(np.linalg.norm(acc, axis=-1))  # (L-2,21)
            i = j
        else:
            i += 1
    if not accs:
        return float("nan")
    return float(np.concatenate(accs).mean())


def _bone_cv(world, connections):
    present = ~np.isnan(world[:, 0, 0])
    w = world[present]
    if len(w) < 2:
        return float("nan")
    cvs = []
    for a, b in connections:
        d = np.linalg.norm(w[:, a] - w[:, b], axis=-1)  # (T,)
        m = d.mean()
        if m > 1e-9:
            cvs.append(d.std() / m)
    return float(np.mean(cvs)) if cvs else float("nan")


def main(path):
    data = json.load(open(path))
    W, H = data["image_width"], data["image_height"]
    conns = data.get("hand_connections", [])
    print(f"file: {path}")
    print(f"frames: {data['frame_count']}  resolution: {W}x{H}")
    print(f"{'hand':6} {'coverage':>9} {'gaps':>5} {'maxgap':>7} "
          f"{'jitter2d_px':>12} {'jitter3d_mm':>12} {'bone_cv':>8}")
    for cat in ("Left", "Right"):
        img, world = _hand_arrays(data, cat)
        present = ~np.isnan(img[:, 0, 0])
        if present.sum() == 0:
            print(f"{cat:6} {'--- not detected ---':>40}")
            continue
        cov, ngaps, maxgap = _coverage_gaps(present)
        # pixel-space landmarks: x,y are normalized -> scale to pixels
        px = img.copy()
        px[:, :, 0] *= W
        px[:, :, 1] *= H
        j2d = _jitter(px[:, :, :2])
        # world landmarks are in meters -> mm
        j3d = _jitter(world * 1000.0)
        bcv = _bone_cv(world * 1000.0, conns)
        print(f"{cat:6} {cov:9.3f} {ngaps:5d} {maxgap:7d} "
              f"{j2d:12.3f} {j3d:12.3f} {bcv:8.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/eval_handpose.py <output.json>")
        sys.exit(1)
    main(sys.argv[1])
