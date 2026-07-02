<div align="center">

# HaWoR — Improvements & Design Notes

**More accurate, temporally stable hand pose for offline egocentric video**
*Code & post-processing only — no model retraining*

</div>

---

## Table of Contents
- [Scope & principles](#scope--principles)
- [1. Zero-lag temporal smoothing](#1-zero-lag-temporal-smoothing)
- [2. Bone-length canonicalization](#2-bone-length-canonicalization)
- [3. Gap-filling](#3-gap-filling)
- [4. Evaluation harness](#4-evaluation-harness)
- [Focal length: why we keep the default](#focal-length-why-we-keep-the-default)
- [Metrics glossary](#metrics-glossary)
- [Results](#results)
- [Tooling reference](#tooling-reference)

---

## Scope & principles

1. **Backward compatible.** The production `api.py` / Celery pipeline is left untouched.
   The original `smooth_joints()` (Savitzky-Golay) is preserved; new filters are added
   alongside it. Only the `main.py` CLI path opts into the improvements.
2. **Offline-first.** These videos are processed as whole clips, so we can use
   **non-causal (zero-phase)** filters that see past *and* future frames — impossible in
   real-time, but ideal here.
3. **No retraining.** Every gain comes from smoothing / geometric post-processing, so the
   model checkpoint is unchanged and results are fully reproducible.

---

## 1. Zero-lag temporal smoothing

**Problem.** A causal **One-Euro** filter reduces jitter but, being causal, makes the
skeleton visibly *lag behind* fast hand motion.

**Fix.** A **non-causal Gaussian** filter (`scipy.ndimage.gaussian_filter1d`, symmetric
kernel) removes jitter **without lag**, because it uses future frames too.

```python
# hawor/utils/smoothing.py
def gaussian_smooth_joints(joints, sigma=2.0):
    """Zero-phase (non-causal) Gaussian temporal smoothing -- NO lag."""
    from scipy.ndimage import gaussian_filter1d
    if joints.shape[0] < 3:
        return joints
    sm = gaussian_filter1d(joints.numpy(), sigma=sigma, axis=0, mode="nearest")
    return torch.from_numpy(sm).to(joints.dtype)
```

`main.py` now calls `gaussian_smooth_joints()` instead of the causal path. Recommended
`sigma` ≈ 1.5–2.5 frames.

---

## 2. Bone-length canonicalization

Human bone lengths are constant, but per-frame estimates fluctuate. We enforce the
**median** length of each bone while keeping its **direction**, which stabilizes hand
geometry and drives the bone-length coefficient of variation (`bone_CV`) to ~0.

- Applied to **3D world landmarks only** (does not alter the 2D overlay).
- Toggle with `--no-bone` in `tools/postprocess_handpose.py`.

---

## 3. Gap-filling

Short interior detection dropouts are linearly interpolated (up to `--maxgap` frames),
yielding smoother, more continuous tracks before smoothing is applied.

---

## 4. Evaluation harness

`tools/eval_handpose.py` reports objective, scale-aware metrics per hand so improvements
can be measured rather than eyeballed.

---

## Focal length: why we keep the default

A focal sweep (600 vs 1800) showed **600 (default) is best**. Raising the focal did **not**
improve 2D alignment and made 3D **worse** (`bone_CV` 0.055 → 0.071). HaWoR expects a focal
near its training distribution; pushing it off-distribution degrades 3D.

> ⚠️ 2D reprojection is self-consistent at any focal, so overlays *always look fine* — the
> scale-invariant **`bone_CV`** is the decisive metric here, not the overlay.

---

## Metrics glossary

| Metric | Meaning |
| --- | --- |
| `jitter2d` / `jitter3d` | Mean acceleration magnitude (lower = smoother) |
| `coverage` | Fraction of frames with a detected hand |
| `gaps` / `maxgap` | Number / length of detection dropouts |
| `bone_CV` | Coefficient of variation of bone lengths (scale-invariant; 0 = perfectly stable) |

---

## Results

| Metric | Baseline | Improved |
| --- | --- | --- |
| 3D jitter (L / R) | 4.6 / 5.6 mm | **1.4 – 1.8 mm** (~−53%) |
| Bone-length CV | ~0.06 | **0.00** |
| Coverage | 98.4% | 98.4% (unchanged) |
| Temporal lag | present | **none** |

---

## Tooling reference

| Tool | Purpose |
| --- | --- |
| `tools/eval_handpose.py` | Stability metrics from a HaWoR JSON |
| `tools/postprocess_handpose.py` | Gap-fill + smoothing + bone canonicalization |
| `tools/render_overlay.py` | Skeleton overlay from JSON + video (CPU only) |

```bash
# End-to-end offline refinement
python tools/postprocess_handpose.py raw.json final.json --method gaussian --sigma 2.0
python tools/eval_handpose.py final.json
python tools/render_overlay.py input.mp4 final.json overlay.mp4
```
