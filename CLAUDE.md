# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does
HaWoR (Hand-World-Space Reconstruction) — CVPR 2025 Highlight. Detects hands in egocentric video using YOLO, estimates 3D hand pose via a ViT + MANO transformer, and outputs MediaPipe-compatible 21-joint hand landmarks in JSON.

## Commands

```bash
# Install dependencies
uv sync

# Run inference on all videos in a directory
./run_inference.sh /path/to/video_dir
# or directly:
uv run python main.py --video_folder /path/to/video_dir --img_focal 600

# Full setup including model weights + DROID-SLAM (first-time only)
./run.sh

# Run as pipeline worker (Celery)
PYTHONPATH="$(cd .. && pwd):$PYTHONPATH" uv run celery -A pipeline.celery_app worker -Q stage_5_queue -c 1
```

There are no tests or linting configured in this repository.

## Environment
- **Python 3.10** (pinned: `>=3.10,<3.11`)
- **uv** for dependency management (`pyproject.toml` + `uv.lock`)
- **PyTorch 2.8+ with CUDA 12.8** from the PyTorch extra index
- **NVIDIA GPU** required for inference
- Must set `MODERNGL_BACKEND=egl` for rendering; `api.py` auto-sets `PYOPENGL_PLATFORM=egl`

## Architecture

### Pipeline flow (6 stages in `api.py:process_hand_pose()`)
1. **Frame extraction** — ffmpeg subprocess extracts PNGs at 30 fps
2. **YOLO hand detection + tracking** — `lib/pipeline/tools.py:detect_track()`, batched (batch_size=16)
3. **HaWoR motion estimation** — ViT backbone → space-time attention → MANO transformer decoder; runs per hand track
4. **MANO forward pass** — `hawor/utils/process.py` (`run_mano` / `run_mano_left`) generates 3D joints (21 joints, 778 vertices)
5. **Temporal smoothing + 2D projection** — Savitzky-Golay filter (`hawor/utils/smoothing.py`), pinhole camera projection
6. **Output** — MediaPipe-compatible JSON; optional overlay video with skeleton rendering

### Two entry points
- **`api.py`** — Programmatic API (`process_hand_pose()`). Used by the pipeline worker. Lazy-loads the model as a singleton. Contains all 6 pipeline stages as private functions.
- **`main.py`** — CLI entry point for batch processing. Calls the lower-level scripts (`detect_track_video`, `hawor_motion_estimation`) directly rather than going through `api.py`.

### Pipeline worker integration
- `cli_process.py` — Subprocess entry point invoked by the parent `pipeline-worker-template`. Reports progress to Redis (both Celery task state and job snapshots). Calls `api.process_hand_pose()`.
- `run.sh` — Full setup + Celery worker launch (stage_5_queue, concurrency 1).
- Import from parent repo: `from HaWoR.api import process_hand_pose, HAWOR_VERSION`

### Key internal modules
| Path | Role |
|------|------|
| `lib/models/hawor.py` | Core model (ViT backbone + MANO transformer head) |
| `lib/pipeline/tools.py` | YOLO detection, tracking, chunk parsing |
| `lib/models/mano_wrapper.py` | MANO parametric hand model wrapper |
| `hawor/configs/__init__.py` | Config loading (yacs CfgNode) |
| `hawor/utils/process.py` | MANO forward pass helpers |
| `hawor/utils/rotation.py` | Rotation matrix <-> angle-axis conversions |
| `hawor/utils/smoothing.py` | Savitzky-Golay temporal smoothing |
| `scripts/scripts_test_video/` | Video detection + motion estimation orchestrators |

### Intermediate outputs (cached on disk)
Each video `<name>` creates a directory `<video_dir>/<name>/` containing:
- `extracted_images/` — PNG frames from ffmpeg
- `tracks_<start>_<end>/model_tracks.npy` — YOLO detection tracks
- `cam_space/{0,1}/<first>_<last>.json` — per-chunk camera-space predictions (0=left, 1=right)
- `tracks_<start>_<end>/frame_chunks_all.npy` — cached frame chunk indices

## Model Weights (not in repo, ~3.5 GB total)
| Path | Purpose |
|------|---------|
| `weights/hawor/checkpoints/hawor.ckpt` | Main HaWoR model (3.1 GB) |
| `weights/hawor/checkpoints/infiller.pt` | Temporal infiller (400 MB, optional) |
| `weights/external/detector.pt` | YOLO hand detector (52 MB) |
| `weights/hawor/model_config.yaml` | Model architecture config |
| `_DATA/data/mano/MANO_RIGHT.pkl` | Right hand MANO model |
| `_DATA/data_left/mano_left/MANO_LEFT.pkl` | Left hand MANO model |
| `_DATA/data/mano_mean_params.npz` | Mean hand pose/shape |

## Known Gotchas
- PyTorch must be the CUDA build — CPU-only torch will fail silently on model loading
- Model weights are git-ignored — download via `run.sh` or mount at runtime
- `_DATA/` MANO `.pkl` files require acceptance of the MANO license
- Frame extraction uses ffmpeg via subprocess — must be installed system-wide
- First inference is slower due to CUDA kernel compilation and model loading
- The `sys.path` manipulation in `api.py` and `cli_process.py` is intentional — HaWoR internal modules expect the repo root on `sys.path`
- Left hand inference flips the image and negates Y/Z rotation axes (see `_run_motion_estimation` in `api.py`)
