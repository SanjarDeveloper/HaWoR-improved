"""Public API for HaWoR hand-pose estimation.

Mirrors the facade pattern used by svo2_to_mcap_converter/api.py so that the
pipeline worker can import a single entry-point without touching HaWoR internals.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

# Ensure the HaWoR repo root is importable regardless of working directory.
_HAWOR_ROOT = str(Path(__file__).resolve().parent)
if _HAWOR_ROOT not in sys.path:
    sys.path.insert(0, _HAWOR_ROOT)

# Headless rendering — must be set before any OpenGL / pyrender import.
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

HAWOR_VERSION = "1.0.0"

ProgressCB = Callable[[str, int, str | None], None]

# ---------------------------------------------------------------------------
# Drawing constants for overlay rendering
# ---------------------------------------------------------------------------
HAND_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],        # thumb
    [0, 5], [5, 6], [6, 7], [7, 8],        # index
    [0, 9], [9, 10], [10, 11], [11, 12],   # middle
    [0, 13], [13, 14], [14, 15], [15, 16], # ring
    [0, 17], [17, 18], [18, 19], [19, 20], # pinky
]

JOINT_COLORS_BGR = [
    (100, 100, 100),                                        # 0: wrist
    (0, 0, 100), (0, 0, 150), (0, 0, 200), (0, 0, 255),   # 1-4: thumb (red)
    (0, 100, 100), (0, 150, 150), (0, 200, 200), (0, 255, 255),  # 5-8: index (yellow)
    (50, 100, 0), (75, 150, 0), (100, 200, 0), (125, 255, 0),    # 9-12: middle (teal)
    (100, 50, 0), (150, 75, 0), (200, 100, 0), (255, 125, 0),    # 13-16: ring (blue)
    (100, 0, 100), (150, 0, 150), (200, 0, 200), (255, 0, 255),  # 17-20: pinky (magenta)
]

BONE_COLORS_BGR = [JOINT_COLORS_BGR[j2] for (_, j2) in HAND_SKELETON]

BBOX_COLOR = {0: (255, 128, 0), 1: (0, 200, 0)}
BBOX_LABEL = {0: "Left", 1: "Right"}


def _is_hand_in_frame(keypoints_2d, img_shape, margin=100):
    H, W = img_shape[:2]
    x, y = keypoints_2d[0, 0], keypoints_2d[0, 1]
    if x < -margin or x > W + margin or y < -margin or y > H + margin:
        return False
    in_bounds = 0
    for j in range(len(keypoints_2d)):
        x, y = keypoints_2d[j, 0], keypoints_2d[j, 1]
        if -margin <= x <= W + margin and -margin <= y <= H + margin:
            in_bounds += 1
    return in_bounds >= len(keypoints_2d) // 2


def _draw_hand_bbox(img, keypoints_2d, is_right=1, padding=15, thickness=2):
    import cv2
    H, W = img.shape[:2]
    if not _is_hand_in_frame(keypoints_2d, img.shape):
        return img
    pts = []
    for j in range(len(keypoints_2d)):
        x, y = int(keypoints_2d[j, 0]), int(keypoints_2d[j, 1])
        if 0 <= x < W and 0 <= y < H:
            pts.append((x, y))
    if len(pts) < 2:
        return img
    xs, ys = zip(*pts)
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(W - 1, max(xs) + padding)
    y2 = min(H - 1, max(ys) + padding)
    color = BBOX_COLOR[is_right]
    label = BBOX_LABEL[is_right]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    label_y = y1 - 6
    if label_y - th < 0:
        label_y = y2 + th + 6
    cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 4), color, -1)
    cv2.putText(img, label, (x1 + 2, label_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _draw_hand_skeleton(img, keypoints_2d, is_right=1, radius=4, thickness=2):
    import cv2
    H, W = img.shape[:2]
    if not _is_hand_in_frame(keypoints_2d, img.shape):
        return img
    img = img.copy()
    for bone_idx, (j1, j2) in enumerate(HAND_SKELETON):
        x1, y1 = int(keypoints_2d[j1, 0]), int(keypoints_2d[j1, 1])
        x2, y2 = int(keypoints_2d[j2, 0]), int(keypoints_2d[j2, 1])
        if not (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
            continue
        cv2.line(img, (x1, y1), (x2, y2), BONE_COLORS_BGR[bone_idx], thickness)
    for j in range(len(keypoints_2d)):
        x, y = int(keypoints_2d[j, 0]), int(keypoints_2d[j, 1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(img, (x, y), radius, JOINT_COLORS_BGR[j], -1)
    return img


def _draw_wrist_axes(img, wrist_2d, axes_ends_2d, thickness=2):
    """Draw three perpendicular orientation axes (X=red, Y=green, Z=blue) at the wrist."""
    import cv2
    H, W = img.shape[:2]
    ox, oy = int(wrist_2d[0]), int(wrist_2d[1])
    if not (0 <= ox < W and 0 <= oy < H):
        return img
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X=red, Y=green, Z=blue
    for axis_idx in range(3):
        ex, ey = int(axes_ends_2d[axis_idx, 0]), int(axes_ends_2d[axis_idx, 1])
        cv2.line(img, (ox, oy), (ex, ey), colors[axis_idx], thickness)
    return img


# Default checkpoint paths relative to the repo root.
_DEFAULT_CHECKPOINT = os.path.join(_HAWOR_ROOT, "weights", "hawor", "checkpoints", "hawor.ckpt")
_DEFAULT_DETECTOR = os.path.join(_HAWOR_ROOT, "weights", "external", "detector.pt")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (heavy imports + GPU model loading deferred)
# ---------------------------------------------------------------------------
_hawor_model = None
_hawor_cfg = None


def _load_model(checkpoint: str | None = None):
    """Load the HaWoR model once and cache it."""
    global _hawor_model, _hawor_cfg
    if _hawor_model is not None:
        return _hawor_model, _hawor_cfg

    import torch
    from scripts.scripts_test_video.hawor_video import load_hawor

    ckpt = checkpoint or _DEFAULT_CHECKPOINT
    model, cfg = load_hawor(ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    _hawor_model = model
    _hawor_cfg = cfg
    return model, cfg


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _get_video_frame_count(video_path: str) -> int | None:
    """Use ffprobe to estimate the number of frames at 30 fps."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        )
        return int(float(result.stdout.strip()) * 30)
    except Exception:
        return None


def _extract_frames(
    video_path: str,
    output_folder: str,
    progress_cb: Callable[[int], None] | None = None,
) -> None:
    """Extract frames at 30 fps using ffmpeg."""
    os.makedirs(output_folder, exist_ok=True)

    total_frames = _get_video_frame_count(video_path) if progress_cb else None

    if progress_cb and total_frames:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-i", video_path,
                "-vf", "fps=30",
                "-start_number", "0",
                "-progress", "pipe:1",
                os.path.join(output_folder, "%04d.png"),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        for line in proc.stdout:
            if line.startswith("frame="):
                try:
                    frame = int(line.split("=", 1)[1].strip())
                    progress_cb(min(99, int(frame / total_frames * 100)))
                except (ValueError, ZeroDivisionError):
                    pass
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    else:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", "fps=30",
                "-start_number", "0",
                "-loglevel", "warning",
                os.path.join(output_folder, "%04d.png"),
            ],
            check=True,
        )


# ---------------------------------------------------------------------------
# Detection + tracking
# ---------------------------------------------------------------------------

def _detect_and_track(
    imgfiles: list[str],
    seq_folder: str,
    progress_cb: Callable[[int], None] | None = None,
) -> tuple:
    """Run YOLO hand detection and tracking. Returns (start_idx, end_idx, tracks)."""
    import numpy as np
    from natsort import natsorted
    from lib.pipeline.tools import detect_track

    start_idx = 0
    end_idx = len(imgfiles)
    tracks_path = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "model_tracks.npy")

    if os.path.exists(tracks_path):
        tracks = np.load(tracks_path, allow_pickle=True).item()
    else:
        os.makedirs(os.path.dirname(tracks_path), exist_ok=True)
        tracks = detect_track(imgfiles, thresh=0.2, progress_cb=progress_cb)
        np.save(tracks_path, tracks)

    return start_idx, end_idx, tracks


# ---------------------------------------------------------------------------
# Motion estimation
# ---------------------------------------------------------------------------

def _run_motion_estimation(
    imgfiles,
    seq_folder: str,
    start_idx: int,
    end_idx: int,
    img_focal: float,
    checkpoint: str | None = None,
    progress_cb: ProgressCB | None = None,
):
    """Run HaWoR inference on all detected hand tracks.

    Returns (frame_chunks_all, img_focal).
    """
    import cv2
    import joblib
    import numpy as np
    import torch
    from collections import defaultdict
    from natsort import natsorted
    from glob import glob

    from lib.pipeline.tools import parse_chunks
    from lib.eval_utils.custom_utils import interpolate_bboxes
    from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

    cached_chunks = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "frame_chunks_all.npy")
    if os.path.exists(cached_chunks):
        frame_chunks_all = joblib.load(cached_chunks)
        return frame_chunks_all, img_focal

    model, _ = _load_model(checkpoint)
    device = next(model.parameters()).device

    tracks = np.load(
        os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "model_tracks.npy"),
        allow_pickle=True,
    ).item()

    imgfiles_arr = np.array(imgfiles)

    tid = np.array([tr for tr in tracks])

    # Split tracks into left/right hands.
    left_trk, right_trk = [], []
    for idx in tid:
        trk = tracks[idx]
        valid = np.array([t["det"] for t in trk])
        is_right = np.concatenate([t["det_handedness"] for t in trk])[valid]
        if is_right.sum() / len(is_right) < 0.5:
            left_trk.extend(trk)
        else:
            right_trk.extend(trk)

    left_trk = sorted(left_trk, key=lambda x: x["frame"])
    right_trk = sorted(right_trk, key=lambda x: x["frame"])
    final_tracks = {0: left_trk, 1: right_trk}

    img = cv2.imread(imgfiles[0])
    img_center = [img.shape[1] / 2, img.shape[0] / 2]

    frame_chunks_all = defaultdict(list)
    total_chunks = 0
    done_chunks = 0

    # Count total work for progress reporting.
    for idx in [0, 1]:
        trk = final_tracks[idx]
        valid = np.array([t["det"] for t in trk])
        if valid.sum() < 2:
            continue
        boxes = np.concatenate([t["det_box"] for t in trk])
        non_zero = np.where(np.any(boxes != 0, axis=1))[0]
        if len(non_zero) == 0:
            continue
        frame = np.array([t["frame"] for t in trk])[valid]
        first_nz, last_nz = non_zero[0], non_zero[-1]
        sub_boxes = boxes[first_nz : last_nz + 1]
        sub_boxes = interpolate_bboxes(sub_boxes)
        fc, _ = parse_chunks(frame, sub_boxes, min_len=1)
        total_chunks += len(fc)

    for idx in [0, 1]:
        trk = final_tracks[idx]
        valid = np.array([t["det"] for t in trk])
        if valid.sum() < 2:
            continue

        boxes = np.concatenate([t["det_box"] for t in trk])
        non_zero = np.where(np.any(boxes != 0, axis=1))[0]
        first_nz, last_nz = non_zero[0], non_zero[-1]
        boxes[first_nz : last_nz + 1] = interpolate_bboxes(boxes[first_nz : last_nz + 1])
        valid[first_nz : last_nz + 1] = True

        boxes = boxes[first_nz : last_nz + 1]
        is_right = np.concatenate([t["det_handedness"] for t in trk])[valid]
        frame = np.array([t["frame"] for t in trk])[valid]

        if is_right.sum() / len(is_right) < 0.5:
            is_right = np.zeros((len(boxes), 1))
        else:
            is_right = np.ones((len(boxes), 1))

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
        frame_chunks_all[idx] = frame_chunks

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            img_ck = imgfiles_arr[frame_ck]
            do_flip = is_right[0] <= 0

            inference_cb = None
            if progress_cb and total_chunks > 0:
                _dc = done_chunks  # capture current value
                def inference_cb(batch_pct, _dc=_dc):
                    overall_pct = int((_dc * 100 + batch_pct) / total_chunks)
                    progress_cb("motion_estimation", overall_pct, f"chunk {_dc+1}/{total_chunks}")

            results = model.inference(
                img_ck, boxes_ck,
                img_focal=img_focal, img_center=img_center,
                do_flip=do_flip,
                progress_cb=inference_cb,
            )

            data_out = {
                "init_root_orient": results["pred_rotmat"][None, :, 0],
                "init_hand_pose": results["pred_rotmat"][None, :, 1:],
                "init_trans": results["pred_trans"][None, :, 0],
                "init_betas": results["pred_shape"][None, :],
            }

            init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip:
                init_root[..., 1] *= -1
                init_root[..., 2] *= -1
                init_hand_pose[..., 1] *= -1
                init_hand_pose[..., 2] *= -1
            data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
            data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

            pred_dict = {k: v.tolist() for k, v in data_out.items()}
            cam_dir = os.path.join(seq_folder, "cam_space", str(idx))
            os.makedirs(cam_dir, exist_ok=True)
            pred_path = os.path.join(cam_dir, f"{frame_ck[0]}_{frame_ck[-1]}.json")
            with open(pred_path, "w") as f:
                json.dump(pred_dict, f, indent=1)

            done_chunks += 1
            if progress_cb and total_chunks > 0:
                pct = int(done_chunks / total_chunks * 100)
                progress_cb("motion_estimation", pct, f"chunk {done_chunks}/{total_chunks}")

    joblib.dump(frame_chunks_all, cached_chunks)
    return frame_chunks_all, img_focal


# ---------------------------------------------------------------------------
# Post-processing: cam-space predictions → MediaPipe JSON
# ---------------------------------------------------------------------------

def _postprocess(
    imgfiles: list[str],
    seq_folder: str,
    frame_chunks_all,
    img_focal: float,
    progress_cb: ProgressCB | None = None,
) -> tuple[dict, dict]:
    """Load cam-space JSONs, run MANO, smooth, project, and build MediaPipe JSON.

    Returns (mp_data, overlay_data) where overlay_data contains the 2D joints
    and validity mask needed for rendering the overlay video.
    """
    import cv2
    import torch
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.smoothing import smooth_joints
    from hawor.utils.rotation import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix

    num_frames = len(imgfiles)
    pred_trans = torch.zeros(2, num_frames, 3)
    pred_rot = torch.zeros(2, num_frames, 3)
    pred_hand_pose = torch.zeros(2, num_frames, 45)
    pred_betas = torch.zeros(2, num_frames, 10)
    pred_valid = torch.zeros((2, num_frames))

    for idx in [0, 1]:
        frame_chunks = frame_chunks_all[idx]
        if len(frame_chunks) == 0:
            continue
        for frame_ck in frame_chunks:
            pred_path = os.path.join(
                seq_folder, "cam_space", str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json"
            )
            with open(pred_path, "r") as f:
                pred_dict = json.load(f)
            data_out = {k: torch.tensor(v) for k, v in pred_dict.items()}
            root_aa = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            hand_aa = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            pred_trans[[idx], frame_ck] = data_out["init_trans"]
            pred_rot[[idx], frame_ck] = root_aa[0]
            pred_hand_pose[[idx], frame_ck] = hand_aa[0].flatten(-2)
            pred_betas[[idx], frame_ck] = data_out["init_betas"]
            pred_valid[[idx], frame_ck] = 1

    if progress_cb:
        progress_cb("postprocess", 30, "running MANO forward pass")

    pred_valid_np = (pred_valid > 0).numpy()
    vis_start, vis_end = 0, num_frames

    # Right hand
    r = run_mano(
        pred_trans[1:2, vis_start:vis_end],
        pred_rot[1:2, vis_start:vis_end],
        pred_hand_pose[1:2, vis_start:vis_end],
        betas=pred_betas[1:2, vis_start:vis_end],
    )
    right_joints = r["joints"][0].cpu().float()

    # Left hand
    l = run_mano_left(
        pred_trans[0:1, vis_start:vis_end],
        pred_rot[0:1, vis_start:vis_end],
        pred_hand_pose[0:1, vis_start:vis_end],
        betas=pred_betas[0:1, vis_start:vis_end],
    )
    left_joints = l["joints"][0].cpu().float()

    if progress_cb:
        progress_cb("postprocess", 50, "smoothing joints")

    # Smooth within valid chunks.
    for hand_idx, joints_3d in {1: right_joints, 0: left_joints}.items():
        valid = pred_valid_np[hand_idx]
        i = 0
        while i < joints_3d.shape[0]:
            if valid[vis_start + i]:
                j = i
                while j < joints_3d.shape[0] and valid[vis_start + j]:
                    j += 1
                joints_3d[i:j] = smooth_joints(joints_3d[i:j].clone())
                i = j
            else:
                i += 1

    if progress_cb:
        progress_cb("postprocess", 70, "projecting to 2D")

    # 2D projection.
    sample_img = cv2.imread(imgfiles[0])
    H, W = sample_img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    T = vis_end - vis_start

    right_joints_2d = torch.stack([
        img_focal * right_joints[..., 0] / right_joints[..., 2] + cx,
        img_focal * right_joints[..., 1] / right_joints[..., 2] + cy,
    ], dim=-1).numpy()

    left_joints_2d = torch.stack([
        img_focal * left_joints[..., 0] / left_joints[..., 2] + cx,
        img_focal * left_joints[..., 1] / left_joints[..., 2] + cy,
    ], dim=-1).numpy()

    # Compute 2D-projected wrist orientation axes for overlay rendering.
    # Each wrist gets three axis endpoints (X/Y/Z) projected to pixel coords.
    AXIS_LEN = 0.10  # 10 cm in world units
    wrist_axes_2d = {}  # hand_idx → (T, 3 axes, 2 xy)
    for hand_idx, joints_3d in {1: right_joints, 0: left_joints}.items():
        rot_mat = angle_axis_to_rotation_matrix(
            pred_rot[hand_idx, vis_start:vis_end]
        )  # (T, 3, 3)
        wrist_3d = joints_3d[:, 0, :]  # (T, 3)
        axes_ends = []
        for axis_col in range(3):
            direction = rot_mat[:, :, axis_col]  # (T, 3)
            end_3d = wrist_3d + direction * AXIS_LEN  # (T, 3)
            end_2d_x = img_focal * end_3d[:, 0] / end_3d[:, 2] + cx
            end_2d_y = img_focal * end_3d[:, 1] / end_3d[:, 2] + cy
            axes_ends.append(torch.stack([end_2d_x, end_2d_y], dim=-1))
        wrist_axes_2d[hand_idx] = torch.stack(axes_ends, dim=1).numpy()  # (T, 3, 2)

    if progress_cb:
        progress_cb("postprocess", 90, "building MediaPipe JSON")

    mp_data = _build_mediapipe_json(
        right_joints_2d, left_joints_2d,
        right_joints.numpy(), left_joints.numpy(),
        pred_valid_np, vis_start, T, W, H,
    )
    overlay_data = {
        "right_joints_2d": right_joints_2d,
        "left_joints_2d": left_joints_2d,
        "pred_valid_np": pred_valid_np,
        "vis_start": vis_start,
        "T": T,
        "wrist_axes_2d": wrist_axes_2d,
    }
    return mp_data, overlay_data


# MediaPipe landmark names (21 joints).
_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

_HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]


def _build_mediapipe_json(
    right_joints_2d, left_joints_2d,
    right_joints_3d, left_joints_3d,
    pred_valid, vis_start, T, W, H,
) -> dict:
    frames = []
    for t in range(T):
        frame_idx = vis_start + t
        handedness = []
        hand_landmarks = []
        hand_world_landmarks = []

        if pred_valid[1, frame_idx]:
            handedness.append({"index": len(handedness), "score": 1.0, "category_name": "Right"})
            wrist_z = float(right_joints_3d[t, 0, 2])
            landmarks, world_landmarks = [], []
            for j in range(21):
                landmarks.append({
                    "x": round(float(right_joints_2d[t, j, 0]) / W, 6),
                    "y": round(float(right_joints_2d[t, j, 1]) / H, 6),
                    "z": round(float(right_joints_3d[t, j, 2]) - wrist_z, 6),
                })
                world_landmarks.append({
                    "x": round(float(right_joints_3d[t, j, 0]), 6),
                    "y": round(float(right_joints_3d[t, j, 1]), 6),
                    "z": round(float(right_joints_3d[t, j, 2]), 6),
                })
            hand_landmarks.append(landmarks)
            hand_world_landmarks.append(world_landmarks)

        if pred_valid[0, frame_idx]:
            handedness.append({"index": len(handedness), "score": 1.0, "category_name": "Left"})
            wrist_z = float(left_joints_3d[t, 0, 2])
            landmarks, world_landmarks = [], []
            for j in range(21):
                landmarks.append({
                    "x": round(float(left_joints_2d[t, j, 0]) / W, 6),
                    "y": round(float(left_joints_2d[t, j, 1]) / H, 6),
                    "z": round(float(left_joints_3d[t, j, 2]) - wrist_z, 6),
                })
                world_landmarks.append({
                    "x": round(float(left_joints_3d[t, j, 0]), 6),
                    "y": round(float(left_joints_3d[t, j, 1]), 6),
                    "z": round(float(left_joints_3d[t, j, 2]), 6),
                })
            hand_landmarks.append(landmarks)
            hand_world_landmarks.append(world_landmarks)

        frames.append({
            "frame_index": t,
            "handedness": handedness,
            "hand_landmarks": hand_landmarks,
            "hand_world_landmarks": hand_world_landmarks,
        })

    return {
        "image_width": W,
        "image_height": H,
        "frame_count": T,
        "landmark_names": _LANDMARK_NAMES,
        "hand_connections": _HAND_CONNECTIONS,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Overlay video rendering
# ---------------------------------------------------------------------------

def _render_overlay(
    imgfiles: list[str],
    overlay_data: dict,
    output_mp4: str,
    progress_cb: ProgressCB | None = None,
) -> None:
    """Render hand skeleton overlay on extracted frames and write to MP4."""
    import cv2
    import imageio

    right_joints_2d = overlay_data["right_joints_2d"]
    left_joints_2d = overlay_data["left_joints_2d"]
    pred_valid_np = overlay_data["pred_valid_np"]
    vis_start = overlay_data["vis_start"]
    T = overlay_data["T"]
    wrist_axes_2d = overlay_data.get("wrist_axes_2d", {})

    image_names = imgfiles[vis_start : vis_start + T]

    os.makedirs(os.path.dirname(output_mp4) or ".", exist_ok=True)
    writer = imageio.get_writer(output_mp4, fps=30, format="FFMPEG", macro_block_size=None)
    for t in range(T):
        img = cv2.imread(image_names[t])
        frame_idx = vis_start + t
        if pred_valid_np[1, frame_idx]:  # right hand
            img = _draw_hand_skeleton(img, right_joints_2d[t], is_right=1)
            img = _draw_hand_bbox(img, right_joints_2d[t], is_right=1)
            if 1 in wrist_axes_2d:
                img = _draw_wrist_axes(img, right_joints_2d[t, 0], wrist_axes_2d[1][t])
        if pred_valid_np[0, frame_idx]:  # left hand
            img = _draw_hand_skeleton(img, left_joints_2d[t], is_right=0)
            img = _draw_hand_bbox(img, left_joints_2d[t], is_right=0)
            if 0 in wrist_axes_2d:
                img = _draw_wrist_axes(img, left_joints_2d[t, 0], wrist_axes_2d[0][t])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(img_rgb)
        if progress_cb and T > 0 and t % max(1, T // 20) == 0:
            progress_cb("overlay", int(t / T * 100), f"frame {t}/{T}")
    writer.close()


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def process_hand_pose(
    input_video: str,
    output_json: str,
    *,
    output_mp4: str | None = None,
    img_focal: float = 600.0,
    checkpoint: str | None = None,
    progress_cb: ProgressCB | None = None,
) -> tuple[str, str | None]:
    """Run full HaWoR hand-pose pipeline on a single video file.

    Stages:
      1. Extract frames (ffmpeg → PNG at 30 fps)
      2. YOLO hand detection + tracking
      3. HaWoR motion estimation (per-hand-track neural inference)
      4. MANO forward pass, temporal smoothing, 2D projection
      5. Write MediaPipe-compatible JSON output
      6. (Optional) Render overlay video with hand skeletons

    Args:
        input_video: Path to input video file (.mp4/.avi/.mov/.mkv/.webm).
        output_json: Path where the output JSON will be written.
        output_mp4: Path where the overlay video will be written. None to skip.
        img_focal: Camera focal length in pixels. Defaults to 600.
        checkpoint: Path to HaWoR checkpoint. None uses bundled default.
        progress_cb: Optional callback(phase, percent, detail).

    Returns:
        Tuple of (output_json_path, output_mp4_path_or_None).

    Raises:
        RuntimeError: If processing fails at any stage.
        FileNotFoundError: If input video does not exist.
    """
    from glob import glob
    from natsort import natsorted

    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    video_dir = os.path.dirname(input_video)
    video_stem = Path(input_video).stem
    seq_folder = os.path.join(video_dir, video_stem)
    img_folder = os.path.join(seq_folder, "extracted_images")

    # Stage 1: Extract frames.
    if progress_cb:
        progress_cb("frame_extraction", 0, "extracting frames from video")
    imgfiles = natsorted(glob(os.path.join(img_folder, "*.png")))
    if not imgfiles:
        frame_cb = (lambda pct: progress_cb("frame_extraction", pct, f"extracting frames ({pct}%)")) if progress_cb else None
        _extract_frames(input_video, img_folder, progress_cb=frame_cb)
        imgfiles = natsorted(glob(os.path.join(img_folder, "*.png")))
    if not imgfiles:
        raise RuntimeError(f"Frame extraction produced no images from {input_video}")
    if progress_cb:
        progress_cb("frame_extraction", 100, f"extracted {len(imgfiles)} frames")

    # Stage 2: Detection + tracking.
    if progress_cb:
        progress_cb("detection", 0, "running hand detection and tracking")
    detect_cb = (lambda pct: progress_cb("detection", pct, f"detecting hands ({pct}%)")) if progress_cb else None
    start_idx, end_idx, _tracks = _detect_and_track(imgfiles, seq_folder, progress_cb=detect_cb)
    if progress_cb:
        progress_cb("detection", 100, "detection complete")

    # Stage 3: Motion estimation.
    if progress_cb:
        progress_cb("motion_estimation", 0, "starting HaWoR inference")
    frame_chunks_all, focal = _run_motion_estimation(
        imgfiles, seq_folder, start_idx, end_idx,
        img_focal=img_focal,
        checkpoint=checkpoint,
        progress_cb=progress_cb,
    )
    if progress_cb:
        progress_cb("motion_estimation", 100, "inference complete")

    # Stage 4: Post-processing → MediaPipe JSON + overlay data.
    if progress_cb:
        progress_cb("postprocess", 0, "post-processing predictions")
    mp_data, overlay_data = _postprocess(imgfiles, seq_folder, frame_chunks_all, focal, progress_cb)
    if progress_cb:
        progress_cb("postprocess", 100, "post-processing complete")

    # Stage 5: Write JSON output.
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(mp_data, f)

    # Stage 6: Render overlay video (optional).
    if output_mp4:
        if progress_cb:
            progress_cb("overlay", 0, "rendering overlay video")
        _render_overlay(imgfiles, overlay_data, output_mp4, progress_cb)
        if progress_cb:
            progress_cb("overlay", 100, "overlay video complete")

    return output_json, output_mp4
