import argparse
import sys
import os
os.environ["DISPLAY"] = ":0"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import json
import torch
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import imageio
from tqdm import tqdm
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.smoothing import smooth_joints, one_euro_smooth_joints, gaussian_smooth_joints
from hawor.utils.rotation import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix


# Hand skeleton connections for 21 joints
HAND_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],        # thumb
    [0, 5], [5, 6], [6, 7], [7, 8],        # index
    [0, 9], [9, 10], [10, 11], [11, 12],   # middle
    [0, 13], [13, 14], [14, 15], [15, 16], # ring
    [0, 17], [17, 18], [18, 19], [19, 20], # pinky
]

# Per-joint BGR colors following the OpenPose convention.
# Each finger has a distinct hue with a dark-to-bright gradient (base to tip).
#   Thumb=red, Index=yellow, Middle=teal, Ring=blue, Pinky=magenta
JOINT_COLORS_BGR = [
    (100, 100, 100),                                        # 0: wrist
    (0, 0, 100), (0, 0, 150), (0, 0, 200), (0, 0, 255),   # 1-4: thumb (red)
    (0, 100, 100), (0, 150, 150), (0, 200, 200), (0, 255, 255),  # 5-8: index (yellow)
    (50, 100, 0), (75, 150, 0), (100, 200, 0), (125, 255, 0),    # 9-12: middle (teal)
    (100, 50, 0), (150, 75, 0), (200, 100, 0), (255, 125, 0),    # 13-16: ring (blue)
    (100, 0, 100), (150, 0, 150), (200, 0, 200), (255, 0, 255),  # 17-20: pinky (magenta)
]

# Bone color = color of the destination (child) joint
BONE_COLORS_BGR = [JOINT_COLORS_BGR[j2] for (_, j2) in HAND_SKELETON]

# Bounding box colors (BGR): left=blue, right=green
BBOX_COLOR = {0: (255, 128, 0), 1: (0, 200, 0)}
BBOX_LABEL = {0: "Left", 1: "Right"}

# MediaPipe-compatible 21 hand landmark names
MEDIAPIPE_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# MediaPipe HAND_CONNECTIONS
MEDIAPIPE_HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]


def build_mediapipe_json(right_joints_2d, left_joints_2d, right_joints_3d, left_joints_3d,
                         pred_valid, vis_start, T, W, H):
    """Build a MediaPipe-compatible JSON dict with per-frame hand landmarks.

    Structure mirrors MediaPipe HandLandmarkerResult:
      - hand_landmarks: normalized (x, y in [0,1], z = depth relative to wrist)
      - hand_world_landmarks: 3D camera-space coordinates in meters
      - handedness: Left / Right classification
    """
    frames = []
    for t in range(T):
        frame_idx = vis_start + t
        handedness = []
        hand_landmarks = []
        hand_world_landmarks = []

        # Right hand
        if pred_valid[1, frame_idx]:
            handedness.append({"index": len(handedness), "score": 1.0, "category_name": "Right"})
            wrist_z = float(right_joints_3d[t, 0, 2])
            landmarks = []
            world_landmarks = []
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

        # Left hand
        if pred_valid[0, frame_idx]:
            handedness.append({"index": len(handedness), "score": 1.0, "category_name": "Left"})
            wrist_z = float(left_joints_3d[t, 0, 2])
            landmarks = []
            world_landmarks = []
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
        "landmark_names": MEDIAPIPE_LANDMARK_NAMES,
        "hand_connections": MEDIAPIPE_HAND_CONNECTIONS,
        "frames": frames,
    }


def is_hand_in_frame(keypoints_2d, img_shape, margin=100):
    """Check if hand keypoints are reasonably within frame bounds."""
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


def draw_hand_bbox(img, keypoints_2d, is_right=1, padding=15, thickness=2):
    """Draw a bounding box with a Left/Right label around the hand."""
    H, W = img.shape[:2]
    if not is_hand_in_frame(keypoints_2d, img.shape):
        return img

    # Collect visible keypoints
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

    # Draw label with filled background above the box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    # Place label above the box; if too close to the top, place below
    label_y = y1 - 6
    if label_y - th < 0:
        label_y = y2 + th + 6
    cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 4), color, -1)
    cv2.putText(img, label, (x1 + 2, label_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_hand_skeleton(img, keypoints_2d, is_right=1, radius=4, thickness=2):
    """Draw 21-joint hand skeleton on image (BGR)."""
    H, W = img.shape[:2]
    if not is_hand_in_frame(keypoints_2d, img.shape):
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


def draw_wrist_axes(img, wrist_2d, axes_ends_2d, thickness=2):
    """Draw three perpendicular orientation axes (X=red, Y=green, Z=blue) at the wrist."""
    H, W = img.shape[:2]
    ox, oy = int(wrist_2d[0]), int(wrist_2d[1])
    if not (0 <= ox < W and 0 <= oy < H):
        return img
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X=red, Y=green, Z=blue
    for axis_idx in range(3):
        ex, ey = int(axes_ends_2d[axis_idx, 0]), int(axes_ends_2d[axis_idx, 1])
        cv2.line(img, (ox, oy), (ex, ey), colors[axis_idx], thickness)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_folder", type=str, default='example/', help='folder containing video files')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    args = parser.parse_args()

    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = sorted([
        os.path.join(args.video_folder, f)
        for f in os.listdir(args.video_folder)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ])
    if not video_files:
        print(f"No video files found in {args.video_folder}")
        sys.exit(1)
    print(f"Found {len(video_files)} video(s) in {args.video_folder}")

    for vi, video_path in enumerate(video_files):
        print(f"\n{'='*60}")
        print(f"[{vi+1}/{len(video_files)}] Processing: {video_path}")
        print(f"{'='*60}")

        # Set video_path on args so downstream functions can use it
        args.video_path = video_path

        try:
            start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

            frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

            # Load camera-space predictions directly from JSON files
            num_frames = len(imgfiles)
            pred_trans = torch.zeros(2, num_frames, 3)
            pred_rot = torch.zeros(2, num_frames, 3)
            pred_hand_pose = torch.zeros(2, num_frames, 45)
            pred_betas = torch.zeros(2, num_frames, 10)
            pred_valid = torch.zeros((2, num_frames))

            for idx in [0, 1]:  # 0=left, 1=right
                frame_chunks = frame_chunks_all[idx]
                if len(frame_chunks) == 0:
                    continue
                for frame_ck in frame_chunks:
                    pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
                    with open(pred_path, "r") as f:
                        pred_dict = json.load(f)
                    data_out = {k: torch.tensor(v) for k, v in pred_dict.items()}
                    # Convert rotation matrices to angle axis
                    root_aa = rotation_matrix_to_angle_axis(data_out["init_root_orient"])  # (1, T, 3)
                    hand_aa = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])    # (1, T, 15, 3)
                    pred_trans[[idx], frame_ck] = data_out["init_trans"]
                    pred_rot[[idx], frame_ck] = root_aa[0]
                    pred_hand_pose[[idx], frame_ck] = hand_aa[0].flatten(-2)
                    pred_betas[[idx], frame_ck] = data_out["init_betas"]
                    pred_valid[[idx], frame_ck] = 1

            pred_valid = (pred_valid > 0).numpy()

            # Compute joints and project to 2D keypoints
            hand2idx = {"right": 1, "left": 0}
            vis_start = 0
            vis_end = pred_trans.shape[1]

            # Right hand joints (camera space)
            hand_idx = hand2idx["right"]
            pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
            right_joints = pred_glob_r['joints'][0].cpu().float()  # (T, 21, 3)

            # Left hand joints (camera space)
            hand_idx = hand2idx["left"]
            pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
            left_joints = pred_glob_l['joints'][0].cpu().float()  # (T, 21, 3)

            # Smooth 3D joints within contiguous valid chunks to avoid
            # zero-parameter frames bleeding into valid neighbors.
            all_joints = {1: right_joints, 0: left_joints}
            for hand_idx, joints_3d in all_joints.items():
                valid = pred_valid[hand_idx]
                i = 0
                while i < joints_3d.shape[0]:
                    if valid[vis_start + i]:
                        j = i
                        while j < joints_3d.shape[0] and valid[vis_start + j]:
                            j += 1
                        joints_3d[i:j] = gaussian_smooth_joints(joints_3d[i:j].clone())
                        i = j
                    else:
                        i += 1
            right_joints = all_joints[1]
            left_joints = all_joints[0]

            # Project 3D joints to 2D (already in camera space, use pinhole projection)
            image_names = imgfiles[vis_start:vis_end]
            T = len(image_names)
            sample_img = cv2.imread(image_names[0])
            H, W = sample_img.shape[:2]
            cx, cy = W / 2.0, H / 2.0

            right_joints_2d = torch.stack([
                img_focal * right_joints[..., 0] / right_joints[..., 2] + cx,
                img_focal * right_joints[..., 1] / right_joints[..., 2] + cy,
            ], dim=-1).numpy()  # (T, 21, 2)

            left_joints_2d = torch.stack([
                img_focal * left_joints[..., 0] / left_joints[..., 2] + cx,
                img_focal * left_joints[..., 1] / left_joints[..., 2] + cy,
            ], dim=-1).numpy()  # (T, 21, 2)

            # Compute 2D-projected wrist orientation axes.
            AXIS_LEN = 0.10  # 10 cm
            wrist_axes_2d = {}
            for hand_idx, joints_3d in {1: right_joints, 0: left_joints}.items():
                rot_mat = angle_axis_to_rotation_matrix(
                    pred_rot[hand_idx, vis_start:vis_end]
                )  # (T, 3, 3)
                wrist_3d = joints_3d[:, 0, :]  # (T, 3)
                axes_ends = []
                for axis_col in range(3):
                    direction = rot_mat[:, :, axis_col]  # (T, 3)
                    end_3d = wrist_3d + direction * AXIS_LEN
                    end_2d_x = img_focal * end_3d[:, 0] / end_3d[:, 2] + cx
                    end_2d_y = img_focal * end_3d[:, 1] / end_3d[:, 2] + cy
                    axes_ends.append(torch.stack([end_2d_x, end_2d_y], dim=-1))
                wrist_axes_2d[hand_idx] = torch.stack(axes_ends, dim=1).numpy()  # (T, 3, 2)

            # Render keypoints overlay video
            file_name = video_path.split('/')[-1].split('.')[0]
            output_video = os.path.join(seq_folder, f"{file_name}.mp4")
            print(f"Rendering keypoints for frames {vis_start} to {vis_end}")

            writer = imageio.get_writer(output_video, fps=30, format='FFMPEG', macro_block_size=None)
            for t in tqdm(range(T), desc="Overlay"):
                img = cv2.imread(image_names[t])
                frame_idx = vis_start + t
                if pred_valid[1, frame_idx]:  # right hand was detected
                    img = draw_hand_skeleton(img, right_joints_2d[t], is_right=1)
                    img = draw_hand_bbox(img, right_joints_2d[t], is_right=1)
                    img = draw_wrist_axes(img, right_joints_2d[t, 0], wrist_axes_2d[1][t])
                if pred_valid[0, frame_idx]:  # left hand was detected
                    img = draw_hand_skeleton(img, left_joints_2d[t], is_right=0)
                    img = draw_hand_bbox(img, left_joints_2d[t], is_right=0)
                    img = draw_wrist_axes(img, left_joints_2d[t, 0], wrist_axes_2d[0][t])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                writer.append_data(img_rgb)
            writer.close()

            print(f"Keypoint video saved to: {output_video}")

            # Save MediaPipe-compatible JSON
            output_json = os.path.join(seq_folder, f"{file_name}.json")
            mp_data = build_mediapipe_json(
                right_joints_2d, left_joints_2d,
                right_joints.numpy(), left_joints.numpy(),
                pred_valid, vis_start, T, W, H,
            )
            with open(output_json, "w") as f:
                json.dump(mp_data, f)
            print(f"MediaPipe JSON saved to: {output_json}")
            print(f"Finished: {video_path}")

        except Exception as e:
            print(f"ERROR processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("All videos finished.")
