"""Render a hand-skeleton overlay video from a HaWoR JSON + the original video.

Local, no GPU/model needed. Draws hand_landmarks (2D) onto each frame.

Usage:
  python tools/render_overlay.py <input_video> <handpose.json> <out_video.mp4>
"""
import sys
import json
import cv2
import numpy as np

COLORS = {"Left": (255, 160, 0), "Right": (0, 200, 0)}  # BGR


def main(video_in, json_path, video_out):
    data = json.load(open(json_path))
    W, H = data["image_width"], data["image_height"]
    conns = data.get("hand_connections", [])
    # frame_index -> list of (category, [(x,y),...])
    per_frame = {}
    for fr in data["frames"]:
        items = []
        for slot, hand in enumerate(fr.get("handedness", [])):
            cat = hand.get("category_name", "Right")
            lm = fr["hand_landmarks"][slot]
            pts = [(int(p["x"] * W), int(p["y"] * H)) for p in lm]
            items.append((cat, pts))
        per_frame[fr["frame_index"]] = items

    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (W, H))
    fi = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        for cat, pts in per_frame.get(fi, []):
            col = COLORS.get(cat, (0, 200, 0))
            for a, b in conns:
                cv2.line(img, pts[a], pts[b], col, 2, cv2.LINE_AA)
            for (x, y) in pts:
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, cat, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        writer.write(img)
        fi += 1
    cap.release()
    writer.release()
    print(f"wrote {video_out}  ({fi} frames)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python tools/render_overlay.py <input_video> <handpose.json> <out.mp4>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
