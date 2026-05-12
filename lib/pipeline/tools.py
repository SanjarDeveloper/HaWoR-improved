import os
import cv2
from tqdm import tqdm
import numpy as np
import torch

from ultralytics import YOLO
from lib.pipeline.bbox_cleaning import clean_bbox_sequences

if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def _enlarge_bbox(bbox_xyxy, scale=1.2):
    """Enlarge bbox by *scale* and make it square (matching Dyn-HaMR's enlarge_bbox)."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    side = max(w, h) * scale
    half = side / 2
    return np.array([cx - half, cy - half, cx + half, cy + half])


def _select_egocentric_hand(boxes, confs, img_w, img_h):
    """Select the most likely egocentric hand from multiple same-class detections.

    In egocentric video the wearer's hands are closer to the camera (larger)
    and not at the extreme frame edges.  Other people's hands in the periphery
    are smaller and near the borders.

    Returns the index into *boxes* of the best candidate, or None.
    """
    if len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return 0

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    img_area = img_w * img_h

    # Reject tiny detections (< 0.3 % of image) — likely distant person's hand.
    MIN_AREA_RATIO = 0.003
    valid = areas > (img_area * MIN_AREA_RATIO)

    # Reject detections whose centre falls in the outermost 2 % margin.
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    MARGIN = 0.02
    in_bounds = (
        (cx > img_w * MARGIN) & (cx < img_w * (1 - MARGIN)) &
        (cy > img_h * MARGIN) & (cy < img_h * (1 - MARGIN))
    )
    valid = valid & in_bounds

    if not valid.any():
        # All filtered out — fall back to the largest detection.
        return int(np.argmax(areas))

    # Among valid candidates pick the largest (closest to ego camera).
    valid_idx = np.where(valid)[0]
    return int(valid_idx[np.argmax(areas[valid_idx])])


def detect_track(imgfiles, thresh=0.2, progress_cb=None):

    hand_det_model = YOLO('./weights/external/detector.pt')

    # Get image dimensions from first frame.
    first_frame = cv2.imread(imgfiles[0])
    img_h, img_w = first_frame.shape[:2]
    del first_frame

    # ------------------------------------------------------------------
    # Pass 1: Batched YOLO detection, keep best per class per frame
    # ------------------------------------------------------------------
    raw_data = [None] * len(imgfiles)
    BATCH_SIZE = 16
    total_batches = (len(imgfiles) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, batch_start in enumerate(tqdm(range(0, len(imgfiles), BATCH_SIZE), desc="Detecting")):
        batch_end = min(batch_start + BATCH_SIZE, len(imgfiles))
        batch_paths = imgfiles[batch_start:batch_end]
        batch_imgs = [cv2.imread(p) for p in batch_paths]

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                results_list = hand_det_model.predict(batch_imgs, conf=thresh, verbose=False)

        for local_idx, results in enumerate(results_list):
            t = batch_start + local_idx
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            frame_dict = {
                'frame_idx': t,
                'img_path': imgfiles[t],
                'left_bbox': None,
                'right_bbox': None,
                'left_keypoints': None,
                'right_keypoints': None,
                'left_conf': 0.0,
                'right_conf': 0.0,
            }

            # Use egocentric filtering to pick the best hand per class
            for cls_id, hand_name in [(0, 'left'), (1, 'right')]:
                cls_mask = classes.astype(int) == cls_id
                if not cls_mask.any():
                    continue

                cls_boxes = boxes[cls_mask]
                cls_confs = confs[cls_mask]

                best = _select_egocentric_hand(cls_boxes, cls_confs, img_w, img_h)
                if best is None:
                    continue

                bbox_raw = cls_boxes[best]
                conf = float(cls_confs[best])
                bbox = _enlarge_bbox(bbox_raw, scale=1.2)

                if conf > frame_dict[f'{hand_name}_conf']:
                    frame_dict[f'{hand_name}_bbox'] = bbox
                    frame_dict[f'{hand_name}_conf'] = conf

            raw_data[t] = frame_dict

        if progress_cb and total_batches > 0:
            progress_cb(int((batch_idx + 1) / total_batches * 100))

    # ------------------------------------------------------------------
    # Pass 2: Heuristic cleaning (Dyn-HaMR pipeline)
    # ------------------------------------------------------------------
    raw_data = clean_bbox_sequences(raw_data)

    # ------------------------------------------------------------------
    # Convert cleaned raw_data -> HaWoR tracks dict format
    #   Track ID 0 = left hand, Track ID 1 = right hand
    #   Each entry: {frame, det, det_box (1,5), det_handedness (1,)}
    # ------------------------------------------------------------------
    tracks = {}

    for frame_data in raw_data:
        t = frame_data['frame_idx']

        for hand_id, hand_name in [(0, 'left'), (1, 'right')]:
            bbox = frame_data[f'{hand_name}_bbox']
            if bbox is None:
                continue

            conf = frame_data[f'{hand_name}_conf']
            # det_box shape (1, 5): [x1, y1, x2, y2, conf]
            det_box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3], conf]])
            # det_handedness shape (1,): class id (0=left, 1=right)
            det_handedness = np.array([float(hand_id)])

            subj = {
                'frame': t,
                'det': True,
                'det_box': det_box,
                'det_handedness': det_handedness,
            }

            if hand_id in tracks:
                tracks[hand_id].append(subj)
            else:
                tracks[hand_id] = [subj]

    tracks = np.array(tracks, dtype=object)

    return tracks


def parse_chunks(frame, boxes, min_len=16):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk)>=min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks

def parse_chunks_hand_frame(frame):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        start = bk
        if len(f_chunk) > 0:
            frame_chunks.append(f_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            if len(f_chunk) > 0:
                frame_chunks.append(f_chunk)

    return frame_chunks
