"""
Dyn-HaMR-style heuristic bbox cleaning for hand detection sequences.

Ported from Dyn-HaMR's third-party/hamer/run.py, with visualization code
and keypoint-specific logic stripped out (YOLO doesn't provide keypoints).

The pipeline operates on a list of per-frame dicts (raw_data) with fields:
  frame_idx, img_path, left_bbox, right_bbox, left_conf, right_conf,
  left_keypoints, right_keypoints
"""

import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def compute_containment_ratio(bbox1, bbox2):
    """
    Compute how much bbox1 is contained within bbox2.
    Returns the ratio of bbox1's area that overlaps with bbox2.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)

    return inter_area / bbox1_area if bbox1_area > 0 else 0


# ---------------------------------------------------------------------------
# Step 1: Overlap removal
# ---------------------------------------------------------------------------

def detect_overlapping_bboxes(raw_data, iou_threshold=0.7, containment_threshold=0.7):
    """
    Detect and remove overlapping bboxes (hallucinations).
    Uses bbox IoU and containment ratio.  Keeps the hand with higher confidence.
    Tracks which hand was removed so it can be restored later if the winner
    is invalidated.
    """
    print("\n" + "-" * 80)
    print("Step 1: Detecting overlapping bboxes (hallucinations)")
    print("-" * 80)

    overlap_count = 0

    for frame_data in raw_data:
        frame_data['left_removed_due_to_overlap_with'] = None
        frame_data['right_removed_due_to_overlap_with'] = None

        if frame_data['left_bbox'] is not None and frame_data['right_bbox'] is not None:
            iou = compute_iou(frame_data['left_bbox'], frame_data['right_bbox'])
            left_in_right = compute_containment_ratio(frame_data['left_bbox'], frame_data['right_bbox'])
            right_in_left = compute_containment_ratio(frame_data['right_bbox'], frame_data['left_bbox'])
            max_containment = max(left_in_right, right_in_left)

            is_overlap = False
            reason = []

            if iou > iou_threshold:
                is_overlap = True
                reason.append(f"IoU={iou:.3f}")

            if max_containment > containment_threshold:
                is_overlap = True
                which_contained = "L_in_R" if left_in_right > right_in_left else "R_in_L"
                reason.append(f"containment={max_containment:.2f} ({which_contained})")

            if is_overlap:
                overlap_count += 1
                print(f"  Frame {frame_data['frame_idx']:04d}: {', '.join(reason)} - OVERLAP DETECTED")

                if frame_data['left_conf'] > frame_data['right_conf']:
                    print(f"    -> Keeping LEFT (conf={frame_data['left_conf']:.3f}), removing RIGHT")
                    frame_data['right_bbox'] = None
                    frame_data['right_keypoints'] = None
                    frame_data['right_conf'] = 0.0
                    frame_data['right_removed_due_to_overlap_with'] = 'left'
                else:
                    print(f"    -> Keeping RIGHT (conf={frame_data['right_conf']:.3f}), removing LEFT")
                    frame_data['left_bbox'] = None
                    frame_data['left_keypoints'] = None
                    frame_data['left_conf'] = 0.0
                    frame_data['left_removed_due_to_overlap_with'] = 'right'

    print(f"Removed overlapping bboxes in {overlap_count} frames")
    return raw_data


# ---------------------------------------------------------------------------
# Handedness swap corrections
# ---------------------------------------------------------------------------

def fix_handedness_swaps_by_trajectory(raw_data, position_threshold=200):
    """
    Detect handedness swaps by analysing trajectory jumps.
    If one hand suddenly jumps to where the other hand was, it's likely mislabeled.
    """
    print("\n" + "-" * 80)
    print(f"Step 1.4a: Detecting handedness swaps via trajectory analysis")
    print("-" * 80)

    swap_count = 0
    n = len(raw_data)

    for i in range(1, n):
        left_bbox = raw_data[i]['left_bbox']
        right_bbox = raw_data[i]['right_bbox']
        prev_left = raw_data[i - 1]['left_bbox']
        prev_right = raw_data[i - 1]['right_bbox']

        # Case 1: Right hand jumps to where left hand was (left disappears)
        if right_bbox is not None and left_bbox is None and prev_left is not None and prev_right is not None:
            prev_left_cx = (prev_left[0] + prev_left[2]) / 2
            prev_right_cx = (prev_right[0] + prev_right[2]) / 2
            curr_right_cx = (right_bbox[0] + right_bbox[2]) / 2

            dist_to_prev_left = abs(curr_right_cx - prev_left_cx)
            dist_to_prev_right = abs(curr_right_cx - prev_right_cx)

            if dist_to_prev_left < dist_to_prev_right and dist_to_prev_right > position_threshold:
                print(f"  Frame {i:04d}: RIGHT jumped to LEFT position "
                      f"(dist to prev_left={dist_to_prev_left:.0f}, dist to prev_right={dist_to_prev_right:.0f}) "
                      f"- Swapping RIGHT -> LEFT")
                raw_data[i]['left_bbox'] = right_bbox
                raw_data[i]['left_keypoints'] = raw_data[i]['right_keypoints']
                raw_data[i]['left_conf'] = raw_data[i]['right_conf']
                raw_data[i]['right_bbox'] = None
                raw_data[i]['right_keypoints'] = None
                raw_data[i]['right_conf'] = 0.0
                swap_count += 1

        # Case 2: Left hand jumps to where right hand was (right disappears)
        elif left_bbox is not None and right_bbox is None and prev_right is not None and prev_left is not None:
            prev_left_cx = (prev_left[0] + prev_left[2]) / 2
            prev_right_cx = (prev_right[0] + prev_right[2]) / 2
            curr_left_cx = (left_bbox[0] + left_bbox[2]) / 2

            dist_to_prev_left = abs(curr_left_cx - prev_left_cx)
            dist_to_prev_right = abs(curr_left_cx - prev_right_cx)

            if dist_to_prev_right < dist_to_prev_left and dist_to_prev_left > position_threshold:
                print(f"  Frame {i:04d}: LEFT jumped to RIGHT position "
                      f"(dist to prev_right={dist_to_prev_right:.0f}, dist to prev_left={dist_to_prev_left:.0f}) "
                      f"- Swapping LEFT -> RIGHT")
                raw_data[i]['right_bbox'] = left_bbox
                raw_data[i]['right_keypoints'] = raw_data[i]['left_keypoints']
                raw_data[i]['right_conf'] = raw_data[i]['left_conf']
                raw_data[i]['left_bbox'] = None
                raw_data[i]['left_keypoints'] = None
                raw_data[i]['left_conf'] = 0.0
                swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} trajectory-based handedness swaps")
    else:
        print(f"  No trajectory-based swaps detected")

    return raw_data


def fix_handedness_swaps(raw_data, context_window=10):
    """
    Detect and fix handedness swaps where YOLO misclassifies hand labels.
    If a hand appears/disappears while the other hand simultaneously
    disappears/appears in a short sequence, it's likely a handedness swap.
    """
    print("\n" + "-" * 80)
    print(f"Step 1.4: Detecting handedness swaps (context window={context_window})")
    print("-" * 80)

    swap_count = 0
    n = len(raw_data)

    for i in range(n):
        left_bbox = raw_data[i]['left_bbox']
        right_bbox = raw_data[i]['right_bbox']

        # Case 1: Only right hand detected, but neighbours mostly have only left hand
        if right_bbox is not None and left_bbox is None:
            left_count = 0
            right_count = 0

            for j in range(max(0, i - context_window), min(n, i + context_window + 1)):
                if j == i:
                    continue
                if raw_data[j]['left_bbox'] is not None and raw_data[j]['right_bbox'] is None:
                    left_count += 1
                elif raw_data[j]['right_bbox'] is not None and raw_data[j]['left_bbox'] is None:
                    right_count += 1

            if left_count > 0 and left_count > right_count * 2:
                print(f"  Frame {i:04d}: Swapping RIGHT -> LEFT "
                      f"(neighbours: {left_count} left-only, {right_count} right-only)")
                raw_data[i]['left_bbox'] = right_bbox
                raw_data[i]['left_keypoints'] = raw_data[i]['right_keypoints']
                raw_data[i]['left_conf'] = raw_data[i]['right_conf']
                raw_data[i]['right_bbox'] = None
                raw_data[i]['right_keypoints'] = None
                raw_data[i]['right_conf'] = 0.0
                swap_count += 1

        # Case 2: Only left hand detected, but neighbours mostly have only right hand
        elif left_bbox is not None and right_bbox is None:
            left_count = 0
            right_count = 0

            for j in range(max(0, i - context_window), min(n, i + context_window + 1)):
                if j == i:
                    continue
                if raw_data[j]['left_bbox'] is not None and raw_data[j]['right_bbox'] is None:
                    left_count += 1
                elif raw_data[j]['right_bbox'] is not None and raw_data[j]['left_bbox'] is None:
                    right_count += 1

            if right_count > 0 and right_count > left_count * 2:
                print(f"  Frame {i:04d}: Swapping LEFT -> RIGHT "
                      f"(neighbours: {left_count} left-only, {right_count} right-only)")
                raw_data[i]['right_bbox'] = left_bbox
                raw_data[i]['right_keypoints'] = raw_data[i]['left_keypoints']
                raw_data[i]['right_conf'] = raw_data[i]['left_conf']
                raw_data[i]['left_bbox'] = None
                raw_data[i]['left_keypoints'] = None
                raw_data[i]['left_conf'] = 0.0
                swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} handedness swaps")
    else:
        print(f"  No handedness swaps detected")

    return raw_data


def fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.7, max_gap=10):
    """
    Detect frame-to-frame handedness swaps: same bbox position but different label.
    If a hand suddenly appears while the other hand recently disappeared
    (within max_gap frames), AND the new hand's bbox has high IoU with the
    disappeared hand's last valid bbox, swap it back.
    """
    print("\n" + "-" * 80)
    print(f"Step 1.4: Detecting frame-to-frame handedness swaps "
          f"(IoU threshold={iou_threshold}, max_gap={max_gap})")
    print("-" * 80)

    swap_count = 0

    for i in range(1, len(raw_data)):
        curr = raw_data[i]

        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            other_hand = 'right' if hand_name == 'left' else 'left'
            other_bbox_key = f'{other_hand}_bbox'
            other_keyp_key = f'{other_hand}_keypoints'
            other_conf_key = f'{other_hand}_conf'

            curr_bbox = curr[bbox_key]
            curr_other_bbox = curr[other_bbox_key]

            # Case: Current hand exists and other hand is missing
            if curr_bbox is not None and curr_other_bbox is None:
                last_valid_other_bbox = None
                last_valid_other_idx = None

                for j in range(i - 1, max(0, i - max_gap - 1), -1):
                    if raw_data[j][other_bbox_key] is not None:
                        last_valid_other_bbox = raw_data[j][other_bbox_key]
                        last_valid_other_idx = j
                        break

                if last_valid_other_bbox is not None:
                    gap = i - last_valid_other_idx
                    iou = compute_iou(curr_bbox, last_valid_other_bbox)
                    if iou > iou_threshold:
                        print(f"  Frame {curr['frame_idx']:04d}: {hand_name} at same position as {other_hand} "
                              f"(last seen at frame {raw_data[last_valid_other_idx]['frame_idx']}, "
                              f"gap={gap}, IoU={iou:.3f}) "
                              f"- SWAPPING {hand_name} -> {other_hand}")

                        curr[other_bbox_key] = curr_bbox.copy()
                        curr[other_keyp_key] = curr[keyp_key].copy() if curr[keyp_key] is not None else None
                        curr[other_conf_key] = curr[conf_key]

                        curr[bbox_key] = None
                        curr[keyp_key] = None
                        curr[conf_key] = 0.0

                        swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} frame-to-frame handedness swaps")
    else:
        print(f"  No frame-to-frame swaps detected")

    return raw_data


def fix_handedness_inconsistencies(raw_data, original_data, context_window=5):
    """
    Detect handedness swaps using spatial-temporal consistency.
    For each detection, compare IoU with recent same-hand vs other-hand bboxes;
    if the other-hand trajectory fits much better, swap the label.
    """
    print("\n" + "-" * 80)
    print(f"Step 1.5: Fixing handedness inconsistencies via "
          f"spatial-temporal consistency (window={context_window})")
    print("-" * 80)

    swap_count = 0

    for i, frame_data in enumerate(raw_data):
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            other_hand = 'right' if hand_name == 'left' else 'left'
            other_bbox_key = f'{other_hand}_bbox'
            other_keyp_key = f'{other_hand}_keypoints'
            other_conf_key = f'{other_hand}_conf'

            current_bbox = frame_data[bbox_key]
            if current_bbox is None:
                continue

            same_hand_bboxes = []
            other_hand_bboxes = []

            for j in range(max(0, i - context_window), min(len(raw_data), i + context_window + 1)):
                if j == i:
                    continue
                if raw_data[j][bbox_key] is not None:
                    same_hand_bboxes.append(raw_data[j][bbox_key])
                if raw_data[j][other_bbox_key] is not None:
                    other_hand_bboxes.append(raw_data[j][other_bbox_key])

            if len(same_hand_bboxes) < 2 and len(other_hand_bboxes) < 2:
                continue

            same_hand_ious = [compute_iou(current_bbox, b) for b in same_hand_bboxes]
            avg_iou_same = np.mean(same_hand_ious) if len(same_hand_ious) > 0 else 0.0

            other_hand_ious = [compute_iou(current_bbox, b) for b in other_hand_bboxes]
            avg_iou_other = np.mean(other_hand_ious) if len(other_hand_ious) > 0 else 0.0

            if (len(other_hand_bboxes) >= 3 and
                    avg_iou_other > 0.5 and
                    avg_iou_other > avg_iou_same * 1.5):

                print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                      f"Bbox fits {other_hand} trajectory better! "
                      f"(IoU with {hand_name}={avg_iou_same:.3f}, "
                      f"IoU with {other_hand}={avg_iou_other:.3f}, "
                      f"neighbours: {len(same_hand_bboxes)} {hand_name}, "
                      f"{len(other_hand_bboxes)} {other_hand}) "
                      f"- SWAPPING {hand_name} -> {other_hand}")

                frame_data[other_bbox_key] = current_bbox.copy()
                frame_data[other_keyp_key] = frame_data[keyp_key].copy() if frame_data[keyp_key] is not None else None
                frame_data[other_conf_key] = frame_data[conf_key]

                frame_data[bbox_key] = None
                frame_data[keyp_key] = None
                frame_data[conf_key] = 0.0

                swap_count += 1

    if swap_count > 0:
        print(f"  Fixed {swap_count} handedness swaps via spatial-temporal consistency")
    else:
        print(f"  No spatial-temporal inconsistencies found")

    return raw_data


# ---------------------------------------------------------------------------
# Jump detection (included but commented out in orchestrator, matching Dyn-HaMR)
# ---------------------------------------------------------------------------

def detect_bbox_jumps(raw_data, base_iou_threshold=0.5):
    """
    Detect and fix bbox jumps with adaptive threshold based on detection gaps.
    Adaptive formula: adjusted = max(0, base - gap * 0.01)
    """
    print("\n" + "-" * 80)
    print(f"Step 2: Detecting bbox jumps with adaptive threshold (base={base_iou_threshold})")
    print("-" * 80)

    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'

        last_valid_bbox = None
        last_valid_idx = None
        frames_since_last_detection = 0
        jump_count = 0

        for i, frame_data in enumerate(raw_data):
            current_bbox = frame_data[bbox_key]

            if current_bbox is not None:
                if last_valid_bbox is not None:
                    iou = compute_iou(current_bbox, last_valid_bbox)
                    gap_frames = frames_since_last_detection
                    adjusted_threshold = max(0.0, base_iou_threshold - (gap_frames * 0.01))

                    if iou < adjusted_threshold:
                        jump_count += 1
                        print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                              f"IoU={iou:.3f} < threshold={adjusted_threshold:.3f} "
                              f"(gap={gap_frames}) - JUMP DETECTED")
                        frame_data[bbox_key] = last_valid_bbox.copy()
                    else:
                        if gap_frames > 0:
                            print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                                  f"IoU={iou:.3f} > threshold={adjusted_threshold:.3f} "
                                  f"(gap={gap_frames}) - ACCEPTING movement")
                        last_valid_bbox = current_bbox.copy()
                        last_valid_idx = i
                        frames_since_last_detection = 0
                else:
                    last_valid_bbox = current_bbox.copy()
                    last_valid_idx = i
                    frames_since_last_detection = 0
            else:
                frames_since_last_detection += 1

        if jump_count > 0:
            print(f"  {hand_name.capitalize()}: Fixed {jump_count} jumps")

    return raw_data


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def clean_bbox_sequences(raw_data):
    """
    Pass 2: Clean bbox sequences using global temporal information.
    Matches Dyn-HaMR's cleaning pipeline order exactly.
    """
    print("\n" + "=" * 80)
    print("PASS 2: Cleaning bbox sequences")
    print("=" * 80)

    # Store original data for comparison and restoration
    original_data = [{
        'left_bbox': f['left_bbox'].copy() if f['left_bbox'] is not None else None,
        'right_bbox': f['right_bbox'].copy() if f['right_bbox'] is not None else None,
        'left_keypoints': f['left_keypoints'].copy() if f['left_keypoints'] is not None else None,
        'right_keypoints': f['right_keypoints'].copy() if f['right_keypoints'] is not None else None,
        'left_conf': f['left_conf'] if f['left_conf'] is not None else None,
        'right_conf': f['right_conf'] if f['right_conf'] is not None else None,
    } for f in raw_data]

    # ---- Remove oversized bboxes (>50% image area) ----
    print("\n" + "-" * 80)
    print("Step 1.5: Removing oversized bboxes (> 50% of image area)")
    print("-" * 80)
    MAX_BBOX_AREA_RATIO = 0.5
    removed_oversized = 0

    first_img = cv2.imread(raw_data[0]['img_path'], cv2.IMREAD_GRAYSCALE)
    img_area = first_img.shape[0] * first_img.shape[1]

    for frame_data in raw_data:
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'

            bbox = frame_data[bbox_key]
            if bbox is not None:
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                bbox_area = bbox_width * bbox_height
                area_ratio = bbox_area / img_area

                if area_ratio > MAX_BBOX_AREA_RATIO:
                    print(f"  Frame {frame_data['frame_idx']:04d} ({hand_name}): "
                          f"Bbox too large ({area_ratio:.1%} of image) - REMOVING")
                    frame_data[bbox_key] = None
                    frame_data[keyp_key] = None
                    frame_data[conf_key] = 0.0
                    removed_oversized += 1

    if removed_oversized > 0:
        print(f"  Removed {removed_oversized} oversized bboxes")
    else:
        print(f"  No oversized bboxes found")

    # ---- Cleaning steps (matching Dyn-HaMR order) ----
    raw_data = detect_overlapping_bboxes(raw_data, iou_threshold=0.7)
    raw_data = fix_handedness_swaps_by_trajectory(raw_data, position_threshold=200)
    raw_data = fix_handedness_inconsistencies(raw_data, original_data, context_window=5)
    # raw_data = detect_bbox_jumps(raw_data, base_iou_threshold=0.4)
    raw_data = fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.6)

    # ---- Patience-based gap filling with distance-aware interpolation ----
    print("\n" + "-" * 80)
    print("Step 3: Applying patience mechanism with interpolation")
    print("-" * 80)
    PATIENCE_FRAMES = 25
    MAX_PATIENCE_WITHOUT_RETURN = 0
    patience_applied_count = 0
    interpolated_count = 0

    for hand_id in [0, 1]:
        hand_name = 'left' if hand_id == 0 else 'right'
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'

        i = 0
        while i < len(raw_data):
            if raw_data[i][bbox_key] is None:
                # Look back for last valid bbox
                last_valid_idx = None
                for j in range(i - 1, -1, -1):
                    if raw_data[j][bbox_key] is not None:
                        last_valid_idx = j
                        break

                if last_valid_idx is None:
                    i += 1
                    continue

                gap_start = i
                gap_end = None
                for j in range(i, min(i + PATIENCE_FRAMES, len(raw_data))):
                    if raw_data[j][bbox_key] is not None:
                        gap_end = j
                        break

                if gap_end is not None:
                    gap_length = gap_end - gap_start
                    start_bbox = raw_data[last_valid_idx][bbox_key]
                    end_bbox = raw_data[gap_end][bbox_key]

                    start_cx = (start_bbox[0] + start_bbox[2]) / 2
                    start_cy = (start_bbox[1] + start_bbox[3]) / 2
                    end_cx = (end_bbox[0] + end_bbox[2]) / 2
                    end_cy = (end_bbox[1] + end_bbox[3]) / 2
                    center_distance = np.sqrt((end_cx - start_cx) ** 2 + (end_cy - start_cy) ** 2)

                    start_width = start_bbox[2] - start_bbox[0]
                    end_width = end_bbox[2] - end_bbox[0]
                    avg_width = (start_width + end_width) / 2
                    max_distance = avg_width

                    if center_distance <= max_distance:
                        print(f"  {hand_name.capitalize()}: Interpolating frames {gap_start}-{gap_end - 1} "
                              f"(gap={gap_length}, from frame {last_valid_idx} to {gap_end}, "
                              f"distance={center_distance:.1f}px < {max_distance:.1f}px)")

                        total_steps = gap_end - last_valid_idx
                        for j in range(gap_start, gap_end):
                            alpha = (j - last_valid_idx) / total_steps
                            interp_bbox = (1 - alpha) * start_bbox + alpha * end_bbox
                            raw_data[j][bbox_key] = interp_bbox
                            raw_data[j][keyp_key] = None
                            raw_data[j][conf_key] = 0.5
                            interpolated_count += 1

                        i = gap_end
                    else:
                        print(f"  {hand_name.capitalize()}: Skipping interpolation for frames {gap_start}-{gap_end - 1} "
                              f"(distance={center_distance:.1f}px > {max_distance:.1f}px "
                              f"- likely different hand instances)")
                        i = gap_end
                else:
                    last_bbox = raw_data[last_valid_idx][bbox_key]
                    frames_to_fill = min(MAX_PATIENCE_WITHOUT_RETURN, len(raw_data) - gap_start)

                    if frames_to_fill > 0:
                        print(f"  {hand_name.capitalize()}: Applying static patience for frames "
                              f"{gap_start}-{gap_start + frames_to_fill - 1} "
                              f"(hand never reappeared, using last bbox from frame {last_valid_idx})")

                    for j in range(gap_start, gap_start + frames_to_fill):
                        raw_data[j][bbox_key] = last_bbox.copy()
                        raw_data[j][keyp_key] = None
                        raw_data[j][conf_key] = 0.5
                        patience_applied_count += 1

                    next_valid_idx = None
                    for j in range(gap_start + frames_to_fill, len(raw_data)):
                        if raw_data[j][bbox_key] is not None:
                            next_valid_idx = j
                            break

                    i = next_valid_idx if next_valid_idx is not None else len(raw_data)
            else:
                i += 1

    if interpolated_count > 0:
        print(f"  Interpolated {interpolated_count} frames where hand reappeared within patience threshold")
    if patience_applied_count > 0:
        print(f"  Applied static patience to {patience_applied_count} frames where hand never reappeared")
    if interpolated_count == 0 and patience_applied_count == 0:
        print(f"  No patience needed")

    # ---- Remove spurious short motions ----
    print("\n" + "-" * 80)
    print("Step 4: Removing spurious short motions")
    print("-" * 80)
    MIN_MOTION_DURATION = 30
    MIN_ABSENCE_BEFORE = 30
    MIN_ABSENCE_AFTER = 30

    for hand_name in ['left', 'right']:
        bbox_key = f'{hand_name}_bbox'
        keyp_key = f'{hand_name}_keypoints'
        conf_key = f'{hand_name}_conf'

        segments = []
        start_idx = None

        for i, frame_data in enumerate(raw_data):
            if frame_data[bbox_key] is not None:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    segments.append((start_idx, i - 1))
                    start_idx = None

        if start_idx is not None:
            segments.append((start_idx, len(raw_data) - 1))

        removed_segments = []
        for seg_start, seg_end in segments:
            seg_duration = seg_end - seg_start + 1
            absence_before = seg_start
            absence_after = len(raw_data) - 1 - seg_end

            if (seg_duration < MIN_MOTION_DURATION and
                    absence_before >= MIN_ABSENCE_BEFORE and
                    absence_after >= MIN_ABSENCE_AFTER):

                print(f"  {hand_name.capitalize()}: Removing spurious short motion at frames {seg_start}-{seg_end} "
                      f"(duration={seg_duration}, absence_before={absence_before}, absence_after={absence_after})")

                for i in range(seg_start, seg_end + 1):
                    raw_data[i][bbox_key] = None
                    raw_data[i][keyp_key] = None
                    raw_data[i][conf_key] = 0.0

                removed_segments.append((seg_start, seg_end))

        if len(removed_segments) > 0:
            print(f"  {hand_name.capitalize()}: Removed {len(removed_segments)} spurious short motion segments")
        else:
            print(f"  {hand_name.capitalize()}: No spurious short motions found")

    # ---- Re-check overlaps after interpolation ----
    print("\n" + "-" * 80)
    print("Step 4: Re-checking overlaps after interpolation")
    print("-" * 80)
    raw_data = detect_overlapping_bboxes(raw_data, iou_threshold=0.7)

    # ---- Restore overlap-removed hands if the winner was invalidated ----
    print("\n" + "-" * 80)
    print("Step 5: Restoring hands removed by overlap if winner was invalidated")
    print("-" * 80)
    restored_count = 0
    for i, (frame_data, orig) in enumerate(zip(raw_data, original_data)):
        for hand_name in ['left', 'right']:
            bbox_key = f'{hand_name}_bbox'
            keyp_key = f'{hand_name}_keypoints'
            conf_key = f'{hand_name}_conf'
            removal_flag = f'{hand_name}_removed_due_to_overlap_with'

            if frame_data.get(removal_flag) is not None:
                other_hand = frame_data[removal_flag]
                other_bbox_key = f'{other_hand}_bbox'
                other_handedness_flag = f'{other_hand}_removed_by_handedness_check'

                if frame_data.get(other_handedness_flag, False):
                    print(f"  Frame {i}: Restoring {hand_name} hand "
                          f"(winner '{other_hand}' was invalidated)")
                    if orig[bbox_key] is not None:
                        frame_data[bbox_key] = orig[bbox_key].copy()
                        frame_data[keyp_key] = orig[keyp_key].copy() if orig[keyp_key] is not None else None
                        frame_data[conf_key] = orig[conf_key]
                    frame_data[removal_flag] = None
                    restored_count += 1

    # ---- Final frame-to-frame swap check ----
    raw_data = fix_handedness_swaps_frame_to_frame(raw_data, iou_threshold=0.6)

    if restored_count > 0:
        print(f"  Restored {restored_count} hand instances")
    else:
        print(f"  No hands needed restoration")

    print("\n" + "=" * 80)
    print("PASS 2 Complete: Bbox sequences cleaned")
    print("=" * 80)

    return raw_data
