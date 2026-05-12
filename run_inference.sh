#!/usr/bin/env bash
# Run HaWoR hand-pose estimation on video files in a directory.
#
# Usage:  ./run_inference.sh <VIDEO_DIR> [--img_focal <FOCAL>]
# Example: ./run_inference.sh /data/videos --img_focal 747.36
#
# Outputs per video: <video_stem>.json (MediaPipe-compatible hand landmarks)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <VIDEO_DIR> [--img_focal <FOCAL>]"
    exit 1
fi

VIDEO_DIR="$(realpath "$1")"
shift

if [[ ! -d "$VIDEO_DIR" ]]; then
    echo "ERROR: Directory not found: $VIDEO_DIR"
    exit 1
fi

echo "=== HaWoR Hand-Pose Inference ==="
echo "Video dir: $VIDEO_DIR"
echo ""

uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/main.py" \
    --video_folder "$VIDEO_DIR" \
    "$@"

echo ""
echo "======================================"
echo "Done."
