#!/usr/bin/env python3
"""CLI entry point for running HaWoR as a subprocess.

Reports progress directly to Redis — both the Celery task state
(celery-task-meta-{id}) and the job snapshot (job:{id}:meta).
Exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure the pipeline package is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="HaWoR hand-pose subprocess")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-mp4", default=None)
    parser.add_argument("--img-focal", type=float, default=600.0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--celery-task-id", required=True)
    parser.add_argument("--input-key", required=True)
    parser.add_argument("--output-key", required=True)
    parser.add_argument("--input-bucket", default="")
    parser.add_argument("--output-bucket", default="")
    args = parser.parse_args()

    # Load .env before importing pipeline config.
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))

    import redis as redis_lib
    from pipeline.config import get_settings

    settings = get_settings()
    r = redis_lib.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_pass,
        decode_responses=True,
    )

    def _publish_state(stage: str, stage_progress: int, detail: str = "") -> None:
        meta = {
            "job_id": args.job_id,
            "input_key": args.input_key,
            "output_key": args.output_key,
            "stage": stage,
            "stage_progress": stage_progress,
            "progress": stage_progress,
        }
        if args.input_bucket:
            meta["input_bucket"] = args.input_bucket
        if args.output_bucket:
            meta["output_bucket"] = args.output_bucket
        if detail:
            meta["detail"] = detail

        # Update Celery task state (what the CLI monitor reads).
        try:
            r.set(
                f"celery-task-meta-{args.celery_task_id}",
                json.dumps({
                    "status": stage,
                    "result": meta,
                    "traceback": None,
                    "children": [],
                    "date_done": None,
                    "task_id": args.celery_task_id,
                }),
            )
        except Exception as exc:
            print(f"[HaWoR] WARNING: celery state update failed: {exc}", file=sys.stderr, flush=True)

        # Update job progress snapshot.
        try:
            r.hset(f"job:{args.job_id}:meta", mapping={
                "last_stage": stage,
                "last_stage_progress": str(stage_progress),
                "last_phase": stage,
                "last_overall_percent": str(stage_progress),
                "last_progress_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            print(f"[HaWoR] WARNING: progress snapshot failed: {exc}", file=sys.stderr, flush=True)

    def _progress_cb(phase: str, percent: int, detail: str | None) -> None:
        msg = f"\r[HaWoR] {phase}: {percent}%"
        if detail:
            msg += f" — {detail}"
        # print(msg, end="", file=sys.stderr, flush=True)
        _publish_state("PROCESSING", percent, f"{phase}: {detail}" if detail else phase)

    try:
        from api import process_hand_pose

        _publish_state("PROCESSING", 0, "starting hand-pose estimation")

        process_hand_pose(
            args.input_video,
            args.output_json,
            output_mp4=args.output_mp4,
            img_focal=args.img_focal,
            checkpoint=args.checkpoint,
            progress_cb=_progress_cb,
        )

        _publish_state("PROCESSING", 100, "hand-pose estimation complete")
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(
            json.dumps({"error": repr(exc), "traceback": traceback.format_exc()}),
            file=sys.stdout,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
