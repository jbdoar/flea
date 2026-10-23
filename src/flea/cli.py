from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .camera import Camera, list_cameras
from .recording import Recorder, RecordingFormat
from .session import CaptureSession, QueuePolicy


def info(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List FlyCapture2 cameras")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    records = []
    for reference in list_cameras():
        with Camera(index=reference.index) as camera:
            records.append(asdict(camera.info))
    if args.json:
        print(json.dumps(records, indent=2))
    elif not records:
        print("No FlyCapture2 cameras found")
    else:
        for index, record in enumerate(records):
            print(
                f"[{index}] {record['model']} serial={record['serial_number']} "
                f"firmware={record['firmware_version']} resolution={record['resolution']}"
            )
    return 0


def capture(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture a FlyCapture2 image collection")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--camera", type=int, default=0, help="zero-based camera index")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--output-format",
        choices=("native", "bgr", "rgb", "mono8", "raw8"),
        default="bgr",
    )
    parser.add_argument(
        "--file-format",
        choices=[value.value for value in RecordingFormat],
        default="npy",
    )
    parser.add_argument(
        "--policy", choices=[value.value for value in QueuePolicy], default="lossless"
    )
    parser.add_argument("--queue-size", type=int, default=16)
    args = parser.parse_args(argv)

    with Camera(index=args.camera) as camera:
        with CaptureSession(
            camera,
            format=args.output_format,
            queue_size=args.queue_size,
            policy=args.policy,
        ) as session:
            with Recorder(args.destination, format=args.file_format) as recorder:
                recorder.record(session, count=args.count)
            stats = session.stats()
    print(
        f"Captured {recorder.frames_written} frames to {recorder.destination} "
        f"at {stats.acquisition_fps:.2f} FPS; queue drops={stats.frames_dropped}"
    )
    return 0


def preview(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview a FlyCapture2 camera with OpenCV")
    parser.add_argument("--camera", type=int, default=0, help="zero-based camera index")
    parser.add_argument("--queue-size", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        import cv2
    except ImportError:
        parser.error("preview requires: py -m pip install -e '.[opencv]'")

    try:
        with Camera(index=args.camera) as camera:
            with CaptureSession(
                camera, format="bgr", queue_size=args.queue_size, policy="latest"
            ) as session:
                for frame in session:
                    cv2.imshow(camera.info.model, frame.data)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    finally:
        cv2.destroyAllWindows()
    return 0
