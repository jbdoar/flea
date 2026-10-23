from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .camera import Camera, list_cameras
from .recording import Recorder, RecordingFormat
from .session import CaptureSession, QueuePolicy
from .types import CameraPropertyInfo, PropertyType

_SLIDER_MAX = 1_000


def _slider_value(position: int, info: CameraPropertyInfo, *, logarithmic: bool) -> float:
    ratio = position / _SLIDER_MAX
    low = info.absolute_minimum
    high = info.absolute_maximum
    if logarithmic:
        low = max(low, 1e-9)
        return math.exp(math.log(low) + ratio * (math.log(high) - math.log(low)))
    return low + ratio * (high - low)


def _slider_position(value: float, info: CameraPropertyInfo, *, logarithmic: bool) -> int:
    low = info.absolute_minimum
    high = info.absolute_maximum
    value = min(max(value, low), high)
    if high <= low:
        return 0
    if logarithmic:
        low = max(low, 1e-9)
        value = max(value, low)
        ratio = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    else:
        ratio = (value - low) / (high - low)
    return round(ratio * _SLIDER_MAX)


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
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args(argv)
    if args.window_width < 1 or args.window_height < 1:
        parser.error("window dimensions must be positive")
    try:
        import cv2
    except ImportError:
        parser.error("preview requires: py -m pip install -e '.[opencv]'")

    try:
        with Camera(index=args.camera) as camera:
            camera_info = camera.info
            shutter = camera.get_property(PropertyType.SHUTTER)
            shutter_info = camera.get_property_info(PropertyType.SHUTTER)
            gain = camera.get_property(PropertyType.GAIN)
            gain_info = camera.get_property_info(PropertyType.GAIN)
            window = camera_info.model
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, args.window_width, args.window_height)
            with CaptureSession(
                camera, format="bgr", queue_size=args.queue_size, policy="latest"
            ) as session:
                controls: list[tuple[str, CameraPropertyInfo, bool]] = []
                if not args.no_controls:
                    if shutter_info.present and shutter_info.absolute_supported:
                        controls.append(("Shutter", shutter_info, True))
                        shutter_ready = False

                        def set_shutter(position: int) -> None:
                            if not shutter_ready:
                                return
                            session.set_property(
                                PropertyType.SHUTTER,
                                _slider_value(position, shutter_info, logarithmic=True),
                                auto=bool(cv2.getTrackbarPos("Auto shutter", window)),
                            )

                        def set_auto_shutter(value: int) -> None:
                            set_shutter(cv2.getTrackbarPos("Shutter", window))

                        cv2.createTrackbar(
                            "Shutter",
                            window,
                            _slider_position(shutter.value, shutter_info, logarithmic=True),
                            _SLIDER_MAX,
                            set_shutter,
                        )
                        cv2.createTrackbar(
                            "Auto shutter",
                            window,
                            int(shutter.auto),
                            1,
                            set_auto_shutter,
                        )
                        shutter_ready = True
                    if gain_info.present and gain_info.absolute_supported:
                        controls.append(("Gain", gain_info, False))
                        gain_ready = False

                        def set_gain(position: int) -> None:
                            if not gain_ready:
                                return
                            session.set_property(
                                PropertyType.GAIN,
                                _slider_value(position, gain_info, logarithmic=False),
                                auto=bool(cv2.getTrackbarPos("Auto gain", window)),
                            )

                        def set_auto_gain(value: int) -> None:
                            set_gain(cv2.getTrackbarPos("Gain", window))

                        cv2.createTrackbar(
                            "Gain",
                            window,
                            _slider_position(gain.value, gain_info, logarithmic=False),
                            _SLIDER_MAX,
                            set_gain,
                        )
                        cv2.createTrackbar(
                            "Auto gain", window, int(gain.auto), 1, set_auto_gain
                        )
                        gain_ready = True
                for frame in session:
                    display = frame.data
                    y = 24
                    for name, property_info, logarithmic in controls:
                        value = _slider_value(
                            cv2.getTrackbarPos(name, window),
                            property_info,
                            logarithmic=logarithmic,
                        )
                        text = f"{name}: {value:.3g} {property_info.unit_abbreviation}"
                        cv2.putText(
                            display,
                            text,
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        y += 24
                    cv2.imshow(window, display)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    finally:
        cv2.destroyAllWindows()
    return 0
