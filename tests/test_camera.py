from __future__ import annotations

import numpy as np
import pytest

from flea import (
    Camera,
    CameraInfo,
    CameraStateError,
    Frame,
    FrameTimestamp,
    ImageMetadata,
    PixelFormat,
    PropertyType,
)


class FakeNative:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.frame = Frame(
            data=np.arange(12, dtype=np.uint8).reshape(3, 4),
            pixel_format=PixelFormat.MONO8,
            rows=3,
            columns=4,
            stride=4,
            received_bytes=12,
            source_pixel_format=PixelFormat.MONO8,
            sequence=1,
            timestamp=FrameTimestamp(0, 0, 0, 0, 0),
            metadata=ImageMetadata(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

    def connect(self, *, index: int, serial: int | None) -> None:
        self.calls.append(("connect", index, serial))

    def camera_info(self) -> CameraInfo:
        return CameraInfo(7, "Flea3", "FLIR", "sensor", "4x3", "driver", "1.0", "today", True)

    def start_capture(self) -> None:
        self.calls.append("start")

    def stop_capture(self) -> None:
        self.calls.append("stop")

    def retrieve_frame(self, output_format: PixelFormat | None) -> Frame:
        self.calls.append(("grab", output_format))
        return self.frame

    def get_property(self, property_type: PropertyType):
        self.calls.append(("get_property", property_type))
        return property_type

    def set_property(self, property_type: PropertyType, value: float, **options: object) -> None:
        self.calls.append(("set_property", property_type, value, options))

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def close(self) -> None:
        self.calls.append("close")


def test_context_manager_runs_complete_capture_lifecycle() -> None:
    native = FakeNative()
    with Camera(index=2, _native=native) as camera:
        assert camera.info.model == "Flea3"
        frame = camera.start().grab()
        assert frame.data.shape == (3, 4)

    assert native.calls == [
        ("connect", 2, None),
        "start",
        ("grab", None),
        "stop",
        "disconnect",
        "close",
    ]


def test_grab_requires_capture() -> None:
    native = FakeNative()
    camera = Camera(_native=native).connect()
    with pytest.raises(CameraStateError, match=r"start\(\)"):
        camera.grab()
    camera.close()


def test_frames_convert_named_format_and_honor_count() -> None:
    native = FakeNative()
    camera = Camera(_native=native).connect().start()
    frames = list(camera.frames(format="bgr", count=2))
    assert len(frames) == 2
    assert native.calls[-2:] == [
        ("grab", PixelFormat.BGR8),
        ("grab", PixelFormat.BGR8),
    ]
    camera.close()


def test_set_property_passes_explicit_modes() -> None:
    native = FakeNative()
    camera = Camera(_native=native).connect()
    camera.set_property(PropertyType.SHUTTER, 4.5, absolute=True, auto=False)
    assert native.calls[-1] == (
        "set_property",
        PropertyType.SHUTTER,
        4.5,
        {
            "absolute": True,
            "auto": False,
            "enabled": True,
            "value_b": None,
            "one_push": None,
        },
    )
    camera.close()
