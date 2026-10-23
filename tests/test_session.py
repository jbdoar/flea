from __future__ import annotations

import time

import numpy as np
import pytest

from flea import (
    CameraStateError,
    CaptureConfig,
    CaptureSession,
    CaptureSessionError,
    Frame,
    FrameTimestamp,
    GrabMode,
    ImageMetadata,
    PixelFormat,
    PropertyType,
    QueuePolicy,
)


def make_frame(sequence: int) -> Frame:
    return Frame(
        data=np.full((2, 2), sequence, dtype=np.uint8),
        pixel_format=PixelFormat.MONO8,
        rows=2,
        columns=2,
        stride=2,
        received_bytes=4,
        source_pixel_format=PixelFormat.MONO8,
        sequence=sequence,
        timestamp=FrameTimestamp(0, 0, 0, 0, 0),
        metadata=ImageMetadata(0, 0, 0, 0, 0, 0, sequence, 0, 0, 0),
    )


class FakeCamera:
    def __init__(self, *, delay: float = 0.0, fail_after: int | None = None) -> None:
        self.index = 0
        self.connected = True
        self.capturing = False
        self.delay = delay
        self.fail_after = fail_after
        self.sequence = 0
        self.config = CaptureConfig(10, -1, GrabMode.LATEST, False)
        self.property_writes: list[tuple[object, ...]] = []

    def get_capture_config(self) -> CaptureConfig:
        return self.config

    def configure_capture(self, **changes: object) -> CaptureConfig:
        self.config = CaptureConfig(
            num_buffers=int(changes.get("num_buffers", self.config.num_buffers)),
            timeout_ms=int(changes.get("timeout_ms", self.config.timeout_ms)),
            grab_mode=GrabMode(changes.get("grab_mode", self.config.grab_mode)),
            high_performance=bool(
                changes.get("high_performance", self.config.high_performance)
            ),
        )
        return self.config

    def start(self) -> FakeCamera:
        self.capturing = True
        return self

    def stop(self) -> None:
        self.capturing = False

    def grab(self, *, format: object = None) -> Frame:
        if self.delay:
            time.sleep(self.delay)
        self.sequence += 1
        if self.fail_after is not None and self.sequence > self.fail_after:
            raise RuntimeError("camera failed")
        return make_frame(self.sequence)

    def set_property(self, property_type: object, value: object, **options: object) -> None:
        self.property_writes.append((property_type, value, options))


def test_session_streams_and_restores_camera_configuration() -> None:
    camera = FakeCamera(delay=0.001)
    original = camera.config
    with CaptureSession(camera, queue_size=3, policy="lossless") as session:
        frames = [session.get(timeout=1) for _ in range(5)]
        assert [frame.sequence for frame in frames] == [1, 2, 3, 4, 5]
        assert camera.config.grab_mode is GrabMode.BUFFERED
        assert camera.config.timeout_ms == 500
    assert camera.config == original
    assert not camera.capturing


def test_latest_policy_drops_oldest_frames_for_slow_consumer() -> None:
    camera = FakeCamera()
    with CaptureSession(camera, queue_size=2, policy=QueuePolicy.LATEST) as session:
        deadline = time.monotonic() + 1
        while session.stats().frames_acquired < 20 and time.monotonic() < deadline:
            time.sleep(0.001)
        frame = session.get(timeout=1)
        stats = session.stats()
        assert frame.sequence > 1
        assert stats.frames_dropped > 0


def test_worker_error_is_propagated() -> None:
    camera = FakeCamera(fail_after=0)
    with CaptureSession(camera) as session:
        with pytest.raises(CaptureSessionError) as caught:
            session.get(timeout=1)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_session_rejects_camera_that_is_already_capturing() -> None:
    camera = FakeCamera()
    camera.capturing = True
    with pytest.raises(CameraStateError, match="stopped"):
        CaptureSession(camera).start()


def test_property_updates_are_coalesced_and_applied_by_worker() -> None:
    camera = FakeCamera(delay=0.001)
    with CaptureSession(camera) as session:
        session.set_property(PropertyType.SHUTTER, 4.0)
        session.set_property(PropertyType.SHUTTER, 5.0)
        deadline = time.monotonic() + 1
        while not camera.property_writes and time.monotonic() < deadline:
            time.sleep(0.001)
    assert camera.property_writes == [
        (
            PropertyType.SHUTTER,
            5.0,
            {
                "absolute": True,
                "auto": False,
                "enabled": True,
                "value_b": None,
                "one_push": None,
            },
        )
    ]
