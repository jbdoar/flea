"""Opt-in tests for a physically connected FlyCapture2 camera.

Run with: ``$env:FLEA_HARDWARE_TESTS=1; py -m pytest -m hardware``
"""

from __future__ import annotations

import os

import pytest

from flea import Camera, CaptureSession, GrabMode, PixelFormat, PropertyType, Recorder

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("FLEA_HARDWARE_TESTS") != "1",
        reason="set FLEA_HARDWARE_TESTS=1 to exercise a connected camera",
    ),
]


def _embedded_values(config: object) -> dict[str, bool]:
    return {
        "timestamp": config.timestamp.enabled,
        "gain": config.gain.enabled,
        "shutter": config.shutter.enabled,
        "brightness": config.brightness.enabled,
        "exposure": config.exposure.enabled,
        "white_balance": config.white_balance.enabled,
        "frame_counter": config.frame_counter.enabled,
        "strobe_pattern": config.strobe_pattern.enabled,
        "gpio_pin_state": config.gpio_pin_state.enabled,
        "roi_position": config.roi_position.enabled,
    }


def test_live_conversion_metadata_configuration_and_format7() -> None:
    with Camera() as camera:
        original_capture = camera.get_capture_config()
        original_embedded = camera.get_embedded_metadata_config()
        original_video = camera._native.get_video_mode_and_frame_rate()

        try:
            configured = camera.configure_capture(
                timeout_ms=2_000,
                num_buffers=4,
                grab_mode=GrabMode.BUFFERED,
            )
            assert configured.timeout_ms == 2_000
            assert configured.num_buffers == 4
            assert configured.grab_mode is GrabMode.BUFFERED

            camera.configure_embedded_metadata(
                timestamp=original_embedded.timestamp.available,
                frame_counter=original_embedded.frame_counter.available,
            )
            camera.start()
            frames = list(camera.frames(format="bgr", count=5))
            camera.stop()

            assert all(frame.data.shape == (frame.height, frame.width, 3) for frame in frames)
            assert all(frame.pixel_format is PixelFormat.BGR8 for frame in frames)
            assert [frame.sequence for frame in frames] == [1, 2, 3, 4, 5]
            if original_embedded.frame_counter.available:
                counters = [frame.metadata.frame_counter for frame in frames]
                assert len(set(counters)) == len(counters)
                assert all(frame.dropped_frames == 0 for frame in frames[1:])

            capabilities = camera.get_format7_capabilities(mode=0)
            assert capabilities.supported
            if original_video[0] != 23:  # FC2_VIDEOMODE_FORMAT7 cannot restore via standard API.
                camera.configure_format7(
                    width=capabilities.max_width,
                    height=capabilities.max_height,
                    pixel_format=PixelFormat.RAW8,
                )
                camera.start()
                raw = camera.grab()
                camera.stop()
                assert raw.data.shape == (capabilities.max_height, capabilities.max_width)
                assert raw.pixel_format is PixelFormat.RAW8
        finally:
            camera.stop()
            if original_video[0] != 23:
                camera._native.set_video_mode_and_frame_rate(*original_video)
            camera.configure_embedded_metadata(**_embedded_values(original_embedded))
            camera.configure_capture(
                timeout_ms=original_capture.timeout_ms,
                num_buffers=original_capture.num_buffers,
                grab_mode=original_capture.grab_mode,
                high_performance=original_capture.high_performance,
            )


def test_live_background_capture_session_and_recording(tmp_path) -> None:
    with Camera() as camera:
        original = camera.get_capture_config()
        with CaptureSession(
            camera,
            format="bgr",
            queue_size=8,
            policy="lossless",
            grab_timeout_ms=1_000,
        ) as session:
            frames = [session.get(timeout=2) for _ in range(30)]
            stats = session.stats()

        assert len(frames) == 30
        assert all(frame.data.shape == (frame.height, frame.width, 3) for frame in frames)
        assert stats.frames_delivered == 30
        assert stats.frames_dropped == 0
        assert camera.get_capture_config() == original

        with CaptureSession(camera, format="bgr", policy="lossless") as session:
            with Recorder(tmp_path / "live-recording", format="png") as recorder:
                recorder.record(session, count=3)
        assert recorder.frames_written == 3
        assert len(list(recorder.destination.glob("frame_*.png"))) == 3


def test_live_short_soak() -> None:
    frame_count = int(os.environ.get("FLEA_SOAK_FRAMES", "300"))
    with Camera() as camera:
        with CaptureSession(
            camera, format="bgr", queue_size=16, policy="lossless"
        ) as session:
            for _ in range(frame_count):
                session.get(timeout=2)
            stats = session.stats()
    assert stats.frames_delivered == frame_count
    assert stats.frames_dropped == 0
    assert stats.acquisition_fps > 20


def test_live_property_updates_during_capture() -> None:
    with Camera() as camera:
        states = camera.list_properties()
        assert set(states) == set(PropertyType)
        assert states[PropertyType.SHUTTER].value is not None
        assert states[PropertyType.GAIN].value is not None
        original_shutter = camera.get_property(PropertyType.SHUTTER)
        original_gain = camera.get_property(PropertyType.GAIN)
        shutter_info = camera.get_property_info(PropertyType.SHUTTER)
        gain_info = camera.get_property_info(PropertyType.GAIN)
        shutter_target = min(
            shutter_info.absolute_maximum,
            max(shutter_info.absolute_minimum, original_shutter.value * 0.9),
        )
        gain_target = min(
            gain_info.absolute_maximum,
            max(gain_info.absolute_minimum, original_gain.value + 0.5),
        )
        try:
            with CaptureSession(camera, format="bgr") as session:
                live_states = session.list_properties()
                assert live_states[PropertyType.SHUTTER].value is not None
                session.set_property_sync(
                    PropertyType.SHUTTER, shutter_target, absolute=True, auto=False
                )
                session.set_property_sync(
                    PropertyType.GAIN, gain_target, absolute=True, auto=False
                )
                for _ in range(10):
                    session.get(timeout=2)
            updated_shutter = camera.get_property(PropertyType.SHUTTER)
            updated_gain = camera.get_property(PropertyType.GAIN)
            assert not updated_shutter.auto
            assert not updated_gain.auto
            assert updated_shutter.value == pytest.approx(shutter_target, rel=0.1)
            assert updated_gain.value == pytest.approx(gain_target, abs=0.2)
        finally:
            camera.set_property(
                PropertyType.SHUTTER,
                original_shutter.value if original_shutter.absolute else original_shutter.value_a,
                absolute=original_shutter.absolute,
                auto=original_shutter.auto,
                enabled=original_shutter.enabled,
                value_b=original_shutter.value_b,
            )
            camera.set_property(
                PropertyType.GAIN,
                original_gain.value if original_gain.absolute else original_gain.value_a,
                absolute=original_gain.absolute,
                auto=original_gain.auto,
                enabled=original_gain.enabled,
                value_b=original_gain.value_b,
            )
