from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._native import NativeCamera
from .errors import CameraStateError
from .types import (
    CameraInfo,
    CameraProperty,
    CameraPropertyInfo,
    CameraPropertyState,
    CameraRef,
    CaptureConfig,
    EmbeddedMetadataConfig,
    Format7Capabilities,
    Frame,
    GrabMode,
    PixelFormat,
    PropertyType,
    TriggerMode,
)

_OUTPUT_FORMATS = {
    "bgr": PixelFormat.BGR8,
    "rgb": PixelFormat.RGB8,
    "mono8": PixelFormat.MONO8,
    "raw8": PixelFormat.RAW8,
}


def list_cameras(
    library_path: str | os.PathLike[str] | None = None,
) -> list[CameraRef]:
    """Return references for every camera visible to FlyCapture2."""
    native = NativeCamera(library_path)
    try:
        return native.list_cameras()
    finally:
        native.close()


class Camera:
    """A single FlyCapture2 camera with explicit connection and capture state."""

    def __init__(
        self,
        index: int = 0,
        *,
        serial: int | None = None,
        library_path: str | os.PathLike[str] | None = None,
        _native: Any | None = None,
    ) -> None:
        self.index = index
        self.serial = serial
        self._native = _native if _native is not None else NativeCamera(library_path)
        self._connected = False
        self._capturing = False
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def capturing(self) -> bool:
        return self._capturing

    def connect(self) -> Camera:
        if self._closed:
            raise CameraStateError("Camera has been closed")
        if not self._connected:
            self._native.connect(index=self.index, serial=self.serial)
            self._connected = True
        return self

    @property
    def info(self) -> CameraInfo:
        self._require_connected()
        return self._native.camera_info()

    def start(self) -> Camera:
        self._require_connected()
        if not self._capturing:
            self._native.start_capture()
            self._capturing = True
        return self

    def stop(self) -> None:
        if self._capturing:
            self._native.stop_capture()
            self._capturing = False

    def grab(self, *, format: str | PixelFormat | None = None) -> Frame:
        if not self._capturing:
            raise CameraStateError("Call start() before grab()")
        return self._native.retrieve_frame(self._resolve_output_format(format))

    def grab_array(
        self, *, format: str | PixelFormat | None = None
    ) -> NDArray[np.generic]:
        """Grab one frame and return only its copied NumPy array."""
        return self.grab(format=format).data

    def frames(
        self,
        *,
        format: str | PixelFormat | None = None,
        count: int | None = None,
    ) -> Iterator[Frame]:
        """Yield frames until *count* is reached, or indefinitely when omitted."""
        produced = 0
        while count is None or produced < count:
            yield self.grab(format=format)
            produced += 1

    @staticmethod
    def _resolve_output_format(format: str | PixelFormat | None) -> PixelFormat | None:
        if format is None or format == "native":
            return None
        if isinstance(format, str):
            try:
                return _OUTPUT_FORMATS[format.lower()]
            except KeyError as exc:
                choices = ", ".join(("native", *_OUTPUT_FORMATS))
                raise ValueError(f"Unknown output format {format!r}; choose {choices}") from exc
        return PixelFormat(format)

    def get_capture_config(self) -> CaptureConfig:
        self._require_connected()
        return self._native.get_capture_config()

    def configure_capture(
        self,
        *,
        timeout_ms: int | None = None,
        num_buffers: int | None = None,
        grab_mode: GrabMode | None = None,
        high_performance: bool | None = None,
    ) -> CaptureConfig:
        """Configure receive behavior. The camera must not be capturing."""
        self._require_connected()
        if self._capturing:
            raise CameraStateError("Stop capture before changing capture configuration")
        if timeout_ms is not None and timeout_ms < -1:
            raise ValueError("timeout_ms must be -1 (infinite), 0, or positive")
        if num_buffers is not None and num_buffers < 1:
            raise ValueError("num_buffers must be at least 1")
        return self._native.configure_capture(
            timeout_ms=timeout_ms,
            num_buffers=num_buffers,
            grab_mode=GrabMode(grab_mode) if grab_mode is not None else None,
            high_performance=high_performance,
        )

    def get_format7_capabilities(self, mode: int = 0) -> Format7Capabilities:
        self._require_connected()
        return self._native.get_format7_capabilities(mode)

    def configure_format7(
        self,
        *,
        width: int,
        height: int,
        pixel_format: PixelFormat = PixelFormat.RAW8,
        offset_x: int = 0,
        offset_y: int = 0,
        mode: int = 0,
        packet_percent: float = 100.0,
    ) -> None:
        """Set a validated Format7 ROI and native camera pixel format."""
        self._require_connected()
        if self._capturing:
            raise CameraStateError("Stop capture before changing Format7 settings")
        if not 1.0 <= packet_percent <= 100.0:
            raise ValueError("packet_percent must be between 1 and 100")
        capabilities = self.get_format7_capabilities(mode)
        if not capabilities.supported:
            raise ValueError(f"Format7 mode {mode} is not supported")
        checks = (
            (offset_x, capabilities.offset_x_step, "offset_x"),
            (offset_y, capabilities.offset_y_step, "offset_y"),
            (width, capabilities.width_step, "width"),
            (height, capabilities.height_step, "height"),
        )
        for value, step, name in checks:
            if step and value % step:
                raise ValueError(f"{name} must be a multiple of {step}")
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if offset_x + width > capabilities.max_width or offset_y + height > capabilities.max_height:
            raise ValueError(
                f"ROI exceeds the {capabilities.max_width}x{capabilities.max_height} sensor area"
            )
        if not capabilities.pixel_format_bits & (int(pixel_format) & 0xFFFFFFFF):
            raise ValueError(f"{pixel_format.name} is not supported by Format7 mode {mode}")
        self._native.configure_format7(
            width=width,
            height=height,
            pixel_format=pixel_format,
            offset_x=offset_x,
            offset_y=offset_y,
            mode=mode,
            packet_percent=packet_percent,
        )

    def get_embedded_metadata_config(self) -> EmbeddedMetadataConfig:
        self._require_connected()
        return self._native.get_embedded_metadata_config()

    def configure_embedded_metadata(
        self,
        *,
        timestamp: bool | None = None,
        gain: bool | None = None,
        shutter: bool | None = None,
        brightness: bool | None = None,
        exposure: bool | None = None,
        white_balance: bool | None = None,
        frame_counter: bool | None = None,
        strobe_pattern: bool | None = None,
        gpio_pin_state: bool | None = None,
        roi_position: bool | None = None,
    ) -> EmbeddedMetadataConfig:
        """Enable selected camera-supplied metadata while preserving omitted fields."""
        self._require_connected()
        if self._capturing:
            raise CameraStateError("Stop capture before changing embedded metadata")
        return self._native.configure_embedded_metadata(
            timestamp=timestamp,
            gain=gain,
            shutter=shutter,
            brightness=brightness,
            exposure=exposure,
            white_balance=white_balance,
            frame_counter=frame_counter,
            strobe_pattern=strobe_pattern,
            gpio_pin_state=gpio_pin_state,
            roi_position=roi_position,
        )

    def get_property(self, property_type: PropertyType) -> CameraProperty:
        self._require_connected()
        return self._native.get_property(PropertyType(property_type))

    def get_property_info(self, property_type: PropertyType) -> CameraPropertyInfo:
        self._require_connected()
        return self._native.get_property_info(PropertyType(property_type))

    def list_properties(self) -> dict[PropertyType, CameraPropertyState]:
        """Return capabilities and current values for every FlyCapture2 property."""
        self._require_connected()
        states = {}
        for property_type in PropertyType:
            info = self.get_property_info(property_type)
            value = self.get_property(property_type) if info.present and info.readable else None
            states[property_type] = CameraPropertyState(info=info, value=value)
        return states

    def set_property(
        self,
        property_type: PropertyType,
        value: float | int,
        *,
        absolute: bool = True,
        auto: bool = False,
        enabled: bool = True,
        value_b: int | None = None,
        one_push: bool | None = None,
    ) -> None:
        self._require_connected()
        self._native.set_property(
            PropertyType(property_type),
            value,
            absolute=absolute,
            auto=auto,
            enabled=enabled,
            value_b=value_b,
            one_push=one_push,
        )

    def get_trigger(self) -> TriggerMode:
        self._require_connected()
        return self._native.get_trigger()

    def configure_trigger(
        self,
        *,
        enabled: bool,
        source: int = 0,
        mode: int = 0,
        polarity: int = 0,
        parameter: int = 0,
    ) -> None:
        self._require_connected()
        self._native.set_trigger(
            TriggerMode(
                enabled=enabled,
                source=source,
                mode=mode,
                polarity=polarity,
                parameter=parameter,
            )
        )

    def fire_software_trigger(self) -> None:
        self._require_connected()
        self._native.fire_software_trigger()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop()
            if self._connected:
                self._native.disconnect()
                self._connected = False
        finally:
            self._native.close()
            self._closed = True

    def _require_connected(self) -> None:
        if not self._connected:
            raise CameraStateError("Call connect() before using the camera")

    def __enter__(self) -> Camera:
        return self.connect()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
