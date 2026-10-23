from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class PropertyType(IntEnum):
    BRIGHTNESS = 0
    AUTO_EXPOSURE = 1
    SHARPNESS = 2
    WHITE_BALANCE = 3
    HUE = 4
    SATURATION = 5
    GAMMA = 6
    IRIS = 7
    FOCUS = 8
    ZOOM = 9
    PAN = 10
    TILT = 11
    SHUTTER = 12
    GAIN = 13
    TRIGGER_MODE = 14
    TRIGGER_DELAY = 15
    FRAME_RATE = 16
    TEMPERATURE = 17


class PixelFormat(IntEnum):
    UNSPECIFIED = 0
    MONO8 = 0x80000000
    YUV411 = 0x40000000
    YUV422 = 0x20000000
    YUV444 = 0x10000000
    RGB8 = 0x08000000
    MONO16 = 0x04000000
    RGB16 = 0x02000000
    SIGNED_MONO16 = 0x01000000
    SIGNED_RGB16 = 0x00800000
    RAW8 = 0x00400000
    RAW16 = 0x00200000
    MONO12 = 0x00100000
    RAW12 = 0x00080000
    BGR8 = 0x80000008
    BGRU8 = 0x40000008
    RGBU8 = 0x40000002
    BGR16 = 0x02000001
    BGRU16 = 0x02000002
    YUV422_JPEG = 0x40000001


class GrabMode(IntEnum):
    """How FlyCapture2 selects frames from its receive buffers."""

    LATEST = 0
    BUFFERED = 1


@dataclass(frozen=True, slots=True)
class FrameTimestamp:
    seconds: int
    microseconds: int
    cycle_seconds: int
    cycle_count: int
    cycle_offset: int

    @property
    def unix_seconds(self) -> float:
        return self.seconds + self.microseconds / 1_000_000


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    timestamp: int
    gain: int
    shutter: int
    brightness: int
    exposure: int
    white_balance: int
    frame_counter: int
    strobe_pattern: int
    gpio_pin_state: int
    roi_position: int


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    num_buffers: int
    timeout_ms: int
    grab_mode: GrabMode
    high_performance: bool


@dataclass(frozen=True, slots=True)
class Format7Capabilities:
    mode: int
    supported: bool
    max_width: int
    max_height: int
    offset_x_step: int
    offset_y_step: int
    width_step: int
    height_step: int
    pixel_format_bits: int


@dataclass(frozen=True, slots=True)
class EmbeddedProperty:
    available: bool
    enabled: bool


@dataclass(frozen=True, slots=True)
class EmbeddedMetadataConfig:
    timestamp: EmbeddedProperty
    gain: EmbeddedProperty
    shutter: EmbeddedProperty
    brightness: EmbeddedProperty
    exposure: EmbeddedProperty
    white_balance: EmbeddedProperty
    frame_counter: EmbeddedProperty
    strobe_pattern: EmbeddedProperty
    gpio_pin_state: EmbeddedProperty
    roi_position: EmbeddedProperty


@dataclass(frozen=True, slots=True)
class CameraRef:
    index: int
    guid: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CameraInfo:
    serial_number: int
    model: str
    vendor: str
    sensor: str
    resolution: str
    driver: str
    firmware_version: str
    firmware_build_time: str
    is_color: bool


@dataclass(frozen=True, slots=True)
class CameraProperty:
    type: PropertyType
    present: bool
    absolute: bool
    auto: bool
    enabled: bool
    value_a: int
    value_b: int
    value: float


@dataclass(frozen=True, slots=True)
class CameraPropertyInfo:
    type: PropertyType
    present: bool
    auto_supported: bool
    manual_supported: bool
    on_off_supported: bool
    absolute_supported: bool
    readable: bool
    minimum: int
    maximum: int
    absolute_minimum: float
    absolute_maximum: float
    units: str
    unit_abbreviation: str


@dataclass(frozen=True, slots=True)
class TriggerMode:
    enabled: bool
    polarity: int
    source: int
    mode: int
    parameter: int


@dataclass(frozen=True, slots=True)
class Frame:
    data: NDArray[np.generic]
    pixel_format: PixelFormat | int
    rows: int
    columns: int
    stride: int
    received_bytes: int
    source_pixel_format: PixelFormat | int
    sequence: int
    timestamp: FrameTimestamp
    metadata: ImageMetadata
    dropped_frames: int | None = None

    @property
    def height(self) -> int:
        return self.rows

    @property
    def width(self) -> int:
        return self.columns
