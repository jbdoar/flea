import ctypes as ct

import numpy as np

from flea._native import (
    NativeCamera,
    _Config,
    _EmbeddedImageInfo,
    _Format7ImageSettings,
    _Image,
    _ImageMetadata,
    _Property,
    _PropertyInfo,
    _TimeStamp,
    _TriggerMode,
)
from flea.types import PixelFormat


def test_core_structure_sizes_match_flycapture2_64_bit_abi() -> None:
    assert ct.sizeof(_Image) == 48
    assert ct.sizeof(_Property) == 68
    assert ct.sizeof(_PropertyInfo) == 1104
    assert ct.sizeof(_TriggerMode) == 52
    assert ct.sizeof(_TimeStamp) == 56
    assert ct.sizeof(_ImageMetadata) == 164
    assert ct.sizeof(_Config) == 108
    assert ct.sizeof(_Format7ImageSettings) == 56
    assert ct.sizeof(_EmbeddedImageInfo) == 80


def test_shape_frame_removes_row_padding() -> None:
    packed = np.array([[1, 2, 3, 99], [4, 5, 6, 99]], dtype=np.uint8)
    frame = NativeCamera._shape_frame(packed, 2, 3, PixelFormat.MONO8)
    np.testing.assert_array_equal(frame, [[1, 2, 3], [4, 5, 6]])
    assert frame.flags.owndata


def test_shape_frame_handles_rgb() -> None:
    packed = np.arange(12, dtype=np.uint8).reshape(2, 6)
    frame = NativeCamera._shape_frame(packed, 2, 2, PixelFormat.RGB8)
    assert frame.shape == (2, 2, 3)


def test_shape_frame_handles_yuv422_as_two_components_per_pixel() -> None:
    packed = np.arange(12, dtype=np.uint8).reshape(2, 6)
    frame = NativeCamera._shape_frame(packed, 2, 3, PixelFormat.YUV422)
    assert frame.shape == (2, 3, 2)
