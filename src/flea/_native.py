from __future__ import annotations

import ctypes as ct
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .errors import FlyCapture2Error, FlyCapture2NotFoundError
from .types import (
    CameraInfo,
    CameraProperty,
    CameraPropertyInfo,
    CameraRef,
    CaptureConfig,
    EmbeddedMetadataConfig,
    EmbeddedProperty,
    Format7Capabilities,
    Frame,
    FrameTimestamp,
    GrabMode,
    ImageMetadata,
    PixelFormat,
    PropertyType,
    TriggerMode,
)

_MAX_STRING_LENGTH = 512
_OK = 0


class _Guid(ct.Structure):
    _fields_ = [("value", ct.c_uint * 4)]


class _IPAddress(ct.Structure):
    _fields_ = [("octets", ct.c_ubyte * 4)]


class _MACAddress(ct.Structure):
    _fields_ = [("octets", ct.c_ubyte * 6)]


class _ConfigROM(ct.Structure):
    _fields_ = [
        ("nodeVendorId", ct.c_uint),
        ("chipIdHi", ct.c_uint),
        ("chipIdLo", ct.c_uint),
        ("unitSpecId", ct.c_uint),
        ("unitSWVer", ct.c_uint),
        ("unitSubSWVer", ct.c_uint),
        ("vendorUniqueInfo_0", ct.c_uint),
        ("vendorUniqueInfo_1", ct.c_uint),
        ("vendorUniqueInfo_2", ct.c_uint),
        ("vendorUniqueInfo_3", ct.c_uint),
        ("pszKeyword", ct.c_char * _MAX_STRING_LENGTH),
        ("reserved", ct.c_uint * 16),
    ]


class _CameraInfo(ct.Structure):
    _fields_ = [
        ("serialNumber", ct.c_uint),
        ("interfaceType", ct.c_int),
        ("driverType", ct.c_int),
        ("isColorCamera", ct.c_int),
        ("modelName", ct.c_char * _MAX_STRING_LENGTH),
        ("vendorName", ct.c_char * _MAX_STRING_LENGTH),
        ("sensorInfo", ct.c_char * _MAX_STRING_LENGTH),
        ("sensorResolution", ct.c_char * _MAX_STRING_LENGTH),
        ("driverName", ct.c_char * _MAX_STRING_LENGTH),
        ("firmwareVersion", ct.c_char * _MAX_STRING_LENGTH),
        ("firmwareBuildTime", ct.c_char * _MAX_STRING_LENGTH),
        ("maximumBusSpeed", ct.c_int),
        ("bayerTileFormat", ct.c_int),
        ("pcieBusSpeed", ct.c_int),
        ("nodeNumber", ct.c_ushort),
        ("busNumber", ct.c_ushort),
        ("iidcVer", ct.c_uint),
        ("configROM", _ConfigROM),
        ("gigEMajorVersion", ct.c_uint),
        ("gigEMinorVersion", ct.c_uint),
        ("userDefinedName", ct.c_char * _MAX_STRING_LENGTH),
        ("xmlURL1", ct.c_char * _MAX_STRING_LENGTH),
        ("xmlURL2", ct.c_char * _MAX_STRING_LENGTH),
        ("macAddress", _MACAddress),
        ("ipAddress", _IPAddress),
        ("subnetMask", _IPAddress),
        ("defaultGateway", _IPAddress),
        ("ccpStatus", ct.c_uint),
        ("applicationIPAddress", ct.c_uint),
        ("applicationPort", ct.c_uint),
        ("reserved", ct.c_uint * 16),
    ]


class _Image(ct.Structure):
    _fields_ = [
        ("rows", ct.c_uint),
        ("cols", ct.c_uint),
        ("stride", ct.c_uint),
        ("pData", ct.POINTER(ct.c_ubyte)),
        ("dataSize", ct.c_uint),
        ("receivedDataSize", ct.c_uint),
        ("format", ct.c_int),
        ("bayerFormat", ct.c_int),
        ("imageImpl", ct.c_void_p),
    ]


class _Property(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("present", ct.c_int),
        ("absControl", ct.c_int),
        ("onePush", ct.c_int),
        ("onOff", ct.c_int),
        ("autoManualMode", ct.c_int),
        ("valueA", ct.c_uint),
        ("valueB", ct.c_uint),
        ("absValue", ct.c_float),
        ("reserved", ct.c_uint * 8),
    ]


class _PropertyInfo(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("present", ct.c_int),
        ("autoSupported", ct.c_int),
        ("manualSupported", ct.c_int),
        ("onOffSupported", ct.c_int),
        ("onePushSupported", ct.c_int),
        ("absValSupported", ct.c_int),
        ("readOutSupported", ct.c_int),
        ("min", ct.c_uint),
        ("max", ct.c_uint),
        ("absMin", ct.c_float),
        ("absMax", ct.c_float),
        ("pUnits", ct.c_char * _MAX_STRING_LENGTH),
        ("pUnitAbbr", ct.c_char * _MAX_STRING_LENGTH),
        ("reserved", ct.c_uint * 8),
    ]


class _TriggerMode(ct.Structure):
    _fields_ = [
        ("onOff", ct.c_int),
        ("polarity", ct.c_uint),
        ("source", ct.c_uint),
        ("mode", ct.c_uint),
        ("parameter", ct.c_uint),
        ("reserved", ct.c_uint * 8),
    ]


class _TimeStamp(ct.Structure):
    _fields_ = [
        ("seconds", ct.c_longlong),
        ("microSeconds", ct.c_uint),
        ("cycleSeconds", ct.c_uint),
        ("cycleCount", ct.c_uint),
        ("cycleOffset", ct.c_uint),
        ("reserved", ct.c_uint * 8),
    ]


class _ImageMetadata(ct.Structure):
    _fields_ = [
        ("embeddedTimeStamp", ct.c_uint),
        ("embeddedGain", ct.c_uint),
        ("embeddedShutter", ct.c_uint),
        ("embeddedBrightness", ct.c_uint),
        ("embeddedExposure", ct.c_uint),
        ("embeddedWhiteBalance", ct.c_uint),
        ("embeddedFrameCounter", ct.c_uint),
        ("embeddedStrobePattern", ct.c_uint),
        ("embeddedGPIOPinState", ct.c_uint),
        ("embeddedROIPosition", ct.c_uint),
        ("reserved", ct.c_uint * 31),
    ]


class _Config(ct.Structure):
    _fields_ = [
        ("numBuffers", ct.c_uint),
        ("numImageNotifications", ct.c_uint),
        ("minNumImageNotifications", ct.c_uint),
        ("grabTimeout", ct.c_int),
        ("grabMode", ct.c_int),
        ("highPerformanceRetrieveBuffer", ct.c_int),
        ("isochBusSpeed", ct.c_int),
        ("asyncBusSpeed", ct.c_int),
        ("bandwidthAllocation", ct.c_int),
        ("registerTimeoutRetries", ct.c_uint),
        ("registerTimeout", ct.c_uint),
        ("reserved", ct.c_uint * 16),
    ]


class _Format7ImageSettings(ct.Structure):
    _fields_ = [
        ("mode", ct.c_int),
        ("offsetX", ct.c_uint),
        ("offsetY", ct.c_uint),
        ("width", ct.c_uint),
        ("height", ct.c_uint),
        ("pixelFormat", ct.c_int),
        ("reserved", ct.c_uint * 8),
    ]


class _Format7Info(ct.Structure):
    _fields_ = [
        ("mode", ct.c_int),
        ("maxWidth", ct.c_uint),
        ("maxHeight", ct.c_uint),
        ("offsetHStepSize", ct.c_uint),
        ("offsetVStepSize", ct.c_uint),
        ("imageHStepSize", ct.c_uint),
        ("imageVStepSize", ct.c_uint),
        ("pixelFormatBitField", ct.c_uint),
        ("vendorPixelFormatBitField", ct.c_uint),
        ("packetSize", ct.c_uint),
        ("minPacketSize", ct.c_uint),
        ("maxPacketSize", ct.c_uint),
        ("percentage", ct.c_float),
        ("reserved", ct.c_uint * 16),
    ]


class _Format7PacketInfo(ct.Structure):
    _fields_ = [
        ("recommendedBytesPerPacket", ct.c_uint),
        ("maxBytesPerPacket", ct.c_uint),
        ("unitBytesPerPacket", ct.c_uint),
        ("reserved", ct.c_uint * 8),
    ]


class _EmbeddedProperty(ct.Structure):
    _fields_ = [("available", ct.c_int), ("onOff", ct.c_int)]


class _EmbeddedImageInfo(ct.Structure):
    _fields_ = [
        ("timestamp", _EmbeddedProperty),
        ("gain", _EmbeddedProperty),
        ("shutter", _EmbeddedProperty),
        ("brightness", _EmbeddedProperty),
        ("exposure", _EmbeddedProperty),
        ("whiteBalance", _EmbeddedProperty),
        ("frameCounter", _EmbeddedProperty),
        ("strobePattern", _EmbeddedProperty),
        ("GPIOPinState", _EmbeddedProperty),
        ("ROIPosition", _EmbeddedProperty),
    ]


def _decode(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _library_candidates(path: str | os.PathLike[str] | None) -> list[Path | str]:
    if path is not None:
        return [Path(path)]
    env_path = os.environ.get("FLYCAPTURE2_LIBRARY")
    candidates: list[Path | str] = []
    if env_path:
        candidates.append(Path(env_path))
    if sys.platform == "win32":
        roots = [os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles")]
        for root in filter(None, roots):
            candidates.append(
                Path(root)
                / "Point Grey Research"
                / "FlyCapture2"
                / "bin64"
                / "FlyCapture2_C_v100.dll"
            )
        candidates.append("FlyCapture2_C_v100.dll")
    else:
        candidates.extend(("libflycapture-c.so.2", "libflycapture-c.so"))
    return candidates


def _load_library(path: str | os.PathLike[str] | None = None) -> tuple[Any, Any | None]:
    failures: list[str] = []
    for candidate in _library_candidates(path):
        dll_dir = None
        try:
            if sys.platform == "win32" and isinstance(candidate, Path) and candidate.is_file():
                dll_dir = os.add_dll_directory(str(candidate.parent))
            return ct.CDLL(str(candidate)), dll_dir
        except (OSError, FileNotFoundError) as exc:
            if dll_dir is not None:
                dll_dir.close()
            failures.append(f"{candidate}: {exc}")
    attempted = "\n  ".join(failures)
    raise FlyCapture2NotFoundError(
        "Could not load FlyCapture2. Set FLYCAPTURE2_LIBRARY to the native library path."
        + (f"\n  {attempted}" if attempted else "")
    )


class NativeCamera:
    """Thin, stateful wrapper around one FlyCapture2 C context."""

    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        self._dll, self._dll_dir = _load_library(library_path)
        self._bind()
        self._context = ct.c_void_p()
        self._image = _Image()
        self._converted_image = _Image()
        self._image_created = False
        self._converted_image_created = False
        self._sequence = 0
        self._last_embedded_counter: int | None = None
        self._closed = False
        self._check("fc2CreateContext", self._dll.fc2CreateContext(ct.byref(self._context)))

    def _bind(self) -> None:
        d = self._dll
        signatures = {
            "fc2CreateContext": ([ct.POINTER(ct.c_void_p)], ct.c_int),
            "fc2DestroyContext": ([ct.c_void_p], ct.c_int),
            "fc2GetNumOfCameras": ([ct.c_void_p, ct.POINTER(ct.c_uint)], ct.c_int),
            "fc2GetCameraFromIndex": ([ct.c_void_p, ct.c_uint, ct.POINTER(_Guid)], ct.c_int),
            "fc2Connect": ([ct.c_void_p, ct.POINTER(_Guid)], ct.c_int),
            "fc2Disconnect": ([ct.c_void_p], ct.c_int),
            "fc2GetCameraInfo": ([ct.c_void_p, ct.POINTER(_CameraInfo)], ct.c_int),
            "fc2StartCapture": ([ct.c_void_p], ct.c_int),
            "fc2StopCapture": ([ct.c_void_p], ct.c_int),
            "fc2CreateImage": ([ct.POINTER(_Image)], ct.c_int),
            "fc2DestroyImage": ([ct.POINTER(_Image)], ct.c_int),
            "fc2RetrieveBuffer": ([ct.c_void_p, ct.POINTER(_Image)], ct.c_int),
            "fc2ConvertImageTo": (
                [ct.c_int, ct.POINTER(_Image), ct.POINTER(_Image)],
                ct.c_int,
            ),
            "fc2GetImageMetadata": (
                [ct.POINTER(_Image), ct.POINTER(_ImageMetadata)],
                ct.c_int,
            ),
            "fc2GetImageTimeStamp": ([ct.POINTER(_Image)], _TimeStamp),
            "fc2GetConfiguration": ([ct.c_void_p, ct.POINTER(_Config)], ct.c_int),
            "fc2SetConfiguration": ([ct.c_void_p, ct.POINTER(_Config)], ct.c_int),
            "fc2GetFormat7Info": (
                [ct.c_void_p, ct.POINTER(_Format7Info), ct.POINTER(ct.c_int)],
                ct.c_int,
            ),
            "fc2ValidateFormat7Settings": (
                [
                    ct.c_void_p,
                    ct.POINTER(_Format7ImageSettings),
                    ct.POINTER(ct.c_int),
                    ct.POINTER(_Format7PacketInfo),
                ],
                ct.c_int,
            ),
            "fc2SetFormat7Configuration": (
                [ct.c_void_p, ct.POINTER(_Format7ImageSettings), ct.c_float],
                ct.c_int,
            ),
            "fc2GetVideoModeAndFrameRate": (
                [ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)],
                ct.c_int,
            ),
            "fc2SetVideoModeAndFrameRate": ([ct.c_void_p, ct.c_int, ct.c_int], ct.c_int),
            "fc2GetEmbeddedImageInfo": (
                [ct.c_void_p, ct.POINTER(_EmbeddedImageInfo)],
                ct.c_int,
            ),
            "fc2SetEmbeddedImageInfo": (
                [ct.c_void_p, ct.POINTER(_EmbeddedImageInfo)],
                ct.c_int,
            ),
            "fc2GetProperty": ([ct.c_void_p, ct.POINTER(_Property)], ct.c_int),
            "fc2GetPropertyInfo": (
                [ct.c_void_p, ct.POINTER(_PropertyInfo)],
                ct.c_int,
            ),
            "fc2SetProperty": ([ct.c_void_p, ct.POINTER(_Property)], ct.c_int),
            "fc2GetTriggerMode": ([ct.c_void_p, ct.POINTER(_TriggerMode)], ct.c_int),
            "fc2SetTriggerMode": ([ct.c_void_p, ct.POINTER(_TriggerMode)], ct.c_int),
            "fc2FireSoftwareTrigger": ([ct.c_void_p], ct.c_int),
            "fc2ErrorToDescription": ([ct.c_int], ct.c_char_p),
        }
        for name, (argtypes, restype) in signatures.items():
            fn = getattr(d, name)
            fn.argtypes = argtypes
            fn.restype = restype

    def _check(self, operation: str, code: int) -> None:
        if code == _OK:
            return
        raw = self._dll.fc2ErrorToDescription(code)
        description = raw.decode("utf-8", errors="replace") if raw else "Unknown SDK error"
        raise FlyCapture2Error(operation, code, description)

    def list_cameras(self) -> list[CameraRef]:
        count = ct.c_uint()
        self._check(
            "fc2GetNumOfCameras",
            self._dll.fc2GetNumOfCameras(self._context, ct.byref(count)),
        )
        cameras = []
        for index in range(count.value):
            guid = _Guid()
            self._check(
                "fc2GetCameraFromIndex",
                self._dll.fc2GetCameraFromIndex(self._context, index, ct.byref(guid)),
            )
            cameras.append(CameraRef(index=index, guid=tuple(guid.value)))
        return cameras

    def connect(self, index: int = 0, serial: int | None = None) -> None:
        cameras = self.list_cameras()
        if serial is not None:
            for camera in cameras:
                self._check("fc2Connect", self._dll.fc2Connect(self._context, self._guid(camera)))
                if self.camera_info().serial_number == serial:
                    return
                self._check("fc2Disconnect", self._dll.fc2Disconnect(self._context))
            raise IndexError(f"No camera with serial number {serial}")
        if index < 0 or index >= len(cameras):
            raise IndexError(f"Camera index {index} is out of range ({len(cameras)} found)")
        self._check("fc2Connect", self._dll.fc2Connect(self._context, self._guid(cameras[index])))

    @staticmethod
    def _guid(camera: CameraRef) -> Any:
        guid = _Guid()
        guid.value[:] = camera.guid
        # Keep the structure alive through the native call by returning a pointer
        # whose private _obj reference owns it.
        return ct.byref(guid)

    def disconnect(self) -> None:
        self._check("fc2Disconnect", self._dll.fc2Disconnect(self._context))

    def camera_info(self) -> CameraInfo:
        info = _CameraInfo()
        self._check("fc2GetCameraInfo", self._dll.fc2GetCameraInfo(self._context, ct.byref(info)))
        return CameraInfo(
            serial_number=info.serialNumber,
            model=_decode(info.modelName),
            vendor=_decode(info.vendorName),
            sensor=_decode(info.sensorInfo),
            resolution=_decode(info.sensorResolution),
            driver=_decode(info.driverName),
            firmware_version=_decode(info.firmwareVersion),
            firmware_build_time=_decode(info.firmwareBuildTime),
            is_color=bool(info.isColorCamera),
        )

    def start_capture(self) -> None:
        if not self._image_created:
            self._check("fc2CreateImage", self._dll.fc2CreateImage(ct.byref(self._image)))
            self._image_created = True
        self._sequence = 0
        self._last_embedded_counter = None
        self._check("fc2StartCapture", self._dll.fc2StartCapture(self._context))

    def stop_capture(self) -> None:
        self._check("fc2StopCapture", self._dll.fc2StopCapture(self._context))

    def retrieve_frame(self, output_format: PixelFormat | None = None) -> Frame:
        self._check(
            "fc2RetrieveBuffer", self._dll.fc2RetrieveBuffer(self._context, ct.byref(self._image))
        )
        source = self._image
        image = source
        if output_format is not None and int(output_format) != (source.format & 0xFFFFFFFF):
            if not self._converted_image_created:
                self._check(
                    "fc2CreateImage", self._dll.fc2CreateImage(ct.byref(self._converted_image))
                )
                self._converted_image_created = True
            self._check(
                "fc2ConvertImageTo",
                self._dll.fc2ConvertImageTo(
                    int(output_format), ct.byref(source), ct.byref(self._converted_image)
                ),
            )
            image = self._converted_image
        byte_count = image.stride * image.rows
        if not image.pData or byte_count == 0:
            raise FlyCapture2Error("fc2RetrieveBuffer", -1, "SDK returned an empty image")
        packed = np.ctypeslib.as_array(image.pData, shape=(byte_count,)).reshape(
            image.rows, image.stride
        )
        data = self._shape_frame(packed, image.rows, image.cols, image.format)
        try:
            pixel_format: PixelFormat | int = PixelFormat(image.format & 0xFFFFFFFF)
        except ValueError:
            pixel_format = image.format & 0xFFFFFFFF
        try:
            source_pixel_format: PixelFormat | int = PixelFormat(source.format & 0xFFFFFFFF)
        except ValueError:
            source_pixel_format = source.format & 0xFFFFFFFF
        timestamp = self._dll.fc2GetImageTimeStamp(ct.byref(source))
        native_metadata = _ImageMetadata()
        self._check(
            "fc2GetImageMetadata",
            self._dll.fc2GetImageMetadata(ct.byref(source), ct.byref(native_metadata)),
        )
        current_counter = native_metadata.embeddedFrameCounter
        dropped_frames = None
        if (
            self._last_embedded_counter is not None
            and current_counter != self._last_embedded_counter
        ):
            delta = (current_counter - self._last_embedded_counter) & 0xFFFFFFFF
            dropped_frames = max(0, delta - 1)
        self._last_embedded_counter = current_counter
        self._sequence += 1
        return Frame(
            data=data,
            pixel_format=pixel_format,
            rows=image.rows,
            columns=image.cols,
            stride=image.stride,
            received_bytes=source.receivedDataSize,
            source_pixel_format=source_pixel_format,
            sequence=self._sequence,
            timestamp=FrameTimestamp(
                seconds=timestamp.seconds,
                microseconds=timestamp.microSeconds,
                cycle_seconds=timestamp.cycleSeconds,
                cycle_count=timestamp.cycleCount,
                cycle_offset=timestamp.cycleOffset,
            ),
            metadata=ImageMetadata(
                timestamp=native_metadata.embeddedTimeStamp,
                gain=native_metadata.embeddedGain,
                shutter=native_metadata.embeddedShutter,
                brightness=native_metadata.embeddedBrightness,
                exposure=native_metadata.embeddedExposure,
                white_balance=native_metadata.embeddedWhiteBalance,
                frame_counter=current_counter,
                strobe_pattern=native_metadata.embeddedStrobePattern,
                gpio_pin_state=native_metadata.embeddedGPIOPinState,
                roi_position=native_metadata.embeddedROIPosition,
            ),
            dropped_frames=dropped_frames,
        )

    def get_capture_config(self) -> CaptureConfig:
        config = _Config()
        self._check(
            "fc2GetConfiguration", self._dll.fc2GetConfiguration(self._context, ct.byref(config))
        )
        return CaptureConfig(
            num_buffers=config.numBuffers,
            timeout_ms=config.grabTimeout,
            grab_mode=GrabMode(config.grabMode),
            high_performance=bool(config.highPerformanceRetrieveBuffer),
        )

    def configure_capture(
        self,
        *,
        timeout_ms: int | None,
        num_buffers: int | None,
        grab_mode: GrabMode | None,
        high_performance: bool | None,
    ) -> CaptureConfig:
        config = _Config()
        self._check(
            "fc2GetConfiguration", self._dll.fc2GetConfiguration(self._context, ct.byref(config))
        )
        if timeout_ms is not None:
            config.grabTimeout = timeout_ms
        if num_buffers is not None:
            config.numBuffers = num_buffers
        if grab_mode is not None:
            config.grabMode = int(grab_mode)
        if high_performance is not None:
            config.highPerformanceRetrieveBuffer = high_performance
        self._check(
            "fc2SetConfiguration", self._dll.fc2SetConfiguration(self._context, ct.byref(config))
        )
        return self.get_capture_config()

    def get_format7_capabilities(self, mode: int) -> Format7Capabilities:
        info = _Format7Info(mode=mode)
        supported = ct.c_int()
        self._check(
            "fc2GetFormat7Info",
            self._dll.fc2GetFormat7Info(self._context, ct.byref(info), ct.byref(supported)),
        )
        return Format7Capabilities(
            mode=mode,
            supported=bool(supported.value),
            max_width=info.maxWidth,
            max_height=info.maxHeight,
            offset_x_step=info.offsetHStepSize,
            offset_y_step=info.offsetVStepSize,
            width_step=info.imageHStepSize,
            height_step=info.imageVStepSize,
            pixel_format_bits=info.pixelFormatBitField,
        )

    def configure_format7(
        self,
        *,
        width: int,
        height: int,
        pixel_format: PixelFormat,
        offset_x: int,
        offset_y: int,
        mode: int,
        packet_percent: float,
    ) -> None:
        settings = _Format7ImageSettings(
            mode=mode,
            offsetX=offset_x,
            offsetY=offset_y,
            width=width,
            height=height,
            pixelFormat=int(pixel_format),
        )
        valid = ct.c_int()
        packet = _Format7PacketInfo()
        self._check(
            "fc2ValidateFormat7Settings",
            self._dll.fc2ValidateFormat7Settings(
                self._context, ct.byref(settings), ct.byref(valid), ct.byref(packet)
            ),
        )
        if not valid.value:
            raise ValueError("The camera rejected the requested Format7 settings")
        self._check(
            "fc2SetFormat7Configuration",
            self._dll.fc2SetFormat7Configuration(
                self._context, ct.byref(settings), ct.c_float(packet_percent)
            ),
        )

    def get_video_mode_and_frame_rate(self) -> tuple[int, int]:
        video_mode = ct.c_int()
        frame_rate = ct.c_int()
        self._check(
            "fc2GetVideoModeAndFrameRate",
            self._dll.fc2GetVideoModeAndFrameRate(
                self._context, ct.byref(video_mode), ct.byref(frame_rate)
            ),
        )
        return video_mode.value, frame_rate.value

    def set_video_mode_and_frame_rate(self, video_mode: int, frame_rate: int) -> None:
        self._check(
            "fc2SetVideoModeAndFrameRate",
            self._dll.fc2SetVideoModeAndFrameRate(self._context, video_mode, frame_rate),
        )

    @staticmethod
    def _embedded_config(info: _EmbeddedImageInfo) -> EmbeddedMetadataConfig:
        def value(prop: _EmbeddedProperty) -> EmbeddedProperty:
            return EmbeddedProperty(available=bool(prop.available), enabled=bool(prop.onOff))

        return EmbeddedMetadataConfig(
            timestamp=value(info.timestamp),
            gain=value(info.gain),
            shutter=value(info.shutter),
            brightness=value(info.brightness),
            exposure=value(info.exposure),
            white_balance=value(info.whiteBalance),
            frame_counter=value(info.frameCounter),
            strobe_pattern=value(info.strobePattern),
            gpio_pin_state=value(info.GPIOPinState),
            roi_position=value(info.ROIPosition),
        )

    def get_embedded_metadata_config(self) -> EmbeddedMetadataConfig:
        info = _EmbeddedImageInfo()
        self._check(
            "fc2GetEmbeddedImageInfo",
            self._dll.fc2GetEmbeddedImageInfo(self._context, ct.byref(info)),
        )
        return self._embedded_config(info)

    def configure_embedded_metadata(self, **changes: bool | None) -> EmbeddedMetadataConfig:
        info = _EmbeddedImageInfo()
        self._check(
            "fc2GetEmbeddedImageInfo",
            self._dll.fc2GetEmbeddedImageInfo(self._context, ct.byref(info)),
        )
        names = {
            "timestamp": "timestamp",
            "gain": "gain",
            "shutter": "shutter",
            "brightness": "brightness",
            "exposure": "exposure",
            "white_balance": "whiteBalance",
            "frame_counter": "frameCounter",
            "strobe_pattern": "strobePattern",
            "gpio_pin_state": "GPIOPinState",
            "roi_position": "ROIPosition",
        }
        for public_name, native_name in names.items():
            requested = changes.get(public_name)
            prop = getattr(info, native_name)
            if requested is not None:
                if requested and not prop.available:
                    raise ValueError(f"Embedded {public_name} metadata is not available")
                prop.onOff = requested
        self._check(
            "fc2SetEmbeddedImageInfo",
            self._dll.fc2SetEmbeddedImageInfo(self._context, ct.byref(info)),
        )
        return self.get_embedded_metadata_config()

    @staticmethod
    def _shape_frame(
        packed: np.ndarray[Any, np.dtype[np.uint8]], rows: int, cols: int, pixel_format: int
    ) -> np.ndarray[Any, Any]:
        fmt = pixel_format & 0xFFFFFFFF
        if fmt in (PixelFormat.MONO8, PixelFormat.RAW8):
            return packed[:, :cols].copy()
        if fmt in (PixelFormat.MONO16, PixelFormat.RAW16, PixelFormat.MONO12, PixelFormat.RAW12):
            return packed[:, : cols * 2].copy().view("<u2").reshape(rows, cols)
        if fmt in (PixelFormat.RGB8, PixelFormat.BGR8):
            return packed[:, : cols * 3].copy().reshape(rows, cols, 3)
        if fmt == PixelFormat.YUV422:
            return packed[:, : cols * 2].copy().reshape(rows, cols, 2)
        if fmt == PixelFormat.YUV444:
            return packed[:, : cols * 3].copy().reshape(rows, cols, 3)
        if fmt in (PixelFormat.RGBU8, PixelFormat.BGRU8):
            return packed[:, : cols * 4].copy().reshape(rows, cols, 4)
        if fmt in (PixelFormat.RGB16, PixelFormat.BGR16):
            return packed[:, : cols * 6].copy().view("<u2").reshape(rows, cols, 3)
        if fmt == PixelFormat.BGRU16:
            return packed[:, : cols * 8].copy().view("<u2").reshape(rows, cols, 4)
        return packed.copy()

    def get_property(self, property_type: PropertyType) -> CameraProperty:
        prop = _Property(type=int(property_type))
        self._check("fc2GetProperty", self._dll.fc2GetProperty(self._context, ct.byref(prop)))
        return CameraProperty(
            type=property_type,
            present=bool(prop.present),
            absolute=bool(prop.absControl),
            one_push=bool(prop.onePush),
            auto=bool(prop.autoManualMode),
            enabled=bool(prop.onOff),
            value_a=prop.valueA,
            value_b=prop.valueB,
            value=prop.absValue,
        )

    def get_property_info(self, property_type: PropertyType) -> CameraPropertyInfo:
        info = _PropertyInfo(type=int(property_type))
        self._check(
            "fc2GetPropertyInfo",
            self._dll.fc2GetPropertyInfo(self._context, ct.byref(info)),
        )
        return CameraPropertyInfo(
            type=property_type,
            present=bool(info.present),
            auto_supported=bool(info.autoSupported),
            manual_supported=bool(info.manualSupported),
            on_off_supported=bool(info.onOffSupported),
            one_push_supported=bool(info.onePushSupported),
            absolute_supported=bool(info.absValSupported),
            readable=bool(info.readOutSupported),
            minimum=info.min,
            maximum=info.max,
            absolute_minimum=info.absMin,
            absolute_maximum=info.absMax,
            units=_decode(info.pUnits),
            unit_abbreviation=_decode(info.pUnitAbbr),
        )

    def set_property(
        self,
        property_type: PropertyType,
        value: float | int,
        *,
        absolute: bool,
        auto: bool,
        enabled: bool,
        value_b: int | None,
        one_push: bool | None,
    ) -> None:
        prop = _Property(type=int(property_type))
        self._check("fc2GetProperty", self._dll.fc2GetProperty(self._context, ct.byref(prop)))
        prop.absControl = absolute
        prop.autoManualMode = auto
        prop.onOff = enabled
        if one_push is not None:
            prop.onePush = one_push
        if absolute:
            prop.absValue = float(value)
        else:
            prop.valueA = int(value)
            if value_b is not None:
                prop.valueB = value_b
        self._check("fc2SetProperty", self._dll.fc2SetProperty(self._context, ct.byref(prop)))

    def get_trigger(self) -> TriggerMode:
        trigger = _TriggerMode()
        self._check(
            "fc2GetTriggerMode", self._dll.fc2GetTriggerMode(self._context, ct.byref(trigger))
        )
        return TriggerMode(
            enabled=bool(trigger.onOff),
            polarity=trigger.polarity,
            source=trigger.source,
            mode=trigger.mode,
            parameter=trigger.parameter,
        )

    def set_trigger(self, trigger: TriggerMode) -> None:
        native = _TriggerMode(
            onOff=trigger.enabled,
            polarity=trigger.polarity,
            source=trigger.source,
            mode=trigger.mode,
            parameter=trigger.parameter,
        )
        self._check(
            "fc2SetTriggerMode", self._dll.fc2SetTriggerMode(self._context, ct.byref(native))
        )

    def fire_software_trigger(self) -> None:
        self._check(
            "fc2FireSoftwareTrigger", self._dll.fc2FireSoftwareTrigger(self._context)
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._image_created:
            self._dll.fc2DestroyImage(ct.byref(self._image))
            self._image_created = False
        if self._converted_image_created:
            self._dll.fc2DestroyImage(ct.byref(self._converted_image))
            self._converted_image_created = False
        if self._context:
            self._dll.fc2DestroyContext(self._context)
            self._context = ct.c_void_p()
        if self._dll_dir is not None:
            self._dll_dir.close()
            self._dll_dir = None
        self._closed = True
