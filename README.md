# flea

Python interface for legacy Point Grey Research / FLIR cameras using the
FlyCapture2 C API. The initial target is the Flea3 FL3-U3-32S2C on Windows with
FlyCapture2 2.13.3.61.

The package uses `ctypes`, so it does not require a compiler or an old binary
Python binding. Frames are copied into NumPy arrays before FlyCapture2 reuses
its capture buffer.

## Requirements

- 64-bit Python 3.10 or newer
- NumPy
- The 64-bit FlyCapture2 SDK and camera driver

Install the optional OpenCV integration dependency with:

```powershell
py -m pip install -e ".[opencv]"
```

On a default Windows installation, the library is found at:

```text
C:\Program Files\Point Grey Research\FlyCapture2\bin64\FlyCapture2_C_v100.dll
```

For a nonstandard installation, set `FLYCAPTURE2_LIBRARY` to the full native
library path or pass `library_path=` to `Camera` or `list_cameras`.

## Quick start

```python
from flea import Camera, PropertyType, list_cameras

print(list_cameras())

with Camera(index=0) as camera:
    print(camera.info)

    # FlyCapture2 reports shutter in milliseconds when absolute mode is used.
    camera.set_property(PropertyType.SHUTTER, 5.0)
    camera.set_property(PropertyType.GAIN, 0.0)

    camera.start()
    frame = camera.grab(format="bgr")
    print(frame.data.shape, frame.data.dtype, frame.pixel_format)
```

Native frames are available unchanged, or FlyCapture2 can convert them before
they are copied into NumPy:

```python
native = camera.grab()
bgr = camera.grab(format="bgr")   # OpenCV channel order
rgb = camera.grab(format="rgb")
gray = camera.grab(format="mono8")
```

Supported conversion names are `bgr`, `rgb`, `mono8`, `raw8`, and `native`.
The SDK performs conversion, including demosaicing when applicable.
The returned BGR array can be passed directly to OpenCV:

```python
import cv2

with Camera() as camera:
    camera.start()
    for frame in camera.frames(format="bgr"):
        cv2.imshow("Flea", frame.data)
        if cv2.waitKey(1) == 27:
            break
cv2.destroyAllWindows()
```

## Streaming and receive configuration

```python
from flea import Camera, GrabMode

with Camera() as camera:
    camera.configure_capture(
        timeout_ms=2_000,
        num_buffers=8,
        grab_mode=GrabMode.BUFFERED,  # preserve frames for collection
    )
    camera.start()

    for frame in camera.frames(format="bgr", count=100):
        consume(frame.data, frame.timestamp, frame.metadata)
```

Use `GrabMode.LATEST` for low-latency display and `GrabMode.BUFFERED` when every
frame matters. A finite timeout prevents a disconnected or stalled camera from
blocking forever.

For processing that may be slower than the camera, use a background session:

```python
from flea import Camera, CaptureSession

with Camera() as camera:
    with CaptureSession(
        camera,
        format="bgr",
        queue_size=16,
        policy="latest",  # or "lossless"
    ) as session:
        for frame in session:
            process(frame.data)

        print(session.stats())
```

`latest` discards the oldest queued frame to minimize latency. `lossless`
applies FlyCapture2's buffered grab mode and back-pressures acquisition when
the Python queue is full. The session temporarily applies a finite SDK timeout
so shutdown cannot hang, and restores the original camera configuration when
it closes. Worker failures are re-raised in the consuming thread.
`CaptureStats` reports acquisition FPS, queue depth and drops, SDK timeouts,
and current/mean capture-to-consumer latency.

Downstream libraries can type against the small `FrameSource` protocol instead
of depending on FlyCapture2-specific configuration APIs.

## Recording datasets

Recordings are written to a temporary sibling directory and atomically renamed
only after successful completion. Each dataset contains numbered images,
`frames.jsonl` with per-frame metadata, and a `manifest.json`. Interrupted
recordings retain their `.partial-*` directory for recovery.

```python
from flea import Camera, CaptureSession, Recorder

with Camera() as camera:
    with CaptureSession(camera, format="bgr", policy="lossless") as session:
        with Recorder("captures/run-001", format="png") as recorder:
            recorder.record(session, count=100)
```

`npy` works with the core NumPy dependency and preserves any frame layout.
`png` and `tiff` require the OpenCV extra and expect BGR, RGB, or grayscale
frames.

## Commands

```powershell
flea-info
flea-info --json
flea-preview
flea-capture captures/run-001 --count 100 --file-format npy
flea-capture captures/run-002 --count 100 --file-format png --output-format bgr
```

## ROI and native pixel format

Flea cameras expose custom image settings through FlyCapture2 Format7:

```python
from flea import Camera, PixelFormat

with Camera() as camera:
    print(camera.get_format7_capabilities(mode=0))
    camera.configure_format7(
        width=1600,
        height=1200,
        offset_x=0,
        offset_y=0,
        pixel_format=PixelFormat.RAW8,
    )
```

The wrapper checks sensor bounds, alignment steps, and supported pixel formats
before asking the SDK to validate and apply the settings.

## Frame metadata

Every `Frame` includes an SDK timestamp, host-side sequence number, native and
converted pixel formats, received byte count, and embedded camera metadata.
Enable the camera counter when dropped-frame detection is needed:

```python
with Camera() as camera:
    camera.configure_embedded_metadata(timestamp=True, frame_counter=True)
    camera.start()
    for frame in camera.frames(count=10):
        print(frame.sequence, frame.metadata.frame_counter, frame.dropped_frames)
```

Unsupported embedded fields are reported through
`get_embedded_metadata_config()` and rejected if enabled.

## Triggering

```python
with Camera() as camera:
    camera.configure_trigger(enabled=True, source=7, mode=0)  # software source
    camera.start()
    camera.fire_software_trigger()
    frame = camera.grab()
```

Trigger source and mode values are camera-specific; consult the camera's
technical reference before enabling an external trigger.

## Development

```powershell
py -m pip install -e ".[dev]"
py -m pytest
```
