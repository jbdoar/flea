from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum

from .camera import Camera
from .errors import CameraStateError, FlyCapture2Error
from .types import CaptureConfig, Frame, GrabMode, PixelFormat, PropertyType

_SDK_TIMEOUT = 18


class QueuePolicy(str, Enum):
    """Behavior when processing cannot keep up with acquisition."""

    LATEST = "latest"
    LOSSLESS = "lossless"


@dataclass(frozen=True, slots=True)
class CaptureStats:
    frames_acquired: int
    frames_delivered: int
    frames_dropped: int
    grab_timeouts: int
    queue_depth: int
    elapsed_seconds: float
    acquisition_fps: float
    latest_latency_ms: float | None
    mean_latency_ms: float | None


class CaptureSessionError(RuntimeError):
    """The background acquisition worker failed."""


class CaptureSession(Iterator[Frame]):
    """Acquire frames on a worker thread into a bounded queue.

    The camera must be connected and stopped. The session temporarily applies
    a finite SDK grab timeout and a buffer mode matching ``policy``, then
    restores the original receive configuration when it closes.
    """

    def __init__(
        self,
        camera: Camera,
        *,
        format: str | PixelFormat | None = None,
        queue_size: int = 16,
        policy: QueuePolicy | str = QueuePolicy.LATEST,
        grab_timeout_ms: int = 500,
        on_frame: Callable[[Frame], None] | None = None,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        if grab_timeout_ms < 1:
            raise ValueError("grab_timeout_ms must be positive")
        self.camera = camera
        self.format = format
        self.queue_size = queue_size
        self.policy = QueuePolicy(policy)
        self.grab_timeout_ms = grab_timeout_ms
        self.on_frame = on_frame
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._original_config: CaptureConfig | None = None
        self._started_at = 0.0
        self._stopped_at = 0.0
        self._frames_acquired = 0
        self._frames_delivered = 0
        self._frames_dropped = 0
        self._grab_timeouts = 0
        self._latest_latency_ms: float | None = None
        self._latency_total_ms = 0.0
        self._latency_samples = 0
        self._lock = threading.Lock()
        self._property_lock = threading.Lock()
        self._pending_properties: dict[
            PropertyType, tuple[float | int, bool, bool, bool, int | None]
        ] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> CaptureSession:
        if self._thread is not None:
            raise CameraStateError("CaptureSession cannot be started more than once")
        if not self.camera.connected:
            raise CameraStateError("Connect the camera before starting a CaptureSession")
        if self.camera.capturing:
            raise CameraStateError("Camera must be stopped before starting a CaptureSession")

        self._original_config = self.camera.get_capture_config()
        grab_mode = (
            GrabMode.BUFFERED if self.policy is QueuePolicy.LOSSLESS else GrabMode.LATEST
        )
        num_buffers = max(self._original_config.num_buffers, min(self.queue_size, 64))
        self.camera.configure_capture(
            timeout_ms=self.grab_timeout_ms,
            num_buffers=num_buffers,
            grab_mode=grab_mode,
        )
        try:
            self.camera.start()
        except BaseException:
            self._restore_config()
            raise

        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name=f"flea-capture-{self.camera.index}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._apply_pending_properties()
                try:
                    frame = self.camera.grab(format=self.format)
                except FlyCapture2Error as exc:
                    if exc.code == _SDK_TIMEOUT:
                        with self._lock:
                            self._grab_timeouts += 1
                        continue
                    raise
                if self.on_frame is not None:
                    self.on_frame(frame)
                latency_ms = None
                if frame.timestamp.seconds > 0:
                    latency_ms = (time.time() - frame.timestamp.unix_seconds) * 1_000
                with self._lock:
                    self._frames_acquired += 1
                    if latency_ms is not None:
                        self._latest_latency_ms = latency_ms
                        self._latency_total_ms += latency_ms
                        self._latency_samples += 1
                self._enqueue(frame)
        except BaseException as exc:
            self._error = exc
            self._stop.set()
        finally:
            self._stopped_at = time.perf_counter()

    def set_property(
        self,
        property_type: PropertyType,
        value: float | int,
        *,
        absolute: bool = True,
        auto: bool = False,
        enabled: bool = True,
        value_b: int | None = None,
    ) -> None:
        """Queue a property write for the acquisition thread.

        Repeated updates to the same property are coalesced so a responsive UI
        cannot build an unbounded command backlog.
        """
        if self._thread is None:
            raise CameraStateError("CaptureSession has not been started")
        with self._property_lock:
            self._pending_properties[PropertyType(property_type)] = (
                value,
                absolute,
                auto,
                enabled,
                value_b,
            )

    def _apply_pending_properties(self) -> None:
        with self._property_lock:
            pending = self._pending_properties
            self._pending_properties = {}
        for property_type, values in pending.items():
            value, absolute, auto, enabled, value_b = values
            self.camera.set_property(
                property_type,
                value,
                absolute=absolute,
                auto=auto,
                enabled=enabled,
                value_b=value_b,
            )

    def _enqueue(self, frame: Frame) -> None:
        if self.policy is QueuePolicy.LATEST:
            while not self._stop.is_set():
                try:
                    self._queue.put_nowait(frame)
                    return
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        continue
                    with self._lock:
                        self._frames_dropped += 1
            return

        while not self._stop.is_set():
            try:
                self._queue.put(frame, timeout=0.05)
                return
            except queue.Full:
                continue

    def get(self, timeout: float | None = None) -> Frame:
        """Return the next frame, propagating worker failures to the consumer."""
        if self._thread is None:
            raise CameraStateError("CaptureSession has not been started")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._error is not None and self._queue.empty():
                raise CaptureSessionError("Background camera acquisition failed") from self._error
            if not self.running and self._queue.empty():
                raise StopIteration
            wait = 0.05
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a captured frame")
                wait = min(wait, remaining)
            try:
                frame = self._queue.get(timeout=wait)
            except queue.Empty:
                continue
            with self._lock:
                self._frames_delivered += 1
            return frame

    def stats(self) -> CaptureStats:
        now = self._stopped_at or time.perf_counter()
        elapsed = max(0.0, now - self._started_at) if self._started_at else 0.0
        with self._lock:
            acquired = self._frames_acquired
            return CaptureStats(
                frames_acquired=acquired,
                frames_delivered=self._frames_delivered,
                frames_dropped=self._frames_dropped,
                grab_timeouts=self._grab_timeouts,
                queue_depth=self._queue.qsize(),
                elapsed_seconds=elapsed,
                acquisition_fps=acquired / elapsed if elapsed else 0.0,
                latest_latency_ms=self._latest_latency_ms,
                mean_latency_ms=(
                    self._latency_total_ms / self._latency_samples
                    if self._latency_samples
                    else None
                ),
            )

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.grab_timeout_ms / 1_000 + 1.0))
        if self._thread.is_alive():
            raise CaptureSessionError("Acquisition thread did not stop after the SDK timeout")
        try:
            self.camera.stop()
        finally:
            self._restore_config()

    def _restore_config(self) -> None:
        if self._original_config is None:
            return
        config = self._original_config
        self.camera.configure_capture(
            timeout_ms=config.timeout_ms,
            num_buffers=config.num_buffers,
            grab_mode=config.grab_mode,
            high_performance=config.high_performance,
        )
        self._original_config = None

    def __iter__(self) -> CaptureSession:
        return self

    def __next__(self) -> Frame:
        return self.get()

    def __enter__(self) -> CaptureSession:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
