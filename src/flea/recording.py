from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .session import CaptureSession
from .types import Frame, PixelFormat


class RecordingFormat(str, Enum):
    NPY = "npy"
    PNG = "png"
    TIFF = "tiff"


class Recorder:
    """Write a self-describing image collection into an atomic directory.

    Data is first written to a sibling ``.partial-*`` directory. A successful
    close renames it to the requested destination. Failed recordings leave the
    partial directory intact for diagnosis or recovery.
    """

    def __init__(
        self,
        destination: str | os.PathLike[str],
        *,
        format: RecordingFormat | str = RecordingFormat.NPY,
    ) -> None:
        self.destination = Path(destination)
        self.format = RecordingFormat(format)
        token = uuid.uuid4().hex[:8]
        self.partial_path = self.destination.with_name(
            f".{self.destination.name}.partial-{token}"
        )
        self._metadata_file: Any | None = None
        self._started_at: str | None = None
        self._closed = False
        self.frames_written = 0

    def open(self) -> Recorder:
        if self._metadata_file is not None:
            raise RuntimeError("Recorder is already open")
        if self.destination.exists():
            raise FileExistsError(f"Recording destination already exists: {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path.mkdir()
        self._metadata_file = (self.partial_path / "frames.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        )
        self._started_at = datetime.now(timezone.utc).isoformat()
        return self

    def write(self, frame: Frame) -> Path:
        if self._metadata_file is None or self._closed:
            raise RuntimeError("Recorder is not open")
        filename = f"frame_{frame.sequence:08d}.{self.format.value}"
        path = self.partial_path / filename
        if self.format is RecordingFormat.NPY:
            np.save(path, frame.data, allow_pickle=False)
        else:
            self._write_image(path, frame)
        self._metadata_file.write(json.dumps(self._frame_record(frame, filename)) + "\n")
        self._metadata_file.flush()
        self.frames_written += 1
        return path

    def _write_image(self, path: Path, frame: Frame) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "PNG/TIFF recording requires the 'opencv' extra: "
                "py -m pip install -e '.[opencv]'"
            ) from exc

        data = frame.data
        if data.ndim == 3 and data.shape[2] == 2:
            raise ValueError("Packed two-channel frames must be converted to bgr or saved as NPY")
        if frame.pixel_format is PixelFormat.RGB8:
            data = data[..., ::-1]
        elif frame.pixel_format is PixelFormat.RGBU8:
            data = data[..., [2, 1, 0, 3]]
        if not cv2.imwrite(str(path), data):
            raise OSError(f"OpenCV failed to write {path}")

    @staticmethod
    def _format_name(value: PixelFormat | int) -> str:
        return value.name if isinstance(value, PixelFormat) else f"0x{value:08x}"

    def _frame_record(self, frame: Frame, filename: str) -> dict[str, Any]:
        return {
            "file": filename,
            "sequence": frame.sequence,
            "width": frame.width,
            "height": frame.height,
            "shape": list(frame.data.shape),
            "dtype": str(frame.data.dtype),
            "pixel_format": self._format_name(frame.pixel_format),
            "source_pixel_format": self._format_name(frame.source_pixel_format),
            "stride": frame.stride,
            "received_bytes": frame.received_bytes,
            "dropped_frames": frame.dropped_frames,
            "timestamp": asdict(frame.timestamp),
            "metadata": asdict(frame.metadata),
        }

    def record(self, session: CaptureSession, *, count: int) -> int:
        if count < 1:
            raise ValueError("count must be at least 1")
        for _ in range(count):
            self.write(next(session))
        return self.frames_written

    def close(self, *, success: bool = True) -> Path:
        if self._closed:
            return self.destination if self.destination.exists() else self.partial_path
        if self._metadata_file is None:
            raise RuntimeError("Recorder is not open")
        self._metadata_file.close()
        self._metadata_file = None
        manifest = {
            "status": "complete" if success else "incomplete",
            "format": self.format.value,
            "frames": self.frames_written,
            "started_at": self._started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.partial_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if success:
            if self.destination.exists():
                raise FileExistsError(
                    f"Recording destination appeared while writing: {self.destination}"
                )
            self.partial_path.replace(self.destination)
        self._closed = True
        return self.destination if success else self.partial_path

    def __enter__(self) -> Recorder:
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close(success=exc_type is None)
