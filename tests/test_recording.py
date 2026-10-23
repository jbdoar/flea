from __future__ import annotations

import json

import numpy as np
import pytest

from flea import Frame, FrameTimestamp, ImageMetadata, PixelFormat, Recorder


def make_frame(sequence: int = 1) -> Frame:
    return Frame(
        data=np.arange(12, dtype=np.uint8).reshape(3, 4),
        pixel_format=PixelFormat.MONO8,
        rows=3,
        columns=4,
        stride=4,
        received_bytes=12,
        source_pixel_format=PixelFormat.MONO8,
        sequence=sequence,
        timestamp=FrameTimestamp(10, 20, 1, 2, 3),
        metadata=ImageMetadata(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    )


def test_recorder_publishes_atomic_npy_dataset(tmp_path) -> None:
    destination = tmp_path / "run-001"
    with Recorder(destination) as recorder:
        partial = recorder.partial_path
        recorder.write(make_frame())
        assert partial.exists()
        assert not destination.exists()

    assert destination.is_dir()
    np.testing.assert_array_equal(
        np.load(destination / "frame_00000001.npy"), make_frame().data
    )
    record = json.loads((destination / "frames.jsonl").read_text().strip())
    assert record["sequence"] == 1
    assert record["pixel_format"] == "MONO8"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["frames"] == 1


def test_recorder_refuses_existing_destination(tmp_path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        Recorder(destination).open()


def test_failed_recording_preserves_partial_directory(tmp_path) -> None:
    recorder = Recorder(tmp_path / "failed")
    partial = recorder.partial_path
    with pytest.raises(RuntimeError):
        with recorder:
            recorder.write(make_frame())
            raise RuntimeError("processing failed")
    assert partial.is_dir()
    manifest = json.loads((partial / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"

