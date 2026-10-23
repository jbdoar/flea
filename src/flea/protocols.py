from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from .types import Frame, PixelFormat


@runtime_checkable
class FrameSource(Protocol):
    """Minimal camera interface that downstream applications can target."""

    def grab(self, *, format: str | PixelFormat | None = None) -> Frame: ...

    def grab_array(
        self, *, format: str | PixelFormat | None = None
    ) -> NDArray: ...

