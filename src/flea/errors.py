class FleaError(RuntimeError):
    """Base error raised by this package."""


class FlyCapture2NotFoundError(FleaError):
    """The FlyCapture2 native library could not be loaded."""


class FlyCapture2Error(FleaError):
    """A FlyCapture2 SDK call failed."""

    def __init__(self, operation: str, code: int, description: str) -> None:
        self.operation = operation
        self.code = code
        self.description = description
        super().__init__(f"{operation} failed ({code}): {description}")


class CameraStateError(FleaError):
    """An operation was attempted in the wrong camera state."""

