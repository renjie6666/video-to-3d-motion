"""Errors shared by replay, validation and the queue worker."""


class PoseAdapterError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PoseAdapterCancelled(PoseAdapterError):
    def __init__(self) -> None:
        super().__init__("cancelled", "pose adaptation was cancelled")
