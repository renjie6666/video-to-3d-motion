"""Typed errors returned by the media stage."""


class MediaPipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ProducerCancelled(MediaPipelineError):
    def __init__(self) -> None:
        super().__init__("cancelled", "frame production was cancelled")
