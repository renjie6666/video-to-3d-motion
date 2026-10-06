"""Typed errors returned by the S1-02 pipeline."""


class Pose2DError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Pose2DCancelled(Pose2DError):
    def __init__(self) -> None:
        super().__init__("cancelled", "2D pose processing was cancelled")
