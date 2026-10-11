"""Observable failures shared by S1-04 and S1-05."""


class Pose3DError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
