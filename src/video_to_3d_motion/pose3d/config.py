"""Validated, file-relative settings for the released 243-frame L model."""

from dataclasses import dataclass, field
import math
from pathlib import Path

import yaml

UPSTREAM_REVISION = "df38d599212c058259f473464baec66d6a6487e0"
WEIGHTS_REVISION = "732bd6d262b628cf2daf979bbd660a7d0974631a"
WEIGHTS_SHA256 = "031a033dbb4266f093d048e55758e55025c663f65db8d84291b6116794124501"


@dataclass(frozen=True, slots=True)
class ModelSettings:
    checkpoint: Path = Path("checkpoints/PoseMamba_L.bin")
    checkpoint_sha256: str = WEIGHTS_SHA256
    device: str = "cuda:0"
    flip_test: bool = True

    def __post_init__(self):
        if self.checkpoint_sha256 != WEIGHTS_SHA256:
            raise ValueError("only the verified official PoseMamba-L checkpoint is supported")
        if self.device != "cpu" and self.device not in ("cuda", "cuda:0"):
            raise ValueError("device must be cpu or cuda:0")
        if type(self.flip_test) is not bool:
            raise ValueError("flip_test must be boolean")


@dataclass(frozen=True, slots=True)
class OutputSettings:
    obs_len: int = 10
    smoothing: str = "none"
    tau_seconds: float = 0.04

    def __post_init__(self):
        if type(self.obs_len) is not int or self.obs_len != 10:
            raise ValueError("obs_len must be 10")
        if self.smoothing not in ("none", "ema"):
            raise ValueError("smoothing must be none or ema")
        if (type(self.tau_seconds) not in (int, float)
                or not math.isfinite(self.tau_seconds) or self.tau_seconds <= 0):
            raise ValueError("tau_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    annotation_h5: Path | None = None
    annotation_names: Path | None = None
    sequence: str | None = None

    def __post_init__(self):
        supplied = [self.annotation_h5 is not None, self.annotation_names is not None, self.sequence is not None]
        if any(supplied) and not all(supplied):
            raise ValueError("evaluation requires annotation_h5, annotation_names and sequence together")


@dataclass(frozen=True, slots=True)
class Pose3DConfig:
    pipeline_mode: str = "baseline"
    clip_len: int = 243
    window_stride: int = 81
    model: ModelSettings = field(default_factory=ModelSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)

    def __post_init__(self):
        if self.pipeline_mode not in ("baseline", "full"):
            raise ValueError("pipeline_mode must be baseline or full")
        if type(self.clip_len) is not int or self.clip_len != 243:
            raise ValueError("the released model requires clip_len=243")
        if type(self.window_stride) is not int or self.window_stride != 81:
            raise ValueError("the current full-sequence protocol uses window_stride=81")


def load_pose3d_config(path: str | Path) -> Pose3DConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or set(raw) - {"pipeline_mode", "clip_len", "window_stride", "model", "output", "evaluation"}:
        raise ValueError("unknown pose3d configuration fields")
    model = dict(raw.pop("model", {}))
    output = dict(raw.pop("output", {}))
    evaluation = dict(raw.pop("evaluation", {}))
    for section, names in ((model, ("checkpoint",)), (evaluation, ("annotation_h5", "annotation_names"))):
        for name in names:
            if section.get(name) is not None:
                section[name] = (path.parent / section[name]).resolve()
    return Pose3DConfig(**raw, model=ModelSettings(**model), output=OutputSettings(**output),
                        evaluation=EvaluationSettings(**evaluation))
