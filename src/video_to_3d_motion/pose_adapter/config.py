"""Explicit configuration for the 243-frame baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import PoseAdapterError


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    clip_len: int = 243
    joint_schema: str = "h36m17"
    joint_profile: str = "rtmpose_l_h36m_v1"
    normalization: str = "h36m_width_center_v1"
    low_confidence_policy: str = "keep"
    short_clip_policy: str = "error"

    def __post_init__(self) -> None:
        if type(self.clip_len) is not int or self.clip_len != 243:
            raise ValueError("the verified baseline requires clip_len=243")
        for name, expected in (
            ("joint_schema", "h36m17"), ("joint_profile", "rtmpose_l_h36m_v1"),
            ("normalization", "h36m_width_center_v1"),
            ("low_confidence_policy", "keep"), ("short_clip_policy", "error"),
        ):
            if getattr(self, name) != expected:
                raise ValueError(f"unsupported {name}: {getattr(self, name)}")


@dataclass(frozen=True, slots=True)
class AdapterQueueSettings:
    window_output_capacity: int = 2
    get_timeout_seconds: float = 0.1
    put_timeout_seconds: float = 0.5

    def __post_init__(self) -> None:
        if type(self.window_output_capacity) is not int or self.window_output_capacity <= 0:
            raise ValueError("window_output_capacity must be a positive integer")
        for value in (self.get_timeout_seconds, self.put_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("queue timeouts must be finite and positive")


@dataclass(frozen=True, slots=True)
class ArchiveSettings:
    enabled: bool = True

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("archive.enabled must be boolean")


@dataclass(frozen=True, slots=True)
class PoseAdapterConfig:
    pipeline_mode: str = "baseline"
    pose_adapter: AdapterSettings = field(default_factory=AdapterSettings)
    queue: AdapterQueueSettings = field(default_factory=AdapterQueueSettings)
    archive: ArchiveSettings = field(default_factory=ArchiveSettings)

    def __post_init__(self) -> None:
        if self.pipeline_mode != "baseline":
            raise PoseAdapterError("mode_not_implemented", "only baseline is implemented")


def load_pose_adapter_config(path: str | Path) -> PoseAdapterConfig:
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict) or set(raw) - {"pipeline_mode", "pose_adapter", "queue", "archive"}:
        raise ValueError("unknown pose adapter configuration fields")
    return PoseAdapterConfig(
        pipeline_mode=raw.get("pipeline_mode", "baseline"),
        pose_adapter=AdapterSettings(**raw.get("pose_adapter", {})),
        queue=AdapterQueueSettings(**raw.get("queue", {})),
        archive=ArchiveSettings(**raw.get("archive", {})),
    )
