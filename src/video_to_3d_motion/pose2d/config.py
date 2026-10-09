"""Configuration models for S1-02."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .models import JointSchema


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    bbox_source: Literal[
        "external_only", "detector_only", "external_or_detector"
    ] = "external_or_detector"
    score_threshold: float = 0.3
    nms_iou_threshold: float = 0.65
    min_bbox_area: float = 256.0
    strict_single_person: bool = True
    detector_config: Path | None = None
    detector_checkpoint: Path | None = None
    person_class_id: int = 0

    def __post_init__(self) -> None:
        if self.bbox_source not in {
            "external_only", "detector_only", "external_or_detector"
        }:
            raise ValueError("detection.bbox_source is invalid")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("detection.score_threshold must be in [0,1]")
        if not 0 <= self.nms_iou_threshold <= 1:
            raise ValueError("detection.nms_iou_threshold must be in [0,1]")
        if self.min_bbox_area <= 0:
            raise ValueError("detection.min_bbox_area must be positive")


@dataclass(frozen=True, slots=True)
class BatchConfig:
    max_size: int = 27
    prefetch_batches: int = 4
    input_queue_capacity: int = 108
    output_queue_capacity: int = 108
    put_timeout_seconds: float = 0.5

    def __post_init__(self) -> None:
        if min(
            self.max_size,
            self.prefetch_batches,
            self.input_queue_capacity,
            self.output_queue_capacity,
        ) <= 0:
            raise ValueError("batch sizes and queue capacities must be positive")
        if self.put_timeout_seconds <= 0:
            raise ValueError("batch.put_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    amp: bool = False
    keypoint_score_threshold: float = 0.2

    def __post_init__(self) -> None:
        if not 0 <= self.keypoint_score_threshold <= 1:
            raise ValueError("runtime.keypoint_score_threshold must be in [0,1]")


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    enabled: bool = True
    output_root: Path = Path("resources/pose2d_visualizations")
    image_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = 95
    draw_bbox: bool = True
    draw_skeleton: bool = True
    draw_joint_index: bool = False
    draw_frame_info: bool = True
    every_n_frames: int = 1
    strict: bool = True

    def __post_init__(self) -> None:
        if self.image_format not in {"jpg", "png"}:
            raise ValueError("visualization.image_format must be jpg or png")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("visualization.jpeg_quality must be in [1,100]")
        if self.every_n_frames <= 0:
            raise ValueError("visualization.every_n_frames must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "rtmpose_m"
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    joint_schema: JointSchema = "coco17"
    score_source: str = "rtmpose_simcc"


@dataclass(frozen=True, slots=True)
class Pose2DConfig:
    pipeline_mode: Literal["baseline", "full"] = "baseline"
    model: ModelConfig = ModelConfig()
    detection: DetectionConfig = DetectionConfig()
    batch: BatchConfig = BatchConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    def __post_init__(self) -> None:
        if self.pipeline_mode not in {"baseline", "full"}:
            raise ValueError("pipeline_mode must be baseline or full")
        if self.model.joint_schema not in {"coco17", "h36m17"}:
            raise ValueError("joint_schema must be coco17 or h36m17")
        # Baseline describes the experiment mode, not the checkpoint's joints.
        # A public H36M-finetuned checkpoint can also serve as the baseline.
        if self.pipeline_mode == "full" and self.model.joint_schema != "h36m17":
            raise ValueError("full requires joint_schema=h36m17")


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _selected_model(raw: dict[str, Any], mode: str, base_dir: Path) -> ModelConfig:
    pose_raw = dict(raw.get("pose2d", {}))
    name = str(pose_raw.get("active_model", "rtmpose_m"))
    models = dict(pose_raw.get("models", {}))
    selected = dict(models.get(name, {}))
    schema_raw = selected.get("joint_schema", {})
    schema = (
        schema_raw.get(mode)
        if isinstance(schema_raw, dict)
        else schema_raw
    ) or ("coco17" if mode == "baseline" else "h36m17")

    def mode_path(key: str) -> Path | None:
        value = selected.get(key)
        if isinstance(value, dict):
            value = value.get(mode)
        return _resolve_path(value, base_dir) if value else None

    return ModelConfig(
        name=name,
        config_path=mode_path("config"),
        checkpoint_path=mode_path("checkpoint"),
        joint_schema=schema,
        score_source=str(
            selected.get(
                "score_source",
                "unknown",
            )
        ),
    )


def load_pose2d_config(path: str | Path) -> Pose2DConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load pose2d config") from exc
    config_path = Path(path).resolve()
    base_dir = config_path.parent
    with config_path.open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}

    mode = str(raw.get("pipeline_mode", "baseline"))
    runtime = RuntimeConfig(**raw.get("runtime", {}))
    detection_raw = dict(raw.get("detection", {}))
    for key in ("detector_config", "detector_checkpoint"):
        if detection_raw.get(key):
            detection_raw[key] = _resolve_path(detection_raw[key], base_dir)
    visualization_raw = dict(raw.get("visualization", {}))
    visualization_raw["output_root"] = _resolve_path(
        visualization_raw.get("output_root", "../results/visualizations/pose2d"), base_dir
    )
    return Pose2DConfig(
        pipeline_mode=mode,
        model=_selected_model(raw, mode, base_dir),
        detection=DetectionConfig(**detection_raw),
        batch=BatchConfig(**raw.get("batch", {})),
        runtime=runtime,
        visualization=VisualizationConfig(**visualization_raw),
    )
