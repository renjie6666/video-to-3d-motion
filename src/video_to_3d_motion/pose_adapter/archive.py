"""Archive numeric windows and provenance without pickle or overwrites."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import yaml

from .config import PoseAdapterConfig
from .exceptions import PoseAdapterError
from .models import Pose2DWindow, PoseAdapterResult


class WindowArchive:
    def __init__(self, output_dir: str | Path, config: PoseAdapterConfig) -> None:
        self.path = Path(output_dir).resolve()
        self.config = config
        try:
            # Exclusive creation keeps a previous run's results intact.
            self.path.mkdir(parents=True, exist_ok=False)
            (self.path / "config_snapshot.yaml").write_text(
                yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
            if config.archive.enabled:
                (self.path / "windows").mkdir()
                (self.path / "windows.jsonl").touch()
        except OSError as exc:
            raise PoseAdapterError("archive_write_failed", str(exc)) from exc

    def emit(self, item) -> None:
        if not isinstance(item, Pose2DWindow) or not self.config.archive.enabled:
            return
        try:
            arrays = {}
            metadata = {}
            for entry in fields(item):
                value = getattr(item, entry.name)
                (arrays if isinstance(value, np.ndarray) else metadata)[entry.name] = value
            relative = f"windows/window_{item.window_index:06d}.npz"
            np.savez_compressed(self.path / relative, **arrays)
            metadata.update(file=relative, first_frame_id=int(item.frame_ids[0]),
                            last_frame_id=int(item.frame_ids[-1]),
                            first_timestamp=float(item.timestamps[0]), last_timestamp=float(item.timestamps[-1]))
            with (self.path / "windows.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False) + "\n")
        except (OSError, ValueError, TypeError) as exc:
            raise PoseAdapterError("archive_write_failed", str(exc)) from exc

    def finish(self, result: PoseAdapterResult, **provenance) -> Path:
        payload = asdict(result)
        count = result.metrics.frames_received * 17
        payload.update(provenance, low_confidence_ratio=result.metrics.low_confidence_points / count if count else 0,
                       archive_enabled=self.config.archive.enabled)
        path = self.path / "adapter_summary.json"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise PoseAdapterError("archive_write_failed", str(exc)) from exc
        return path
