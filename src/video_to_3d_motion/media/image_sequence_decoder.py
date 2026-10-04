"""Decode a numerically ordered image directory as a frame source."""

from __future__ import annotations

import io
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import ImageSequenceConfig
from .exceptions import MediaPipelineError
from .models import DecodedFrame, VideoInfo

ImageArray = NDArray[np.uint8]
ImageLoader = Callable[[Path], ImageArray]


@dataclass(frozen=True, slots=True)
class ImageFrameEntry:
    path: Path
    source_frame_id: int


@dataclass(frozen=True, slots=True)
class ImageSequenceManifest:
    directory: Path
    sequence_name: str
    entries: tuple[ImageFrameEntry, ...]
    first_source_frame_id: int
    width: int
    height: int


def decode_image_bgr(path: Path) -> ImageArray:
    """Decode one JPEG/PNG with PyAV and return contiguous BGR uint8 pixels."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MediaPipelineError(
            "dependency_missing", "PyAV is required to decode image sequences"
        ) from exc

    try:
        with av.open(io.BytesIO(path.read_bytes()), mode="r", format="image2") as container:
            frames = container.decode(video=0)
            frame = next(frames, None)
            if frame is None:
                raise MediaPipelineError(
                    "image_decode_failed", f"no image frame decoded: {path}"
                )
            if next(frames, None) is not None:
                raise MediaPipelineError(
                    "image_decode_failed", f"multiple frames decoded from image: {path}"
                )
            return np.ascontiguousarray(
                frame.to_ndarray(format="bgr24"), dtype=np.uint8
            )
    except MediaPipelineError:
        raise
    except Exception as exc:
        raise MediaPipelineError(
            "image_decode_failed", f"cannot decode image {path}: {exc}"
        ) from exc


class ImageSequenceDecoder:
    """Expose a single sequence directory through the producer decoder protocol."""

    def __init__(
        self,
        config: ImageSequenceConfig | None = None,
        image_loader: ImageLoader = decode_image_bgr,
    ) -> None:
        self.config = config or ImageSequenceConfig()
        self.image_loader = image_loader
        self._frame_regex = re.compile(self.config.frame_pattern, re.IGNORECASE)
        self._manifest_cache: dict[Path, ImageSequenceManifest] = {}

    def probe(self, sequence_path: str | Path) -> VideoInfo:
        manifest = self._build_manifest(Path(sequence_path))
        self._manifest_cache[manifest.directory] = manifest
        return VideoInfo(
            path=manifest.directory,
            width=manifest.width,
            height=manifest.height,
            source_fps=self.config.source_fps,
            duration_seconds=len(manifest.entries) / self.config.source_fps,
        )

    def iter_frames(self, video_info: VideoInfo) -> Iterator[DecodedFrame]:
        directory = video_info.path.resolve()
        manifest = self._manifest_cache.get(directory)
        if manifest is None:
            manifest = self._build_manifest(directory)
            self._manifest_cache[directory] = manifest

        for entry in manifest.entries:
            image = self._load_and_validate_image(entry.path)
            height, width = image.shape[:2]
            if (width, height) != (manifest.width, manifest.height):
                raise MediaPipelineError(
                    "image_size_changed",
                    f"image size changed from {manifest.width}x{manifest.height} "
                    f"to {width}x{height}: {entry.path}",
                )

            source_timestamp = (
                entry.source_frame_id - manifest.first_source_frame_id
            ) / self.config.source_fps
            yield DecodedFrame(
                image=image,
                source_frame_id=entry.source_frame_id,
                source_timestamp=source_timestamp,
                pts=None,
                timestamp_source="filename",
            )

    def _build_manifest(self, sequence_path: Path) -> ImageSequenceManifest:
        directory = sequence_path.resolve()
        if not directory.is_dir():
            raise MediaPipelineError(
                "image_sequence_not_found",
                f"image sequence directory does not exist: {directory}",
            )

        allowed = {suffix.lower() for suffix in self.config.extensions}
        candidates = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        ]
        if not candidates:
            raise MediaPipelineError(
                "image_sequence_empty", f"no supported images found: {directory}"
            )

        entries: list[ImageFrameEntry] = []
        frame_ids: dict[int, Path] = {}
        sequence_prefix: str | None = None
        for path in candidates:
            match = self._frame_regex.search(path.name)
            if match is None:
                raise MediaPipelineError(
                    "invalid_frame_name",
                    f"cannot parse frame number from image name: {path.name}",
                )

            current_prefix = path.name[: match.start()]
            if sequence_prefix is None:
                sequence_prefix = current_prefix
            elif current_prefix != sequence_prefix:
                raise MediaPipelineError(
                    "mixed_sequence",
                    f"multiple sequence prefixes found in {directory}: "
                    f"{sequence_prefix!r} and {current_prefix!r}",
                )

            source_frame_id = int(match.group("frame_id"))
            duplicate = frame_ids.get(source_frame_id)
            if duplicate is not None:
                raise MediaPipelineError(
                    "duplicate_frame",
                    f"frame {source_frame_id} appears in both "
                    f"{duplicate.name} and {path.name}",
                )
            frame_ids[source_frame_id] = path
            entries.append(ImageFrameEntry(path, source_frame_id))

        entries.sort(key=lambda item: item.source_frame_id)
        if self.config.require_contiguous:
            self._validate_continuity(entries)

        first_image = self._load_and_validate_image(entries[0].path)
        height, width = first_image.shape[:2]
        return ImageSequenceManifest(
            directory=directory,
            sequence_name=sequence_prefix or directory.name,
            entries=tuple(entries),
            first_source_frame_id=entries[0].source_frame_id,
            width=width,
            height=height,
        )

    @staticmethod
    def _validate_continuity(entries: list[ImageFrameEntry]) -> None:
        if entries[0].source_frame_id != 1:
            raise MediaPipelineError(
                "frame_start_invalid",
                f"expected first image frame 000001, found "
                f"{entries[0].source_frame_id:06d}",
            )
        for previous, current in zip(entries, entries[1:]):
            expected = previous.source_frame_id + 1
            if current.source_frame_id != expected:
                raise MediaPipelineError(
                    "frame_missing",
                    f"expected frame {expected:06d} after "
                    f"{previous.source_frame_id:06d}, found "
                    f"{current.source_frame_id:06d}",
                )

    def _load_and_validate_image(self, path: Path) -> ImageArray:
        try:
            image = self.image_loader(path)
        except MediaPipelineError:
            raise
        except Exception as exc:
            raise MediaPipelineError(
                "image_decode_failed", f"cannot decode image {path}: {exc}"
            ) from exc

        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise MediaPipelineError(
                "invalid_image_format",
                f"expected uint8[H,W,3], got dtype={image.dtype}, "
                f"shape={image.shape}: {path}",
            )
        return np.ascontiguousarray(image)
