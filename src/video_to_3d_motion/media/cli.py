"""Command-line consumers used to verify video and image-sequence inputs."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, TextIO

from .config import AppConfig, load_config
from .image_sequence_decoder import ImageSequenceDecoder
from .models import DecodeFailure, EndOfVideo, FrameTask, QueueItem
from .producer import Decoder, FrameTaskProducer


class ParsedArgs(Protocol):
    source: Path
    config: Path
    video_id: str | None
    tasks_output: Path
    summary_output: Path


def _common_parser(description: str, source_help: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("source", type=Path, help=source_help)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-id", help="override the sequence identifier")
    parser.add_argument(
        "--tasks-output",
        type=Path,
        required=True,
        help="write one FrameTask metadata record per line as UTF-8 JSONL",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
        help="write the UTF-8 JSON run summary to this path",
    )
    return parser


def build_video_parser() -> argparse.ArgumentParser:
    return _common_parser(
        "Decode a video into FrameTask records", "path to an MP4/AVI video"
    )


def build_images_parser() -> argparse.ArgumentParser:
    return _common_parser(
        "Decode a continuous image directory into FrameTask records",
        "path to one ordered JPEG/PNG sequence directory",
    )


def _task_metadata(task: FrameTask) -> dict[str, object]:
    """Return the inspectable queue fields without serializing image pixels."""
    return {
        "video_id": task.video_id,
        "frame_id": task.frame_id,
        "source_frame_id": task.source_frame_id,
        "timestamp": task.timestamp,
        "source_timestamp": task.source_timestamp,
        "timestamp_source": task.timestamp_source,
        "image_size": list(task.image_size),
        "image_shape": list(task.image.shape),
        "image_dtype": str(task.image.dtype),
        "fps": task.fps,
    }


def _open_tasks_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="\n")


class ConsoleProgress:
    """Small dependency-free terminal progress bar."""

    def __init__(self, total: int | None) -> None:
        self.total = total if total and total > 0 else None
        self._last_marker = -1

    def update(self, count: int) -> None:
        if self.total is None:
            marker = count // 50
            if marker == self._last_marker:
                return
            self._last_marker = marker
            sys.stderr.write(f"\rDecoding: {count} frames")
            sys.stderr.flush()
            return

        percent = min(100, int(count * 100 / self.total))
        if percent == self._last_marker and count < self.total:
            return
        self._last_marker = percent
        filled = min(30, int(percent * 30 / 100))
        bar = "#" * filled + "-" * (30 - filled)
        sys.stderr.write(
            f"\rDecoding: [{bar}] {percent:3d}% ({count}/{self.total})"
        )
        sys.stderr.flush()

    def finish(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _estimate_total_frames(
    args: ParsedArgs, config: AppConfig, producer: FrameTaskProducer
) -> int | None:
    try:
        info = producer.decoder.probe(args.source)
    except Exception:
        return None
    if info.duration_seconds is None:
        return None
    return max(1, round(info.duration_seconds * config.media.target_fps))


def _run(args: ParsedArgs, config: AppConfig, decoder: Decoder | None) -> int:
    capacity = int(config.queue.capacity or 0)
    output_queue: queue.Queue[QueueItem] = queue.Queue(maxsize=capacity)
    stop_event = threading.Event()
    producer = FrameTaskProducer(config, decoder=decoder)
    result_holder = []
    progress = ConsoleProgress(_estimate_total_frames(args, config, producer))
    tasks_stream = _open_tasks_output(args.tasks_output)

    sequence_id = args.video_id
    if sequence_id is None and decoder is not None:
        sequence_id = args.source.name

    worker = threading.Thread(
        target=lambda: result_holder.append(
            producer.run(args.source, output_queue, stop_event, sequence_id)
        ),
        name="frame-task-producer",
    )
    worker.start()

    first_task: FrameTask | None = None
    last_task: FrameTask | None = None
    terminal: EndOfVideo | DecodeFailure | None = None
    task_count = 0
    try:
        while terminal is None:
            item = output_queue.get()
            try:
                if isinstance(item, FrameTask):
                    first_task = first_task or item
                    last_task = item
                    task_count += 1
                    tasks_stream.write(
                        json.dumps(_task_metadata(item), ensure_ascii=False) + "\n"
                    )
                    progress.update(task_count)
                else:
                    terminal = item
            finally:
                output_queue.task_done()
    except KeyboardInterrupt:
        stop_event.set()
    except BaseException:
        stop_event.set()
        raise
    finally:
        tasks_stream.close()
        worker.join()
        progress.finish()

    if not result_holder:
        return 130
    result = result_holder[0]
    summary = {
        "status": result.status,
        "video_id": result.video_id,
        "error_code": result.error_code,
        "message": result.message,
        "metrics": asdict(result.metrics),
        "first_frame_id": first_task.frame_id if first_task else None,
        "last_frame_id": last_task.frame_id if last_task else None,
        "task_records_written": task_count,
        "tasks_output": str(args.tasks_output),
        "terminal_message": type(terminal).__name__ if terminal else None,
    }
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(summary_json + "\n", encoding="utf-8")
    print(summary_json)

    if result.status == "completed":
        print(
            f"SUCCESS: decoded {result.metrics.emitted_frames} frames; "
            f"summary={args.summary_output}; tasks={args.tasks_output}",
            file=sys.stderr,
        )
        return 0

    print(
        f"FAILED: {result.error_code or 'unknown_error'}: "
        f"{result.message or 'no details'}",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    args = build_video_parser().parse_args()
    return _run(args, load_config(args.config), decoder=None)


def images_main() -> int:
    args = build_images_parser().parse_args()
    config = load_config(args.config)
    decoder = ImageSequenceDecoder(config.image_sequence)
    return _run(args, config, decoder=decoder)


if __name__ == "__main__":
    raise SystemExit(main())