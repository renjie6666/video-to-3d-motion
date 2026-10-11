"""Replay S1-02 predictions through the CPU-only S1-03 baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .archive import WindowArchive
from .config import load_pose_adapter_config
from .exceptions import PoseAdapterError
from .jsonl_reader import iter_pose2d_jsonl
from .worker import PoseAdapterWorker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate and normalize a terminal 243-frame H36M17 window")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True, help="S1-02 JSON or single-experiment detailed CSV")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_pose_adapter_config(args.config)
        archive = WindowArchive(args.output_dir, config)
        result = PoseAdapterWorker(config).run_stream(
            iter_pose2d_jsonl(args.input, args.input_summary), archive.emit)
        summary = archive.finish(result, input=str(args.input.resolve()),
                                 input_summary=str(args.input_summary.resolve()))
    except (PoseAdapterError, OSError, ValueError, TypeError) as exc:
        print(f"FAILED: {getattr(exc, 'code', 'invalid_config')}: {exc}", file=sys.stderr)
        return 1
    print(f"{result.status.upper()}: {summary}")
    if result.status != "completed":
        print(f"{result.error_code}: {result.message}", file=sys.stderr)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
