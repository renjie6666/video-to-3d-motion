"""Check local CUDA and S1-02 dependencies without installing the project.

Run with the Python interpreter intended for inference:
    python tools/check_local_gpu.py
    python tools/check_local_gpu.py --require-pose2d
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


MARKER = "GPU_CHECK_RESULT="


def probe_torch() -> dict:
    import torch

    info = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_device_count": torch.cuda.device_count(),
    }
    if not info["cuda_available"]:
        return {"status": "failed", "details": info, "error": "PyTorch cannot access CUDA"}
    info["devices"] = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2),
        }
        for index in range(torch.cuda.device_count())
    ]
    # S1-02 uses the first visible GPU. Compare actual GPU math against CPU.
    cpu = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 256
    gpu = cpu.to("cuda:0")
    actual = (gpu @ gpu.T).cpu()
    torch.cuda.synchronize(0)
    torch.testing.assert_close(actual, cpu @ cpu.T, rtol=1e-3, atol=1e-4)
    info["cuda_matrix_test"] = "passed"
    return {"status": "passed", "details": info}


def probe_pose2d() -> dict:
    versions = {}
    errors = {}
    for name in ("numpy", "yaml", "av", "torch", "mmengine", "mmcv", "mmdet", "mmpose"):
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    # Import real entry points to expose version assertions and missing extensions.
    for name, symbols in (
        ("mmdet.apis", ("init_detector", "inference_detector")),
        ("mmpose.apis", ("init_model", "inference_topdown")),
    ):
        try:
            module = importlib.import_module(name)
            for symbol in symbols:
                getattr(module, symbol)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    try:
        import torch
        from mmcv.ops import nms

        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device="cuda:0")
        scores = torch.tensor([0.9, 0.8], device="cuda:0")
        _, keep = nms(boxes, scores, 0.5)
        torch.cuda.synchronize(0)
        if keep.cpu().tolist() != [0]:
            raise RuntimeError("Unexpected MMCV CUDA NMS result")
        versions["mmcv_cuda_nms_test"] = "passed"
    except Exception as exc:
        errors["mmcv_cuda_nms_test"] = f"{type(exc).__name__}: {exc}"
    return {"status": "failed" if errors else "passed", "details": versions, "errors": errors}


def run_probe(name: str, timeout: float) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--internal-probe", name],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(MARKER):
                report = json.loads(line[len(MARKER):])
                if result.stderr.strip():
                    report["stderr"] = result.stderr[-8000:]
                return report
        return {
            "status": "failed", "error": f"Probe exited with code {result.returncode}",
            "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def probe_driver(timeout: float) -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "output": result.stdout.strip(), "error": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local GPU math and S1-02 CUDA dependencies")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "results/local_gpu_check.json",
    )
    parser.add_argument("--timeout", type=float, default=120, help="Timeout in seconds per dependency probe")
    parser.add_argument("--require-pose2d", action="store_true", help="Fail unless OpenMMLab checks also pass")
    parser.add_argument("--internal-probe", choices=("torch", "pose2d"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.internal_probe:
        try:
            report = probe_torch() if args.internal_probe == "torch" else probe_pose2d()
        except Exception as exc:
            report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        print(MARKER + json.dumps(report, ensure_ascii=True))
        return 0 if report["status"] == "passed" else 1

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "checks": {},
    }
    for name in ("driver", "torch", "pose2d"):
        print(f"Checking {name} ...", flush=True)
        result = probe_driver(min(args.timeout, 15)) if name == "driver" else run_probe(name, args.timeout)
        report["checks"][name] = result
        print(f"  {result['status'].upper()}", flush=True)
        if result.get("error"):
            print(f"  {result['error']}")
        for key, value in result.get("errors", {}).items():
            print(f"  {key}: {value}")
    report["gpu_usable"] = report["checks"]["torch"]["status"] == "passed"
    report["pose2d_dependencies_ready"] = report["gpu_usable"] and report["checks"]["pose2d"]["status"] == "passed"
    report["scope"] = "Environment checks only; model configs, checkpoints and real inference are not validated."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GPU usable: {report['gpu_usable']}")
    print(f"S1-02 dependencies ready: {report['pose2d_dependencies_ready']}")
    print(f"Report: {args.output.resolve()}")
    ready = report["pose2d_dependencies_ready"] if args.require_pose2d else report["gpu_usable"]
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
