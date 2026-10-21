"""Download first-party MotionGPT/MotionCritic model assets.

Run in the isolated environment described in docs/MotionGPT_第一阶段复现.md.
Large files are resumable. Downloads stay in ignored checkpoints/ and work/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "checkpoints/motiongpt_stage1"


def open_retry(request, timeout=90):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def download(url: str, path: Path, optional: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        print(f"EXISTS {path}", flush=True)
        return
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "MotionGPT-stage1-reproduction"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = open_retry(urllib.request.Request(url, headers=headers), timeout=90)
    except urllib.error.HTTPError as exc:
        if optional and exc.code == 404:
            return
        raise
    with response:
        resumed = response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("Unexpected Content-Range; refusing to corrupt download")
        size = int(response.headers.get("Content-Length", 0))
        if size > 64 * 1024 * 1024:
            response.close()
            total = offset + size if resumed else size
            if not resumed:
                offset = 0
            ranges = []
            step = (total - offset + 7) // 8
            for index, start in enumerate(range(offset, total, step)):
                ranges.append((index, start, min(start + step, total) - 1))

            def get_piece(item):
                index, start, end = item
                piece = path.with_name(path.name + f".chunk-{index}-{start}-{end}")
                expected = end - start + 1
                if piece.exists() and piece.stat().st_size == expected:
                    return piece
                failures = 0
                while failures < 6:
                    have = piece.stat().st_size if piece.exists() else 0
                    if have == expected:
                        break
                    if have > expected:
                        raise RuntimeError("Partial range is larger than expected")
                    current = start + have
                    request_end = min(end, current + 8 * 1024 * 1024 - 1)
                    request = urllib.request.Request(
                        url + ("&" if "?" in url else "?") + f"download=true&chunk={current}",
                        headers={"Range": f"bytes={current}-{request_end}", "User-Agent": headers["User-Agent"]},
                    )
                    try:
                        with urllib.request.urlopen(request, timeout=90) as stream:
                            if stream.status != 206 or not stream.headers.get("Content-Range", "").startswith(f"bytes {current}-{request_end}/"):
                                raise RuntimeError("Server did not honor byte range")
                            with piece.open("ab") as target:
                                while block := stream.read(1024 * 1024):
                                    target.write(block)
                    except (OSError, urllib.error.URLError) as exc:
                        print(f"RETRY {path.name} chunk {index + 1}: {type(exc).__name__}", flush=True)
                    after = piece.stat().st_size if piece.exists() else 0
                    if after == expected:
                        break
                    failures = failures + 1 if after == have else 0
                if not piece.exists() or piece.stat().st_size != expected:
                    raise RuntimeError("Incomplete range download after retries")
                print(f"CHUNK {path.name}: {index + 1}/{len(ranges)}", flush=True)
                return piece

            with ThreadPoolExecutor(max_workers=8) as pool:
                pieces = list(pool.map(get_piece, ranges))
            with partial.open("ab" if offset else "wb") as target:
                for piece in pieces:
                    with piece.open("rb") as stream:
                        while block := stream.read(1024 * 1024):
                            target.write(block)
            if partial.stat().st_size != total:
                raise RuntimeError("Assembled file size mismatch")
            partial.replace(path)
            for piece in pieces:
                piece.unlink()
            print(f"READY {path}", flush=True)
            return
        received = 0
        last = time.monotonic()
        with partial.open("ab" if resumed else "wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                received += len(chunk)
                if time.monotonic() - last > 15:
                    print(f"DOWNLOAD {path.name}: {received / 2**20:.1f}/{size / 2**20:.1f} MiB this request", flush=True)
                    last = time.monotonic()
        if size and received != size:
            raise RuntimeError(f"Incomplete download: {path}")
    partial.replace(path)
    print(f"READY {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="+", choices=["model", "critic", "t5", "evaluator", "smpl", "example"])
    args = parser.parse_args()
    for component in args.components:
        if component == "model":
            download("https://huggingface.co/OpenMotionLab/MotionGPT-base/resolve/main/motiongpt_s3_h3d.tar", ASSETS / "motiongpt_s3_h3d.tar")
        elif component == "critic":
            download("https://huggingface.co/littlekoyo/MotionCritic/resolve/main/motioncritic_pre.pth", ASSETS / "motioncritic_pre.pth")
        elif component == "t5":
            for filename in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model", "generation_config.json"]:
                download(f"https://huggingface.co/google/flan-t5-base/resolve/main/{filename}", ASSETS / "flan-t5-base" / filename, optional=True)
        else:
            file_id, filename = {
                "evaluator": ("1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8", "t2m.tar.gz"),
                "smpl": ("1qrFkPZyRwRGd0Q3EY76K8oJaIgs_WK9i", "smpl.tar.gz"),
                "example": ("1NED-lJ8vT9KOXwghUcMvJKUsY-fCU1hd", "visexample.pth"),
            }[component]
            target = ASSETS / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if component in ("evaluator", "smpl"):
                # The official script's public Drive file supports HTTP ranges.
                download(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t", target)
            elif not target.exists():
                import gdown
                result = gdown.download(id=file_id, output=str(target), resume=True)
                if result is None:
                    raise RuntimeError(f"Download failed: {filename}")
            print(f"READY {target}", flush=True)
    manifest = ASSETS / "download_sources.json"
    manifest.write_text(json.dumps({
        "motiongpt": "https://github.com/OpenMotionLab/MotionGPT",
        "motioncritic": "https://github.com/ou524u/MotionCritic",
        "humanml3d": "https://github.com/EricGuo5513/HumanML3D",
        "dataset_note": "Official metadata and an independently validated public mirror are handled by tools/prepare_humanml3d.py; original data terms apply.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
