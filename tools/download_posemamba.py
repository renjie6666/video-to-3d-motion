"""Download only the pinned official L checkpoint, verify before replacing."""

from pathlib import Path
import urllib.request

from video_to_3d_motion.pose3d.backend import file_sha256
from video_to_3d_motion.pose3d.config import WEIGHTS_REVISION, WEIGHTS_SHA256


def main():
    target = Path(__file__).resolve().parents[1] / "checkpoints/PoseMamba_L.bin"
    if target.is_file():
        if file_sha256(target) != WEIGHTS_SHA256:
            raise RuntimeError("existing checkpoint SHA mismatch; it was left unchanged")
        print(f"VERIFIED: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".download")
    if temporary.exists():
        raise RuntimeError(f"download temporary file already exists: {temporary}")
    url = f"https://huggingface.co/nankingjings/PoseMamba-weights/resolve/{WEIGHTS_REVISION}/PoseMamba_L.bin"
    urllib.request.urlretrieve(url, temporary)
    if file_sha256(temporary) != WEIGHTS_SHA256:
        raise RuntimeError("download SHA mismatch; final checkpoint was not written")
    temporary.rename(target)
    print(f"VERIFIED: {target}")


if __name__ == "__main__":
    main()
