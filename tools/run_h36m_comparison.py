"""Recompare saved predictions without loading models or rerunning GPU inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_to_3d_motion.pose2d.comparison import main

if __name__ == "__main__":
    raise SystemExit(main())
