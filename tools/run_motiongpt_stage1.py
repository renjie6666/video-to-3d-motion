"""Run the phase-one CLI from the isolated environment without pip installing this repo."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_to_3d_motion.motion_generation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
