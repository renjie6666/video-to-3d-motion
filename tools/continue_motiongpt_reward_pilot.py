"""Wait for the authorized four-score job, freeze five conditions, then run the pilot."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
summary = ROOT / "results/motiongpt_stage1/baseline_20261019_230154/rep_00/critic_summary.json"
started = time.monotonic()
print("Waiting for four additional scores before starting stage two", flush=True)
while not summary.is_file():
    if time.monotonic() - started > 7200:
        raise TimeoutError("Four-score stage did not finish within two hours; training was not started")
    time.sleep(2)
report = json.loads(summary.read_text(encoding="utf-8"))
if report["requested"] != 4 or report["scored"] != 4:
    raise RuntimeError("All four additional scores must succeed; training was not started")
for script in ("build_motioncritic_panel.py", "render_motiongpt_report.py",
               "run_motiongpt_reward_pilot.py", "render_motiongpt_report.py"):
    print(f"RUN {script}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", "-u", str(ROOT / "tools" / script)], cwd=ROOT, check=True)
print("Five-score panel and reward-training pilot finished", flush=True)
