"""Persist detailed and compact experiment CSVs without overwriting other runs."""
from __future__ import annotations

import csv
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


SIMPLE_FIELDS = (
    "experiment_id", "active_model", "mean_error_px", "common10_mean_error_px",
    "confident_mean_error_px", "confidence_coverage",
)
# Shoulders, elbows, wrists, hips and knees; exclude ankle/foot proxies.
COMMON10_H36M_INDICES = frozenset((1, 2, 4, 5, 11, 12, 13, 14, 15, 16))


def flatten_summary(summary: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in summary.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten_summary(value, name + "."))
        elif isinstance(value, list):
            result[name] = "; ".join(map(str, value))
        elif isinstance(value, bool):
            result[name] = str(value).lower()
        else:
            result[name] = "" if value is None else value
    return result


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv_atomic(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def _summary_lock(root: Path):
    """OS releases the lock if the process exits; the lock file may remain."""
    path = root / ".experiment_summary.lock"
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + 10
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"experiment summary is busy: {root}")
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def compact_row(row: dict) -> dict:
    result = dict(experiment_id=row["experiment_id"], active_model=row["active_model"],
                  mean_error_px=row.get("comparison.all_visible_pairs.mean_px", ""),
                  common10_mean_error_px=row.get("comparison.common10_pairs.mean_px", ""),
                  confident_mean_error_px=row.get("comparison.confident_pairs.mean_px", ""),
                  confidence_coverage=row.get("comparison.confidence_coverage", ""))
    # A partial comparison from a failed/cancelled run is not a final score.
    if row.get("status") != "completed" or row.get("comparison.status") != "completed":
        return {key: value if key in ("experiment_id", "active_model") else ""
                for key, value in result.items()}
    if result["common10_mean_error_px"] in (None, ""):
        value = row.get("comparison.joint_comparison")
        if value and Path(value).is_file():
            _, joints = read_csv(Path(value))
            errors = [float(joint["error_px"]) for joint in joints
                      if joint.get("mapping") == "direct"
                      and joint.get("gt_joint_index") in {str(i) for i in COMMON10_H36M_INDICES}
                      and joint.get("gt_visible", "").lower() == "true"
                      and joint.get("error_px") not in (None, "")]
            if any(not math.isfinite(error) or error < 0 for error in errors):
                raise ValueError("invalid common-joint pixel error")
            result["common10_mean_error_px"] = sum(errors) / len(errors) if errors else ""
    return result


def save_experiment(root: Path, run: Path, row: dict) -> None:
    """Save the local row first, then update both cumulative CSVs under one lock."""
    write_csv_atomic(run / (row["experiment_id"] + "_summary.csv"), list(row), [row])
    with _summary_lock(root):
        fields, old = read_csv(root / "experiment_summary.csv")
        fields.extend(key for key in row if key not in fields)
        rows = [item for item in old if item["experiment_id"] != row["experiment_id"]]
        rows.append(row)
        write_csv_atomic(root / "experiment_summary.csv", fields, rows)
        write_csv_atomic(root / "experiment_summary_simple.csv", SIMPLE_FIELDS,
                         [compact_row(item) for item in rows])


def rebuild_reports(root: Path) -> None:
    """Recover from local rows and regenerate the compact view without GPU inference."""
    with _summary_lock(root):
        original_fields, original_rows = read_csv(root / "experiment_summary.csv")
        fields, rows = list(original_fields), list(original_rows)
        indexed = {row["experiment_id"]: row for row in rows}
        for path in sorted(root.glob("*/*_summary.csv")):
            local_fields, local_rows = read_csv(path)
            if len(local_rows) != 1 or "experiment_id" not in local_fields:
                continue
            indexed[local_rows[0]["experiment_id"]] = local_rows[0]
            fields.extend(key for key in local_fields if key not in fields)
        rows = list(indexed.values())
        # Calculate before replacing either report if a source table is corrupt.
        compact = [compact_row(row) for row in rows]
        if fields != original_fields or rows != original_rows:
            write_csv_atomic(root / "experiment_summary.csv", fields, rows)
        write_csv_atomic(root / "experiment_summary_simple.csv", SIMPLE_FIELDS, compact)


def rebuild_simple_report(root: Path) -> None:
    """Refresh just the compact view; the detailed CSV is read-only here."""
    with _summary_lock(root):
        _, rows = read_csv(root / "experiment_summary.csv")
        if not (root / "experiment_summary.csv").is_file():
            raise FileNotFoundError("No detailed report; use --rebuild-reports to recover local rows")
        write_csv_atomic(root / "experiment_summary_simple.csv", SIMPLE_FIELDS,
                         [compact_row(row) for row in rows])
