"""Select an image sequence from the large list-of-dictionaries H36M pickle.

Only NumPy reconstruction globals are accepted. Records are consumed in pickle
list batches instead of retaining the entire dataset. Cross-record references to
discarded containers are rejected; this reader targets independent frame records.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np


def image_key(value: str) -> str:
    parts = value.replace("\\", "/").split("/")
    return "/".join(parts[-2:])


class _FoundAll(Exception):
    pass


class _FrameMemo(dict):
    """Keep pickle memo numbering stable when old frame objects are released."""

    def __init__(self):
        super().__init__()
        self.next_index = 0
        self.discardable = set()
        self.shared_metadata_ids = set()

    def __len__(self):
        return self.next_index

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.next_index = max(self.next_index, key + 1)
        small_tuple = isinstance(value, tuple) and len(value) <= 8 and all(
            isinstance(item, (str, int, float, bool, type, np.dtype))
            or item is None for item in value
        )
        if isinstance(value, (dict, list, np.ndarray)) or (
            isinstance(value, tuple) and not small_tuple
        ) or (isinstance(value, bytes) and len(value) > 64):
            self.discardable.add(key)

    def release_frames(self, root):
        for key in self.discardable:
            if self.get(key) is not root and id(self.get(key)) not in self.shared_metadata_ids:
                self.pop(key, None)
        self.discardable.clear()

    def keep_metadata(self, value):
        if id(value) in self.shared_metadata_ids:
            return
        self.shared_metadata_ids.add(id(value))
        if isinstance(value, dict):
            for item in value.values():
                self.keep_metadata(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.keep_metadata(item)


class _SequenceUnpickler(pickle._Unpickler):
    # Python's implementation exposes opcode dispatch, needed to consume the
    # outer list incrementally. Covered by protocol 4 and real-file tests.
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, stream, consume):
        super().__init__(stream)
        self.memo = _FrameMemo()
        self.root = None
        self.consume = consume
        self.records_read = 0

    def find_class(self, module, name):
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            if name in {"_reconstruct", "scalar"}:
                return getattr(np.core.multiarray, name)
        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        raise pickle.UnpicklingError(f"unsupported annotation global: {module}.{name}")

    def load_empty_list(self):
        is_root = self.root is None and not self.stack and not self.metastack
        super().load_empty_list()
        if is_root:
            self.root = self.stack[-1]

    dispatch[pickle.EMPTY_LIST[0]] = load_empty_list

    def _consume_root(self):
        if self.stack and self.stack[-1] is self.root:
            for record in self.root:
                if not isinstance(record, dict) or "image" not in record:
                    raise ValueError("expected H36M list of per-image annotation dictionaries")
                self.records_read += 1
                # Camera calibration is intentionally shared across frames in
                # the user's pickle; preserve it and its child arrays.
                if "camera" in record:
                    self.memo.keep_metadata(record["camera"])
                self.consume(record)
            self.root.clear()
            self.memo.release_frames(self.root)
            if self.records_read % 10000 == 0:
                print(f"H36M: scanned {self.records_read} annotations", file=sys.stderr)

    def load_appends(self):
        super().load_appends()
        self._consume_root()

    dispatch[pickle.APPENDS[0]] = load_appends

    def load_append(self):
        super().load_append()
        self._consume_root()

    dispatch[pickle.APPEND[0]] = load_append


def _present_names(path: Path, names: set[str]) -> set[str]:
    """Cheap byte scan avoids deserializing the remaining GB after all matches.

    A name found here is only a candidate; its actual annotation still has to
    pass the restricted reader and exact sequence/image matching below.
    """
    if not names:
        return set()
    prefix = next(iter(names)).rsplit("_", 1)[0]
    pattern = re.compile(re.escape(prefix.encode("ascii")) + rb"_\d{6}\.(?:jpg|jpeg|png)")
    overlap = max(map(len, names)) + 32
    found = set()
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            data = tail + chunk
            found.update(match.group().decode("ascii") for match in pattern.finditer(data))
            tail = data[-overlap:]
    return found & names


def load_sequence_annotations(
    path: Path, sequence_dir: Path, *, use_cache: bool = True
) -> dict[str, dict]:
    """Return original-pixel GT for matching files, with a local subset cache."""
    path, sequence_dir = path.resolve(), sequence_dir.resolve()
    names = {p.name for p in sequence_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    stat = path.stat()
    identity = json.dumps([str(path), stat.st_size, stat.st_mtime_ns, sequence_dir.name, sorted(names)])
    digest = hashlib.sha256(identity.encode()).hexdigest()
    cache = path.parent / ".annotation_cache" / f"{digest}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    print(f"H36M: locating {sequence_dir.name} in {path.name}", file=sys.stderr)
    candidates = _present_names(path, names)
    result = {}

    def consume(record):
        key = image_key(str(record["image"]))
        if key.split("/")[0] != sequence_dir.name or key.split("/")[-1] not in names:
            return
        if key in result:
            raise ValueError(f"duplicate annotation for {key}")
        joints = np.asarray(record["joints_2d"], dtype=np.float64)
        visibility = np.asarray(record["joints_vis"])
        if joints.shape != (17, 2) or not np.isfinite(joints).all():
            raise ValueError(f"invalid joints_2d for {key}")
        if visibility.ndim == 2 and visibility.shape[0] == 17 and visibility.shape[1] >= 2:
            visible = (visibility[:, :2] > 0).all(axis=1)
        elif visibility.shape == (17,):
            visible = visibility > 0
        else:
            raise ValueError(f"invalid joints_vis for {key}")
        result[key] = {"keypoints": joints.tolist(), "visible": visible.tolist()}
        if {key.rsplit("/", 1)[-1] for key in result} >= candidates:
            raise _FoundAll

    if candidates:
        try:
            with path.open("rb") as stream:
                reader = _SequenceUnpickler(stream, consume)
                root = reader.load()
                if root is not reader.root or root is None:
                    raise ValueError("annotation pickle must contain a top-level list")
        except _FoundAll:
            pass
        except (KeyError, pickle.UnpicklingError) as exc:
            if isinstance(exc, pickle.UnpicklingError) and "Memo value not found" not in str(exc):
                raise
            raise ValueError(
                "annotation pickle reuses a released frame object; "
                "expected independent per-frame dictionaries"
            ) from exc
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(cache)
    print(f"H36M: matched {len(result)}/{len(names)} images", file=sys.stderr)
    return result
