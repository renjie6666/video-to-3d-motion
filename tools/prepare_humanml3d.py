"""Download a public preprocessed mirror and verify it against official metadata.

The mirror is NOT an official release. Record its immutable revision and LFS
SHA256, then compare splits, statistics and public example motions to upstream.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from prepare_motiongpt_stage1 import ROOT, download

MIRROR = "yonful/gem-analysis-humanml3d"
FEATURE_MIRROR = "lavinal712/HumanML3D"
FEATURE_REVISION = "430eae544e2d71b8e582123c1ea01cdb24eef725"
FEATURE_SHA256 = "d857a60b3d266ebe1ab95334e3f2845104c2855ba470cfccf2c237936897a682"


def main() -> None:
    dataset = ROOT / "datasets/HumanML3D"
    dataset.mkdir(parents=True, exist_ok=True)
    official = ROOT / "work/humanml3d_metadata/HumanML3D"
    official.mkdir(parents=True, exist_ok=True)
    # Fixed using the official Git remote, not a moving API query on each run.
    official_revision = "9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23"
    for filename in ("train.txt", "val.txt", "test.txt", "train_val.txt", "all.txt", "Mean.npy", "Std.npy", "texts.zip"):
        download(f"https://raw.githubusercontent.com/EricGuo5513/HumanML3D/{official_revision}/HumanML3D/{filename}", official / filename)
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{MIRROR}", timeout=60))
    revision = info["sha"]
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{MIRROR}/tree/{revision}", timeout=60))
    entry = next(item for item in tree if item["path"] == "humanml3d.tar.gz")
    expected = entry["lfs"]["oid"]
    archive = dataset / "humanml3d.tar.gz"
    download(f"https://huggingface.co/datasets/{MIRROR}/resolve/{revision}/humanml3d.tar.gz", archive)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError("Downloaded dataset SHA256 differs from mirror LFS metadata")
    file_count = 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            parts = list(PurePosixPath(member.name).parts)
            if parts and parts[0].lower() == "humanml3d":
                parts = parts[1:]
            if not parts:
                continue
            if any(part in ('.cache', '__MACOSX', '.git') or part.startswith('._') for part in parts):
                continue
            if parts[0] == 'new_joints' and member.isfile() and parts[-1].endswith('.npy'):
                parts = ['new_joints', parts[-1]]
            path = (dataset / Path(*parts)).resolve()
            if not path.is_relative_to(dataset.resolve()) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe dataset archive member: {member.name}")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists() or path.stat().st_size != member.size:
                    source = stream.extractfile(member)
                    if source is None:
                        raise ValueError(member.name)
                    with source, path.open("wb") as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                file_count += 1
                if file_count % 5000 == 0:
                    print(f"EXTRACTED {file_count} dataset files", flush=True)
    # Use canonical author-supplied captions, even if a community package
    # augmented its text descriptions for another downstream task.
    with zipfile.ZipFile(official / 'texts.zip') as archive_texts:
        for name in archive_texts.namelist():
            parts = PurePosixPath(name).parts
            if not name.endswith('.txt') or any(part.startswith('._') for part in parts):
                continue
            identity = Path(name).name
            (dataset / 'texts').mkdir(exist_ok=True)
            canonical = archive_texts.read(name)
            destination = dataset / 'texts' / identity
            if not destination.exists() or destination.read_bytes() != canonical:
                destination.write_bytes(canonical)
    checks = {"captions": "canonical official texts.zip replaces any mirror-specific augmentation"}
    for filename in ("train.txt", "val.txt", "test.txt", "train_val.txt", "all.txt"):
        actual = (dataset / filename).read_text(encoding="utf-8").splitlines()
        reference = (official / filename).read_text(encoding="utf-8").splitlines()
        if actual != reference:
            raise ValueError(f"Mirror split differs from official: {filename}")
        checks[filename] = {"count": len(actual), "matches_official": True}
    for filename in ("Mean.npy", "Std.npy"):
        np.testing.assert_allclose(np.load(dataset / filename, allow_pickle=False),
                                   np.load(official / filename, allow_pickle=False), rtol=1e-6, atol=1e-8)
        checks[filename] = "matches_official"
    # The XYZ mirror omits model features. Obtain archived 263-D features
    # separately; XYZ reconstruction is not assumed equivalent to that data.
    joint_root = dataset / 'new_joints'
    for path in list(joint_root.rglob('*.npy')):
        if path.name.startswith('._') or path.parent == joint_root:
            continue
        destination = joint_root / path.name
        if not path.resolve().is_relative_to(joint_root.resolve()) or not destination.resolve().is_relative_to(joint_root.resolve()):
            raise ValueError('Joint file leaves intended dataset directory')
        if destination.exists():
            raise ValueError(f'Duplicate motion ID: {path.name}')
        path.replace(destination)
    feature_archive = dataset / 'new_joint_vecs.zip'
    download(f'https://huggingface.co/datasets/{FEATURE_MIRROR}/resolve/{FEATURE_REVISION}/new_joint_vecs.zip', feature_archive)
    with feature_archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256') if hasattr(hashlib, 'file_digest') else None
        if digest is None:
            digest = hashlib.sha256()
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
    if digest.hexdigest() != FEATURE_SHA256:
        raise ValueError('Feature archive SHA256 mismatch')
    feature_root = dataset / 'new_joint_vecs'
    feature_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(feature_archive) as feature_zip:
        identities = set()
        for member in feature_zip.infolist():
            parts = PurePosixPath(member.filename).parts
            if member.is_dir() or not parts or not parts[-1].endswith('.npy'):
                continue
            if any(part == '..' or part.startswith('._') or part == '__MACOSX' for part in parts):
                raise ValueError(f'Unexpected feature path: {member.filename}')
            identity = parts[-1]
            if identity in identities:
                raise ValueError(f'Duplicate feature ID: {identity}')
            identities.add(identity)
            # Always replace locally reconstructed attempts with archive bytes.
            # ZipFile.read also validates each member CRC.
            temporary = feature_root / (identity + '.part')
            temporary.write_bytes(feature_zip.read(member))
            temporary.replace(feature_root / identity)
            if len(identities) % 2000 == 0:
                print(f'FEATURES extracted {len(identities)}', flush=True)
    checks['feature_source'] = {'mirror': FEATURE_MIRROR, 'revision': FEATURE_REVISION,
                                'sha256': FEATURE_SHA256, 'locally_reconstructed': False}
    example = official / 'new_joint_vecs/012314.npy'
    download(f'https://raw.githubusercontent.com/EricGuo5513/HumanML3D/{official_revision}/HumanML3D/new_joint_vecs/012314.npy', example)
    example_actual = np.load(feature_root / example.name, allow_pickle=False)
    example_expected = np.load(example, allow_pickle=False)
    np.testing.assert_allclose(example_actual, example_expected, rtol=1e-4, atol=2e-5)
    checks['official_feature_example_012314'] = {
        'matches_with_tolerance': True, 'rtol': 1e-4, 'atol': 2e-5,
        'max_absolute_difference': float(np.max(np.abs(example_actual - example_expected))),
        'array_identical': bool(np.array_equal(example_actual, example_expected))}
    motion_files = sorted((dataset / "new_joint_vecs").glob("*.npy"))
    if not motion_files:
        raise ValueError("Archive has no HumanML3D motion features")
    for path in motion_files:
        motion = np.load(path, allow_pickle=False, mmap_mode="r")
        if motion.ndim != 2 or motion.shape[1] != 263 or not np.isfinite(motion).all():
            raise ValueError(f"Invalid motion: {path}")
    expected_ids = set((official / "all.txt").read_text(encoding="utf-8").split())
    available_ids = {path.stem for path in motion_files}
    missing = sorted(expected_ids - available_ids)
    if missing:
        raise ValueError(f"Mirror lacks {len(missing)} official motions, first: {missing[:10]}")
    checks["motions"] = {"count": len(motion_files), "dimensions": 263, "finite": True, "all_official_ids_present": True}
    for identity in expected_ids:
        if not (dataset / "texts" / f"{identity}.txt").exists():
            raise ValueError(f"Missing text: {identity}")
    report = {"status": "validated_public_mirror", "mirror": f"https://huggingface.co/datasets/{MIRROR}",
              "mirror_revision": revision, "archive_sha256": expected,
              "official_source": "https://github.com/EricGuo5513/HumanML3D",
              "official_source_commit": official_revision,
              "checks": checks, "files_extracted": file_count,
              "note": "Community preprocessed copy. Matching metadata does not prove byte-identical preprocessing of every motion. Original data terms still apply."}
    (dataset / "source_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
