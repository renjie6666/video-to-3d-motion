"""Load pinned official architectures without training/rendering dependencies."""
from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

PINS = {
    "motiongpt_source": "001aaca8d0ee218fc17f8265d11ac124044fe42f",
    "motioncritic_source": "71e9f52e67df4d9924607afd6ac4e0d4dc0dd27e",
}


def load_config(path: Path) -> dict:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str((path.parent / value).resolve())
    os.environ.setdefault("HF_HOME", str(Path(cfg["paths"]["assets"]) / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cfg["paths"]["assets"]) / "hf_cache"))
    return cfg


def load_file(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_modules(cfg: dict):
    source = Path(cfg["paths"]["motiongpt_source"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    # The upstream mGPT.data initializer imports the entire Lightning training
    # stack. Its humanml files use relative imports and can be loaded as a
    # separate namespace package without modifying upstream source.
    if "motiongpt_humanml" not in sys.modules:
        package = types.ModuleType("motiongpt_humanml")
        package.__path__ = [str(source / "mGPT/data/humanml")]
        sys.modules[package.__name__] = package
    return importlib.import_module("motiongpt_humanml.scripts.motion_process")


def provenance(cfg: dict) -> dict:
    versions = {}
    for key, expected in PINS.items():
        path = cfg["paths"][key]
        actual = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"{key}: expected {expected}, got {actual}")
        changes = subprocess.check_output(["git", "-C", path, "status", "--porcelain", "--untracked-files=all"], text=True)
        material = [line for line in changes.splitlines()
                    if not (line.startswith("?? ") and "__pycache__" in line[3:].replace("\\", "/").split("/")
                            and line.endswith(".pyc"))]
        if material:
            raise RuntimeError(f"Official source has local changes: {path}")
        versions[key] = actual
    def digest(path: Path) -> str:
        result = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                result.update(block)
        return result.hexdigest()
    assets = Path(cfg["paths"]["assets"])
    asset_hashes = {name: digest(assets / name) for name in
                    ("motiongpt_s3_h3d.tar", "motioncritic_pre.pth", "flan-t5-base/config.json",
                     "flan-t5-base/tokenizer.json") if (assets / name).is_file()}
    if cfg.get("adapter_checkpoint"):
        asset_hashes["adapter_checkpoint"] = digest(Path(cfg["adapter_checkpoint"]))
    implementation = {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    dataset_manifest = Path(cfg["paths"]["dataset"]) / "source_manifest.json"
    return {"sources": versions, "model_asset_sha256": asset_hashes, "implementation_sha256": implementation,
            "dataset_manifest_sha256": digest(dataset_manifest) if dataset_manifest.is_file() else None,
            "python": sys.version, "python_executable": sys.executable,
            "environment_prefix": sys.prefix, "is_conda_environment": (Path(sys.prefix) / "conda-meta").is_dir(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def find_asset(root: Path, suffix: str) -> Path:
    matches = [p for p in root.rglob(Path(suffix).name) if p.as_posix().endswith(suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {suffix} under {root}; found {len(matches)}")
    return matches[0]


def norms(cfg: dict):
    root = Path(cfg["paths"]["assets"])
    def pair(suffix):
        mean = np.load(find_asset(root, suffix + "/mean.npy"))
        std = np.load(find_asset(root, suffix + "/std.npy"))
        if mean.shape != (263,) or std.shape != (263,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"Invalid normalization statistics: {suffix}")
        return mean.astype(np.float32), std.astype(np.float32)
    return pair("t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"), pair("t2m/Comp_v6_KLD01/meta")


class Generator:
    def __init__(self, cfg: dict):
        official_modules(cfg)
        from transformers import AutoTokenizer, GenerationConfig, T5Config, T5ForConditionalGeneration
        from mGPT.archs.mgpt_lm import MLM
        from mGPT.archs.mgpt_vq import VQVae

        self.device = torch.device(cfg["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this configuration")
        assets = Path(cfg["paths"]["assets"])
        # The complete fine-tuned checkpoint already contains all T5 weights.
        # Construct the official architecture from its config, avoiding a
        # redundant download of the original pretrained T5 parameters.
        lm = MLM.__new__(MLM)
        torch.nn.Module.__init__(lm)
        for key, value in {"m_codebook_size": 512, "max_length": 256, "framerate": 20.,
                           "down_t": 4, "predict_ratio": .2, "inbetween_ratio": .25,
                           "quota_ratio": .5, "stage": "lm_instruct", "lm_type": "encdec"}.items():
            setattr(lm, key, value)
        lm.tokenizer = AutoTokenizer.from_pretrained(assets / "flan-t5-base", local_files_only=True, legacy=True)
        lm.tokenizer.add_tokens([f"<motion_id_{i}>" for i in range(515)])
        lm.language_model = T5ForConditionalGeneration(T5Config.from_pretrained(assets / "flan-t5-base", local_files_only=True))
        if (assets / "flan-t5-base/generation_config.json").exists():
            lm.language_model.generation_config = GenerationConfig.from_pretrained(assets / "flan-t5-base", local_files_only=True)
        lm.language_model.resize_token_embeddings(len(lm.tokenizer))
        vae = VQVae(nfeats=263, down_t=2, norm="None")
        state = torch.load(assets / "motiongpt_s3_h3d.tar", map_location="cpu")["state_dict"]
        for prefix, model in [("lm.", lm), ("vae.", vae)]:
            subset = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            model.load_state_dict(subset, strict=True)
            model.eval().requires_grad_(False).to(self.device)
        self.lm, self.vae = lm, vae
        if cfg.get("adapter_checkpoint"):
            from .adapters import inject_lora, restore_adapter_state
            adapter = torch.load(cfg["adapter_checkpoint"], map_location="cpu")
            actual_base = provenance(cfg)["model_asset_sha256"]["motiongpt_s3_h3d.tar"]
            if adapter["base_model_sha256"] != actual_base:
                raise ValueError("Adapter was trained against a different MotionGPT checkpoint")
            targets = inject_lora(lm.language_model, **adapter["lora"])
            if targets != adapter["target_modules"]:
                raise ValueError("Adapter target architecture differs from this generator")
            restore_adapter_state(lm.language_model, adapter["adapter_state"])
            lm.eval().requires_grad_(False)
        self.mean, self.std = norms(cfg)[0]
        self.motion_process = official_modules(cfg)

    @torch.inference_mode()
    def generate(self, captions: list[str], reference_lengths: list[int] | None = None):
        outputs = self.lm.generate_conditional(captions, stage="test")
        records = []
        for index, tokens in enumerate(outputs):
            tokens = tokens.clamp(0, 511)
            valid = len(tokens) > 1
            if valid:
                features = self.vae.decode(tokens)[0]
            else:
                # Same fallback as official val_t2m_forward. Record the failure
                # instead of silently reporting the fallback as valid output.
                length = reference_lengths[index] if reference_lengths else 40
                features = torch.zeros((length, 263), device=self.device)
            raw = features.cpu().numpy() * self.std + self.mean
            joints = self.motion_process.recover_from_ric(torch.from_numpy(raw), 22).numpy()
            records.append({"features": raw, "joints": joints, "tokens": tokens.cpu().numpy(),
                            "valid_generation": valid})
        return records


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
