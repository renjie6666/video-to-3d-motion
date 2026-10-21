"""Perceptual scoring with explicit SMPL fitting and time-preserving windows.

HumanML3D IK rotations are not SMPL pose parameters. Always fit XYZ joints
using the official MotionGPT SMPLify implementation before scoring.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from .runtime import find_asset, load_file


def window_starts(length: int, frames: int = 60, stride: int = 60) -> list[int]:
    if frames < 2 or stride < 1 or stride > frames:
        raise ValueError("Require frames >= 2 and 1 <= stride <= frames")
    if length < frames:
        return []
    starts = list(range(0, length - frames + 1, stride))
    if starts[-1] != length - frames:
        starts.append(length - frames)
    return starts


def resample_smpl(poses: np.ndarray, translations: np.ndarray, source_fps: float, target_fps: float):
    """SLERP rotations, linearly interpolate translations, preserve duration."""
    poses, translations = np.asarray(poses), np.asarray(translations)
    if poses.ndim != 3 or poses.shape[1:] != (24, 3) or translations.shape != (len(poses), 3):
        raise ValueError("Expected SMPL axis-angle [T,24,3] and translation [T,3]")
    if len(poses) < 2 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Need >= 2 frames and positive frame rates")
    if not np.isfinite(poses).all() or not np.isfinite(translations).all():
        raise ValueError("Non-finite SMPL input")
    original = np.arange(len(poses), dtype=np.float64) / source_fps
    target = np.arange(int(np.floor(original[-1] * target_fps + 1e-8)) + 1) / target_fps
    rotations = np.stack([Slerp(original, Rotation.from_rotvec(poses[:, j]))(target).as_rotvec()
                          for j in range(24)], axis=1)
    translations = np.stack([np.interp(target, original, translations[:, j]) for j in range(3)], axis=-1)
    return rotations.astype(np.float32), translations.astype(np.float32)


class Critic:
    def __init__(self, cfg: dict):
        source = Path(cfg["paths"]["motioncritic_source"]) / "MotionCritic"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from lib.model.critic import MotionCritic
        self.device = torch.device(cfg["device"])
        self.model = MotionCritic(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
        checkpoint = torch.load(Path(cfg["paths"]["assets"]) / "motioncritic_pre.pth", map_location="cpu")
        state = {key.removeprefix("module."): value for key, value in checkpoint["model_state_dict"].items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.protocol = cfg["critic"]
        if self.protocol.get("translation_origin") != "per_window_first_frame":
            raise ValueError("Critic protocol requires translation relative to each window's first frame")

    @torch.inference_mode()
    def score_input(self, motion: np.ndarray) -> list[float]:
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 4 or motion.shape[2:] != (25, 3) or not 1 <= motion.shape[1] <= 243 or not np.isfinite(motion).all():
            raise ValueError("MotionCritic expects finite [B,T<=243,25,3] SMPL input")
        scores = self.model.batch_critic(torch.from_numpy(motion).to(self.device)).flatten()
        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite MotionCritic scores")
        return scores.cpu().tolist()

    def score_smpl(self, pose: np.ndarray, translation: np.ndarray) -> dict:
        pose, translation = resample_smpl(pose, translation, self.protocol["source_fps"], self.protocol["target_fps"])
        starts = window_starts(len(pose), self.protocol["window_frames"], self.protocol["stride_frames"])
        frames = self.protocol["window_frames"]
        scores = []
        for start in starts:
            # The upstream HumanAct12 loader uses pelvis displacement relative
            # to the first selected frame, not absolute world/SMPL height.
            # Rebase each crop just as its loader rebases each sampled clip.
            root = translation[start:start + frames] - translation[start]
            x = np.concatenate([pose[start:start + frames], root[:, None]], axis=1)
            scores.extend(self.score_input(x[None]))
        return {"score": float(np.mean(scores)) if scores else None, "window_scores": scores,
                "window_starts": starts, "resampled_frames": len(pose),
                "status": "scored" if scores else f"too_short_for_{frames}_frames_at_{self.protocol['target_fps']}fps",
                "protocol": (f"smplify_{self.protocol['optimizer']}_xyz_{self.protocol['source_fps']}_to_"
                             f"{self.protocol['target_fps']}fps_{frames}frame_relative_root_windows_v2")}


class SmplFitter:
    def __init__(self, cfg: dict):
        import h5py
        # Compatibility is confined to loading the old official SMPL pickle.
        # No source files or installed third-party modules are patched on disk.
        for name, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                            "object": object, "str": str, "unicode": str}.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
        import smplx

        self.device = torch.device(cfg["device"])
        assets = Path(cfg["paths"]["assets"])
        model_path = find_asset(assets, "SMPL_NEUTRAL.pkl")
        mean_path = find_asset(assets, "neutral_smpl_mean_params.h5")
        prior_path = find_asset(assets, "gmm_08.pkl")
        source = Path(cfg["paths"]["motiongpt_source"])
        sys.path.insert(0, str(source))
        fit_source = source / "mGPT/data/transforms/joints2rots"
        sys.path.insert(0, str(fit_source))
        fit_config = load_file("config", fit_source / "config.py")
        fit_config.GMM_MODEL_DIR = str(prior_path.parent)
        smplify = load_file("motiongpt_smplify", fit_source / "smplify.py")
        self.body = smplx.SMPL(str(model_path), batch_size=1).to(self.device)
        self.body.requires_grad_(False)
        with h5py.File(mean_path, "r") as handle:
            self.mean_pose = torch.tensor(handle["pose"][:], dtype=torch.float32, device=self.device).reshape(1, 72)
            self.mean_shape = torch.tensor(handle["shape"][:], dtype=torch.float32, device=self.device).reshape(1, 10)
        if cfg["critic"]["optimizer"] not in ("adam", "lbfgs"):
            raise ValueError("SMPL fitting optimizer must be adam or lbfgs")
        self.fitter = smplify.SMPLify3D(self.body, batch_size=1, joints_category="AMASS",
                                       use_lbfgs=cfg["critic"]["optimizer"] == "lbfgs",
                                       num_iters=cfg["critic"]["smplify_iterations"], device=self.device)
        self.max_rmse = cfg["critic"]["max_fit_rmse_m"]

    def fit(self, joints: np.ndarray) -> dict:
        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim != 3 or joints.shape[1:] != (22, 3) or not np.isfinite(joints).all():
            raise ValueError("Expected finite HumanML3D XYZ [T,22,3] in meters, Y-up")
        pose, shape = self.mean_pose.clone(), self.mean_shape.clone()
        translation = torch.zeros((1, 3), device=self.device)
        poses, translations, errors = [], [], []
        for frame, xyz in enumerate(joints):
            target = torch.from_numpy(xyz[None]).to(self.device)
            # Fitting needs gradients even though all learned model weights are
            # frozen. Do not wrap the fitting loop in torch.no_grad().
            with torch.enable_grad():
                _, fitted, pose, shape, translation, _ = self.fitter(
                    pose, shape, translation, target, conf_3d=torch.ones(22, device=self.device), seq_ind=frame)
            rmse = ((fitted[:, :22] + translation[:, None] - target).square().mean().sqrt()).item()
            poses.append(pose.detach().cpu().numpy().reshape(24, 3))
            translations.append(translation.detach().cpu().numpy()[0])
            errors.append(rmse)
            if frame % 20 == 0:
                print(f"  SMPL fit {frame + 1}/{len(joints)}, RMSE {rmse:.4f} m", flush=True)
        errors = np.asarray(errors)
        if not np.isfinite(errors).all() or float(errors.max()) > self.max_rmse:
            raise RuntimeError(f"SMPL fitting rejected: max RMSE {errors.max():.4f} m > {self.max_rmse} m")
        return {"pose": np.asarray(poses), "translation": np.asarray(translations), "fit_rmse_m": errors}
