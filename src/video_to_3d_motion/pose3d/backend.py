"""Released PoseMamba-L inference with strict weights and explicit units."""

from pathlib import Path
from time import perf_counter

import numpy as np

from video_to_3d_motion.pose_adapter.models import Pose2DWindow

from .config import ModelSettings, UPSTREAM_REVISION, WEIGHTS_REVISION
from .exceptions import Pose3DError
from .models import Pose3DWindow

LEFT = [4, 5, 6, 11, 12, 13]
RIGHT = [1, 2, 3, 14, 15, 16]


def file_sha256(path: str | Path) -> str:
    # Python 3.10 support, without reading an entire model/annotation into RAM.
    from hashlib import sha256
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flip_h36m(points):
    flipped = points.clone() if hasattr(points, "clone") else points.copy()
    flipped[..., 0] *= -1
    flipped[..., LEFT + RIGHT, :] = flipped[..., RIGHT + LEFT, :].clone() if hasattr(flipped, "clone") else flipped[..., RIGHT + LEFT, :].copy()
    return flipped


class PoseMambaBackend:
    def __init__(self, settings: ModelSettings):
        self.settings = settings
        checkpoint = Path(settings.checkpoint)
        if not checkpoint.is_file():
            raise Pose3DError("checkpoint_missing", f"missing checkpoint: {checkpoint}; run tools/setup_posemamba.ps1")
        digest = file_sha256(checkpoint)
        if digest != settings.checkpoint_sha256:
            raise Pose3DError("checkpoint_mismatch", f"PoseMamba-L checkpoint SHA256 mismatch: {digest}")
        try:
            import torch
            from .vendor.model import PoseMamba
        except ImportError as exc:
            raise Pose3DError("dependency_missing", "install the pose3d extra or run tools/setup_posemamba.ps1") from exc
        self.torch = torch
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise Pose3DError("device_unavailable", "CUDA was requested but is unavailable")
        started = perf_counter()
        try:
            # The immutable, SHA-verified official file includes optimizer metadata.
            loaded = torch.load(checkpoint, map_location="cpu")
            state = loaded["model_pos"]
            state = {key.removeprefix("module."): value for key, value in state.items()}
            model = PoseMamba(num_frame=243, num_joints=17, in_chans=2,
                              embed_dim_ratio=128, depth=20, mlp_ratio=2)
            model.load_state_dict(state, strict=True)
            self.model = model.to(settings.device).eval()
            del loaded, state
            if settings.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except (RuntimeError, KeyError, TypeError) as exc:
            raise Pose3DError("model_load_failed", str(exc)) from exc
        self.model_info = dict(
            model="PoseMamba-L", checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
            upstream_revision=UPSTREAM_REVISION, weights_revision=WEIGHTS_REVISION,
            backend="torch_affine_prefix_scan", device=settings.device, torch_version=str(torch.__version__),
            flip_test=settings.flip_test, clip_len=243, dim_feat=128, depth=20,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            coord_system="root_relative_image_3d", units="h36m_image_normalized",
            load_seconds=perf_counter() - started,
        )

    def predict(self, window: Pose2DWindow) -> Pose3DWindow:
        if (not isinstance(window, Pose2DWindow) or window.keypoints_norm.shape != (1, 243, 17, 2)
                or window.keypoints_norm.dtype != np.float32 or not np.isfinite(window.keypoints_norm).all()
                or window.joint_schema != "h36m17" or window.joint_mapping != "identity"
                or window.joint_order_id != "h36m17_rtmpose_h36m_v1"
                or window.normalization_id != "h36m_width_center_v1"
                or window.coordinate_space != "h36m_screen_normalized"):
            raise Pose3DError("invalid_window", "expected a verified normalized H36M17 243-frame window")
        for name in ("frame_ids", "source_frame_ids", "timestamps", "source_timestamps"):
            values = getattr(window, name)
            expected_dtype = np.int64 if name.endswith("frame_ids") else np.float64
            if (values.shape != (243,) or values.dtype != expected_dtype
                    or not np.isfinite(values).all() or not (np.diff(values) > 0).all()):
                raise Pose3DError("invalid_window", f"invalid {name} metadata")
        if (window.keypoint_scores_raw.shape != (243, 17) or window.keypoint_scores_raw.dtype != np.float32
                or not np.isfinite(window.keypoint_scores_raw).all()
                or window.valid_mask.shape != (243, 17) or window.valid_mask.dtype != np.bool_
                or len(window.timestamp_sources) != 243):
            raise Pose3DError("invalid_window", "invalid per-frame quality metadata")
        torch = self.torch
        started = perf_counter()
        try:
            with torch.inference_mode():
                inputs = torch.from_numpy(np.ascontiguousarray(window.keypoints_norm)).to(self.settings.device)
                result = self.model(inputs)
                if self.settings.flip_test:
                    result_flip = self.model(flip_h36m(inputs))
                    result = (result + flip_h36m(result_flip)) / 2
                # Matches official rootrel evaluation; other joints are already
                # trained root-relative and must not inherit root prediction noise.
                result[:, :, 0, :] = 0
                output = np.ascontiguousarray(result[0].cpu().numpy(), dtype=np.float32)
        except torch.cuda.OutOfMemoryError as exc:
            raise Pose3DError("pose3d_oom", "PoseMamba inference exhausted GPU memory at batch=1") from exc
        except RuntimeError as exc:
            raise Pose3DError("pose3d_inference_failed", str(exc)) from exc
        if output.shape != (243, 17, 3) or not np.isfinite(output).all():
            raise Pose3DError("invalid_model_output", "PoseMamba returned an invalid or nonfinite result")
        return Pose3DWindow(output, window, dict(self.model_info), perf_counter() - started)

    def runtime_info(self):
        result = dict(self.model_info)
        if self.settings.device.startswith("cuda"):
            result["cuda_peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = self.torch.cuda.max_memory_reserved()
        return result
