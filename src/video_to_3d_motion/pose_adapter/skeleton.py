"""The audited RTMPose checkpoint uses identity H36M17 joint ordering."""

from video_to_3d_motion.pose2d.models import Pose2DFrame

from .exceptions import PoseAdapterError

H36M17_JOINT_NAMES = (
    "root", "right_hip", "right_knee", "right_foot", "left_hip", "left_knee",
    "left_foot", "spine", "thorax", "neck_base", "head", "left_shoulder",
    "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
)
# Pin the audited weights; a new checkpoint needs a new verified joint profile.
VERIFIED_CHECKPOINT_SHA256 = "3c0b3796b0405a8a105e2a379db4a6552b0a4270f9acb6626c8163d694bd67dc"


def validate_joint_profile(pose: Pose2DFrame) -> None:
    if pose.joint_schema != "h36m17":
        raise PoseAdapterError("joint_schema_unsupported", f"unsupported joints: {pose.joint_schema}")
    if (pose.model != "rtmpose_l_h36m" or pose.score_source != "rtmpose_simcc"
            or pose.checkpoint_sha256 != VERIFIED_CHECKPOINT_SHA256):
        raise PoseAdapterError("joint_profile_mismatch", "input is not the audited RTMPose H36M checkpoint")
