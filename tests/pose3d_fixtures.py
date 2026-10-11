"""Shared input windows for S1-04 and S1-05 behavior checks."""

from dataclasses import replace
from tests.test_pose_adapter import frame, terminal
from video_to_3d_motion.pose_adapter.config import PoseAdapterConfig
from video_to_3d_motion.pose_adapter.window_buffer import WindowBuffer


def input_windows(count=472):
    buffer = WindowBuffer(PoseAdapterConfig())
    windows = []
    for number in range(1, count + 1):
        buffer.push(frame(number))
        if number >= 243 and (number - 243) % 81 == 0:
            windows.append(buffer.make_window(window_index=len(windows) + 1, pipeline_mode="full"))
    tail = buffer.finish(terminal(count))
    if not windows or windows[-1].window_id != tail.window_id:
        windows.append(replace(tail, window_index=len(windows) + 1, pipeline_mode="full"))
    return windows
