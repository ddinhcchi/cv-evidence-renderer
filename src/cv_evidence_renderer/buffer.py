"""Ring buffer for pre-event frames. Keyframe-aware seek so flushed clips start cleanly."""

from __future__ import annotations


class FrameRingBuffer:
    """Holds the last N seconds of decoded frames + their detections.

    On `flush()`, returns frames snapped to the most recent keyframe ≥ `oldest_kept_ts`.
    """

    def __init__(self, capacity_seconds: float, fps: float) -> None:
        self.capacity_seconds = capacity_seconds
        self.fps = fps
        self.capacity_frames = int(capacity_seconds * fps)

    def push(self, frame_idx: int, frame: object, detections: object, is_keyframe: bool) -> None:
        raise NotImplementedError("v0.2")

    def flush(self) -> list[object]:
        raise NotImplementedError("v0.2")
