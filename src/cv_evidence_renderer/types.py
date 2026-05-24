"""Core data types — `Detection`, `Event`, encoder enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Encoder(str, Enum):
    NVENC_H264 = "nvenc_h264"
    NVENC_H265 = "nvenc_h265"
    LIBX264 = "libx264"
    LIBX265 = "libx265"
    AUTO = "auto"


@dataclass
class Detection:
    """A single detection in a single frame."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixel coords
    label: str
    score: float | None = None
    track_id: int | None = None
    color: tuple[int, int, int] | None = None  # BGR; None = derive from label hash
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class FrameDetections:
    """All detections for a single frame, keyed by either frame_idx or timestamp."""

    detections: list[Detection]
    frame_idx: int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if self.frame_idx is None and self.timestamp is None:
            raise ValueError("FrameDetections needs either frame_idx or timestamp")


@dataclass
class Event:
    """An event window over a video source — used in offline mode."""

    event_id: str
    start_seconds: float
    end_seconds: float
    label: str | None = None
    frames: list[FrameDetections] = field(default_factory=list)
