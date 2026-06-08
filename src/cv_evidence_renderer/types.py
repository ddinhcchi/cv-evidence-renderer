"""Core data types — `Detection`, `Event`, `ClipSource`, encoder enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


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


@dataclass
class Clip:
    """A single output evidence file: a list of `ClipSource` segments plus per-clip knobs.

    Used by `render_clips` to render many evidence files in one call. When
    several clips reference the same source file, the batch decodes that file
    once and dispatches frames to each clip's encoder.
    """

    sources: list[ClipSource]
    output: str | Path
    playback_speed: float = 1.0
    label_formatter: object = None  # actually overlay.LabelFormatter | None
    max_duration_seconds: float | None = None
    duration_strategy: str = "timelapse"


@dataclass
class ClipSource:
    """One contiguous segment of source footage feeding into a `render_clip` call.

    A clip is built from one or more `ClipSource`s in chronological order. Each
    source contributes the half-open window `[from_seconds, to_seconds)` of its
    own video file, optionally with its own detections JSONL.

    Args:
        video: Path to the source video file.
        detections: Path to a detections JSONL keyed against *this* video's
            local timeline (frame_idx 0 = first frame of `video`). `None` skips
            the overlay layer for this segment.
        from_seconds: Inclusive start, seconds from the start of `video`.
        to_seconds: Exclusive end. `None` means "decode to EOF" — useful when
            a source ends partway and the next ClipSource picks up.
    """

    video: str | Path
    detections: str | Path | None = None
    from_seconds: float = 0.0
    to_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.from_seconds < 0:
            raise ValueError(f"from_seconds must be >= 0, got {self.from_seconds}")
        if self.to_seconds is not None and self.to_seconds <= self.from_seconds:
            raise ValueError(
                f"to_seconds must be > from_seconds; got "
                f"from={self.from_seconds}, to={self.to_seconds}"
            )
