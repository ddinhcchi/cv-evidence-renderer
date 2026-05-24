"""`render_from_jsonl` — offline batch render from saved video + detection JSONL (USE CASE B / MVP).

This is the MVP entrypoint. See SPEC.md §8 MVP.
"""

from __future__ import annotations

from pathlib import Path

from cv_evidence_renderer.types import Encoder


def render_from_jsonl(
    video: str | Path,
    detections_jsonl: str | Path,
    event_start: float,
    event_end: float,
    output: str | Path,
    encoder: Encoder | str = Encoder.AUTO,
) -> Path:
    """Render an evidence clip from a saved video + JSONL detections.

    Args:
        video: Path to input video file (MP4 or anything PyAV can decode).
        detections_jsonl: Path to JSONL file with one detection per line.
        event_start: Start of event window, seconds from video start.
        event_end: End of event window, seconds from video start.
        output: Output MP4 path.
        encoder: NVENC H264/H265, libx264/libx265, or "auto".

    Returns:
        Path to the written evidence MP4.
    """
    raise NotImplementedError("MVP — see SPEC.md §8")
