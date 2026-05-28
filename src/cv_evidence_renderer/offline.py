"""`render_from_jsonl` — offline batch render from saved video + detection JSONL.

This is the MVP entrypoint: read a video file and a JSONL of detections,
trim to the event window, burn in bboxes/labels, and encode the result.
"""

from __future__ import annotations

from pathlib import Path

import av
import av.container
import av.video

from cv_evidence_renderer.adapters import from_jsonl
from cv_evidence_renderer.encoder.libx264 import Libx264Encoder
from cv_evidence_renderer.overlay import draw_detections
from cv_evidence_renderer.types import Detection, Encoder


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
        video: Path to input video file (anything PyAV can decode).
        detections_jsonl: Path to JSONL file with one detection per line.
        event_start: Start of event window, seconds from video start.
        event_end: End of event window, seconds from video start. Exclusive.
        output: Output MP4 path.
        encoder: One of the `Encoder` enum values. `AUTO` resolves to
            `LIBX264` in MVP; NVENC variants raise `NotImplementedError`.

    Returns:
        Path to the written evidence MP4.

    Raises:
        ValueError: invalid event window.
        NotImplementedError: requested encoder is not available in MVP.
    """
    if event_start < 0:
        raise ValueError(f"event_start must be >= 0, got {event_start}")
    if event_end <= event_start:
        raise ValueError(
            f"event_end must be > event_start; got start={event_start}, end={event_end}"
        )

    encoder_choice = _resolve_encoder(encoder)
    output = Path(output)

    container = av.open(str(video))
    try:
        stream = container.streams.video[0]
        if stream.average_rate is None or float(stream.average_rate) <= 0:
            raise ValueError(f"could not determine fps of {video}")
        fps = float(stream.average_rate)
        width = stream.codec_context.width
        height = stream.codec_context.height

        start_frame = round(event_start * fps)
        end_frame = round(event_end * fps)  # exclusive

        detections_by_frame = _index_detections_by_frame(detections_jsonl, fps)

        if encoder_choice == Encoder.LIBX264:
            encoder_obj = Libx264Encoder(output, width=width, height=height, fps=fps)
        else:  # defensive — _resolve_encoder should have raised already
            raise NotImplementedError(f"encoder {encoder_choice} not supported in MVP")

        with encoder_obj:
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx < start_frame:
                    continue
                if frame_idx >= end_frame:
                    break
                bgr = frame.to_ndarray(format="bgr24")
                draw_detections(bgr, detections_by_frame.get(frame_idx, []))
                encoder_obj.write(bgr)
    finally:
        container.close()

    return output


def _resolve_encoder(encoder: Encoder | str) -> Encoder:
    """Normalise an encoder choice and reject ones not available in MVP."""
    choice = Encoder(encoder) if isinstance(encoder, str) else encoder
    if choice == Encoder.AUTO:
        return Encoder.LIBX264
    if choice == Encoder.LIBX264:
        return choice
    raise NotImplementedError(f"encoder {choice.value} is not supported in MVP — use libx264")


def _index_detections_by_frame(
    detections_jsonl: str | Path,
    fps: float,
) -> dict[int, list[Detection]]:
    """Load detections and index them by frame number.

    Timestamp-keyed detections are snapped to the nearest frame via `round(ts * fps)`.
    """
    by_frame: dict[int, list[Detection]] = {}
    for fd in from_jsonl(detections_jsonl):
        if fd.frame_idx is not None:
            key = fd.frame_idx
        elif fd.timestamp is not None:
            key = round(fd.timestamp * fps)
        else:  # FrameDetections.__post_init__ already guarantees at least one is set
            continue
        by_frame.setdefault(key, []).extend(fd.detections)
    return by_frame
