"""Offline rendering — turn a saved video + detection JSONL into an evidence MP4.

Two public entry points:

    render_clip(sources=[ClipSource(...)], output=..., ...)
        Lower-level API that will grow to support multi-source concatenation
        (one event spanning several video files) in the next milestone. Today
        it accepts exactly one ClipSource.

    render_from_jsonl(video, detections_jsonl, event_start, event_end, ...)
        Thin convenience wrapper around `render_clip` for the common
        "one event, one file" case. Backward-compatible with v0.0.1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import av
import av.container
import av.video

from cv_evidence_renderer.adapters import from_jsonl
from cv_evidence_renderer.encoder.libx264 import Libx264Encoder
from cv_evidence_renderer.overlay import LabelFormatter, draw_detections
from cv_evidence_renderer.types import ClipSource, Detection, Encoder

DurationStrategy = Literal["timelapse", "framedrop"]
_DURATION_STRATEGIES: tuple[DurationStrategy, ...] = ("timelapse", "framedrop")


def render_clip(
    sources: list[ClipSource],
    output: str | Path,
    encoder: Encoder | str = Encoder.AUTO,
    playback_speed: float = 1.0,
    label_formatter: LabelFormatter | None = None,
    max_duration_seconds: float | None = None,
    duration_strategy: DurationStrategy = "timelapse",
) -> Path:
    """Render an evidence clip from one or more `ClipSource` segments.

    The full evidence is the concatenation of each source's
    `[from_seconds, to_seconds)` window, in list order, encoded as a single MP4.

    Multi-source concatenation arrives in v0.1 milestone 14-15. Today exactly
    one source is supported; passing more raises `NotImplementedError`.

    Args:
        sources: list of `ClipSource` segments (currently exactly one).
        output: Output MP4 path.
        encoder: `Encoder` enum value or string. `AUTO` → `LIBX264` in MVP.
        playback_speed: Output playback multiplier (floor when a duration cap
            also applies). Must be > 0.
        label_formatter: Optional caption renderer per `Detection`.
        max_duration_seconds: Optional cap on output wall-clock duration.
        duration_strategy: `"timelapse"` or `"framedrop"` (see SPEC.md).

    Returns:
        Path to the written evidence MP4.

    Raises:
        ValueError: empty `sources`, bad knob values.
        NotImplementedError: more than one source (v0.1 next milestone) or an
            encoder not yet supported.
    """
    if not sources:
        raise ValueError("render_clip needs at least one ClipSource")
    if len(sources) > 1:
        raise NotImplementedError("multi-source concatenation ships in v0.1 milestone 14-15")
    if playback_speed <= 0:
        raise ValueError(f"playback_speed must be > 0, got {playback_speed}")
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise ValueError(f"max_duration_seconds must be > 0, got {max_duration_seconds}")
    if duration_strategy not in _DURATION_STRATEGIES:
        raise ValueError(
            f"duration_strategy must be one of {_DURATION_STRATEGIES}, got {duration_strategy!r}"
        )

    encoder_choice = _resolve_encoder(encoder)
    output = Path(output)

    _render_single_source(
        source=sources[0],
        output=output,
        encoder_choice=encoder_choice,
        playback_speed=playback_speed,
        label_formatter=label_formatter,
        max_duration_seconds=max_duration_seconds,
        duration_strategy=duration_strategy,
    )
    return output


def render_from_jsonl(
    video: str | Path,
    detections_jsonl: str | Path,
    event_start: float,
    event_end: float,
    output: str | Path,
    encoder: Encoder | str = Encoder.AUTO,
    playback_speed: float = 1.0,
    label_formatter: LabelFormatter | None = None,
    max_duration_seconds: float | None = None,
    duration_strategy: DurationStrategy = "timelapse",
) -> Path:
    """Render an evidence clip from one video file + one detections JSONL.

    Convenience wrapper around `render_clip`. Same knobs, same semantics; the
    only difference is the input shape (positional source/window args instead
    of a list of `ClipSource`).
    """
    # event_start/end validation happens here so the error mentions these names
    # rather than the inner ClipSource.from_seconds/to_seconds.
    if event_start < 0:
        raise ValueError(f"event_start must be >= 0, got {event_start}")
    if event_end <= event_start:
        raise ValueError(
            f"event_end must be > event_start; got start={event_start}, end={event_end}"
        )

    return render_clip(
        sources=[
            ClipSource(
                video=video,
                detections=detections_jsonl,
                from_seconds=event_start,
                to_seconds=event_end,
            )
        ],
        output=output,
        encoder=encoder,
        playback_speed=playback_speed,
        label_formatter=label_formatter,
        max_duration_seconds=max_duration_seconds,
        duration_strategy=duration_strategy,
    )


# ----------------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------------


def _render_single_source(
    source: ClipSource,
    output: Path,
    encoder_choice: Encoder,
    playback_speed: float,
    label_formatter: LabelFormatter | None,
    max_duration_seconds: float | None,
    duration_strategy: DurationStrategy,
) -> None:
    container = av.open(str(source.video))
    try:
        stream = container.streams.video[0]
        if stream.average_rate is None or float(stream.average_rate) <= 0:
            raise ValueError(f"could not determine fps of {source.video}")
        source_fps = float(stream.average_rate)
        width = stream.codec_context.width
        height = stream.codec_context.height

        start_frame = round(source.from_seconds * source_fps)
        end_frame: int | None = (
            round(source.to_seconds * source_fps) if source.to_seconds is not None else None
        )
        window_seconds = (
            (source.to_seconds - source.from_seconds) if source.to_seconds is not None else None
        )

        effective_speed, sample_stride = _resolve_duration_cap(
            window_seconds=window_seconds,
            playback_speed=playback_speed,
            max_duration_seconds=max_duration_seconds,
            duration_strategy=duration_strategy,
        )
        output_fps = source_fps * effective_speed

        detections_by_frame = (
            _index_detections_by_frame(source.detections, source_fps)
            if source.detections is not None
            else {}
        )

        if encoder_choice == Encoder.LIBX264:
            encoder_obj = Libx264Encoder(output, width=width, height=height, fps=output_fps)
        else:  # defensive
            raise NotImplementedError(f"encoder {encoder_choice} not supported in MVP")

        with encoder_obj:
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx < start_frame:
                    continue
                if end_frame is not None and frame_idx >= end_frame:
                    break
                if (frame_idx - start_frame) % sample_stride != 0:
                    continue
                bgr = frame.to_ndarray(format="bgr24")
                draw_detections(
                    bgr,
                    detections_by_frame.get(frame_idx, []),
                    label_formatter=label_formatter,
                )
                encoder_obj.write(bgr)
    finally:
        container.close()


def _resolve_encoder(encoder: Encoder | str) -> Encoder:
    """Normalise an encoder choice and reject ones not available in MVP."""
    choice = Encoder(encoder) if isinstance(encoder, str) else encoder
    if choice == Encoder.AUTO:
        return Encoder.LIBX264
    if choice == Encoder.LIBX264:
        return choice
    raise NotImplementedError(f"encoder {choice.value} is not supported in MVP — use libx264")


def _resolve_duration_cap(
    window_seconds: float | None,
    playback_speed: float,
    max_duration_seconds: float | None,
    duration_strategy: DurationStrategy,
) -> tuple[float, int]:
    """Return (effective_playback_speed, sample_stride) honouring the duration cap.

    `playback_speed` is treated as a floor — we never make the clip play *slower*
    than the caller explicitly asked for, even to stay under the cap.

    When `window_seconds` is unknown (`to_seconds=None`) the cap is ignored —
    we can't know in advance how long the source will be.
    """
    if max_duration_seconds is None or window_seconds is None:
        return playback_speed, 1
    effective_duration = window_seconds / playback_speed
    if effective_duration <= max_duration_seconds:
        return playback_speed, 1

    if duration_strategy == "timelapse":
        return window_seconds / max_duration_seconds, 1
    # framedrop
    stride = math.ceil(effective_duration / max_duration_seconds)
    return playback_speed, stride


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
