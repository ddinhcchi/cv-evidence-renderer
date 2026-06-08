"""Offline rendering — turn a saved video + detection JSONL into an evidence MP4.

Two public entry points:

    render_clip(sources=[ClipSource(...), ...], output=..., ...)
        Lower-level API. The output is the concatenation of each source's
        `[from_seconds, to_seconds)` window, encoded as a single MP4. All
        sources must share width, height, and (to within 1%) fps.

    render_from_jsonl(video, detections_jsonl, event_start, event_end, ...)
        Thin convenience wrapper around `render_clip` for the common
        "one event, one file" case. Backward-compatible with v0.0.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import av.container
import av.video

from cv_evidence_renderer.adapters import from_jsonl
from cv_evidence_renderer.encoder.libx264 import Libx264Encoder
from cv_evidence_renderer.overlay import LabelFormatter, draw_detections
from cv_evidence_renderer.types import Clip, ClipSource, Detection, Encoder

DurationStrategy = Literal["timelapse", "framedrop"]
_DURATION_STRATEGIES: tuple[DurationStrategy, ...] = ("timelapse", "framedrop")
_FPS_TOLERANCE = 0.01  # 1% relative tolerance when comparing source fps values


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

    The output is the concatenation of each source's `[from_seconds, to_seconds)`
    window, in list order, encoded as a single MP4. All sources must share
    `width`, `height`, and (to within 1%) fps — otherwise the encoder can't
    write a single stream without resampling, which is out of scope.

    Args:
        sources: list of `ClipSource` segments. Must be non-empty.
        output: Output MP4 path.
        encoder: `Encoder` enum value or string. `AUTO` → `LIBX264` in MVP.
        playback_speed: Output playback multiplier (floor when a duration cap
            also applies). Must be > 0.
        label_formatter: Optional caption renderer per `Detection`.
        max_duration_seconds: Optional cap on output wall-clock duration. The
            cap is computed against the *total* window across all sources;
            ignored if any source has `to_seconds=None`.
        duration_strategy: `"timelapse"` or `"framedrop"`.

    Returns:
        Path to the written evidence MP4.

    Raises:
        ValueError: empty `sources`, bad knob values, or incompatible source
            metadata (dims/fps).
        NotImplementedError: encoder not yet supported.
    """
    if not sources:
        raise ValueError("render_clip needs at least one ClipSource")
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

    metas = [_probe_source(src) for src in sources]
    _check_sources_compatible(metas)
    canonical = metas[0]

    total_window = _total_window_seconds(sources)
    effective_speed, sample_stride = _resolve_duration_cap(
        window_seconds=total_window,
        playback_speed=playback_speed,
        max_duration_seconds=max_duration_seconds,
        duration_strategy=duration_strategy,
    )
    output_fps = canonical.fps * effective_speed

    if encoder_choice != Encoder.LIBX264:  # defensive — _resolve_encoder should have raised
        raise NotImplementedError(f"encoder {encoder_choice} not supported in MVP")

    encoder_obj = Libx264Encoder(
        output, width=canonical.width, height=canonical.height, fps=output_fps
    )

    with encoder_obj:
        # Single counter across all sources so the framedrop stride is uniform.
        kept_input_index = 0
        for src, meta in zip(sources, metas, strict=True):
            kept_input_index = _decode_source_into_encoder(
                source=src,
                meta=meta,
                encoder_obj=encoder_obj,
                label_formatter=label_formatter,
                sample_stride=sample_stride,
                kept_input_index=kept_input_index,
            )

    return output


def render_clips(
    clips: list[Clip],
    encoder: Encoder | str = Encoder.AUTO,
) -> list[Path]:
    """Render many evidence clips in a single batch call.

    Returns the output paths in the same order as `clips`.

    When several clips reference the same source video file (the very common
    "1 long recording, N events" case) the batch opens and decodes that file
    only once, dispatching each decoded frame to every clip that wants it.
    Clips with multiple sources or with a unique source path fall back to the
    regular `render_clip` path.

    Args:
        clips: list of `Clip` objects (must be non-empty).
        encoder: shared encoder choice for every clip.

    Raises:
        ValueError: empty `clips`, or per-clip knob validation upstream.
    """
    if not clips:
        raise ValueError("render_clips needs at least one Clip")

    encoder_choice = _resolve_encoder(encoder)

    # Index clips by their order so we can return paths in input order.
    outputs: dict[int, Path] = {}

    # Group single-source clips by their source path; everything else is rendered standalone.
    shared: dict[Path, list[tuple[int, Clip]]] = {}
    standalone: list[tuple[int, Clip]] = []
    for i, clip in enumerate(clips):
        if len(clip.sources) == 1:
            key = Path(clip.sources[0].video).resolve()
            shared.setdefault(key, []).append((i, clip))
        else:
            standalone.append((i, clip))

    # Run standalone (multi-source) clips through the regular path.
    for i, clip in standalone:
        outputs[i] = render_clip(
            sources=clip.sources,
            output=clip.output,
            encoder=encoder,
            playback_speed=clip.playback_speed,
            label_formatter=clip.label_formatter,  # type: ignore[arg-type]
            max_duration_seconds=clip.max_duration_seconds,
            duration_strategy=clip.duration_strategy,  # type: ignore[arg-type]
        )

    # For each unique single-source path: decode once if shared, otherwise standalone.
    for source_path, group in shared.items():
        if len(group) == 1:
            i, clip = group[0]
            outputs[i] = render_clip(
                sources=clip.sources,
                output=clip.output,
                encoder=encoder,
                playback_speed=clip.playback_speed,
                label_formatter=clip.label_formatter,  # type: ignore[arg-type]
                max_duration_seconds=clip.max_duration_seconds,
                duration_strategy=clip.duration_strategy,  # type: ignore[arg-type]
            )
        else:
            paths = _render_shared_source(source_path, group, encoder_choice)
            outputs.update(paths)

    return [outputs[i] for i in range(len(clips))]


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


@dataclass(frozen=True)
class _SourceMeta:
    fps: float
    width: int
    height: int


def _probe_source(src: ClipSource) -> _SourceMeta:
    """Open the source just long enough to read width/height/fps, then close."""
    with av.open(str(src.video)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None or float(stream.average_rate) <= 0:
            raise ValueError(f"could not determine fps of {src.video}")
        return _SourceMeta(
            fps=float(stream.average_rate),
            width=stream.codec_context.width,
            height=stream.codec_context.height,
        )


def _check_sources_compatible(metas: list[_SourceMeta]) -> None:
    first = metas[0]
    for i, m in enumerate(metas[1:], start=1):
        if (m.width, m.height) != (first.width, first.height):
            raise ValueError(
                f"source {i} dimensions ({m.width}x{m.height}) do not match "
                f"source 0 ({first.width}x{first.height}) — resizing is out of scope"
            )
        if abs(m.fps - first.fps) / first.fps > _FPS_TOLERANCE:
            raise ValueError(
                f"source {i} fps ({m.fps:.3f}) differs from source 0 ({first.fps:.3f}) "
                f"by more than {_FPS_TOLERANCE * 100:.0f}% — resampling is out of scope"
            )


def _total_window_seconds(sources: list[ClipSource]) -> float | None:
    """Sum of all source windows in seconds, or None if any source has open end."""
    total = 0.0
    for src in sources:
        if src.to_seconds is None:
            return None
        total += src.to_seconds - src.from_seconds
    return total


def _decode_source_into_encoder(
    source: ClipSource,
    meta: _SourceMeta,
    encoder_obj: Libx264Encoder,
    label_formatter: LabelFormatter | None,
    sample_stride: int,
    kept_input_index: int,
) -> int:
    """Decode one source's window and feed kept frames to the encoder.

    Returns the updated `kept_input_index` so the next source continues the
    framedrop stride seamlessly.
    """
    start_frame = round(source.from_seconds * meta.fps)
    end_frame: int | None = (
        round(source.to_seconds * meta.fps) if source.to_seconds is not None else None
    )
    detections_by_frame = (
        _index_detections_by_frame(source.detections, meta.fps)
        if source.detections is not None
        else {}
    )

    container = av.open(str(source.video))
    try:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx < start_frame:
                continue
            if end_frame is not None and frame_idx >= end_frame:
                break
            if kept_input_index % sample_stride != 0:
                kept_input_index += 1
                continue
            bgr = frame.to_ndarray(format="bgr24")
            draw_detections(
                bgr,
                detections_by_frame.get(frame_idx, []),
                label_formatter=label_formatter,
            )
            encoder_obj.write(bgr)
            kept_input_index += 1
    finally:
        container.close()

    return kept_input_index


def _render_shared_source(
    source_path: Path,
    group: list[tuple[int, Clip]],
    encoder_choice: Encoder,
) -> dict[int, Path]:
    """Decode `source_path` exactly once, dispatch frames to every clip in `group`.

    Each clip gets its own encoder, its own framedrop stride counter, and its
    own overlay (label_formatter + per-source detections). The decoded frame is
    copied per consumer so overlays don't bleed across outputs.
    """
    # Probe metadata once. All clips share width/height/source_fps because they
    # share the source file; per-clip output fps still differs via playback_speed.
    meta = _probe_source(ClipSource(video=source_path))

    if encoder_choice != Encoder.LIBX264:  # defensive
        raise NotImplementedError(f"encoder {encoder_choice} not supported in MVP")

    # Build per-clip state.
    states: list[_ClipState] = []
    for clip_index, clip in group:
        src = clip.sources[0]
        window_seconds = (src.to_seconds - src.from_seconds) if src.to_seconds is not None else None
        effective_speed, sample_stride = _resolve_duration_cap(
            window_seconds=window_seconds,
            playback_speed=clip.playback_speed,
            max_duration_seconds=clip.max_duration_seconds,
            duration_strategy=clip.duration_strategy,  # type: ignore[arg-type]
        )
        detections_by_frame = (
            _index_detections_by_frame(src.detections, meta.fps)
            if src.detections is not None
            else {}
        )
        encoder_obj = Libx264Encoder(
            clip.output, width=meta.width, height=meta.height, fps=meta.fps * effective_speed
        )
        encoder_obj.open()
        states.append(
            _ClipState(
                clip_index=clip_index,
                output_path=Path(clip.output),
                label_formatter=clip.label_formatter,  # type: ignore[arg-type]
                start_frame=round(src.from_seconds * meta.fps),
                end_frame=(
                    round(src.to_seconds * meta.fps) if src.to_seconds is not None else None
                ),
                stride=sample_stride,
                in_window_count=0,
                detections_by_frame=detections_by_frame,
                encoder=encoder_obj,
            )
        )

    # Single decode pass.
    container = av.open(str(source_path))
    try:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            consumers = [
                s
                for s in states
                if s.start_frame <= frame_idx and (s.end_frame is None or frame_idx < s.end_frame)
            ]
            if not consumers:
                # Early-exit when every clip is past its end_frame.
                if all(s.end_frame is not None and frame_idx >= s.end_frame for s in states):
                    break
                continue

            bgr_original = frame.to_ndarray(format="bgr24")
            for state in consumers:
                if state.in_window_count % state.stride != 0:
                    state.in_window_count += 1
                    continue
                bgr = bgr_original.copy()  # don't let overlays bleed across encoders
                draw_detections(
                    bgr,
                    state.detections_by_frame.get(frame_idx, []),
                    label_formatter=state.label_formatter,
                )
                state.encoder.write(bgr)
                state.in_window_count += 1
    finally:
        container.close()
        for state in states:
            state.encoder.close()

    return {state.clip_index: state.output_path for state in states}


@dataclass
class _ClipState:
    """Per-clip mutable bookkeeping during a shared-source decode pass."""

    clip_index: int
    output_path: Path
    label_formatter: LabelFormatter | None
    start_frame: int
    end_frame: int | None
    stride: int
    in_window_count: int
    detections_by_frame: dict[int, list[Detection]]
    encoder: Libx264Encoder


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

    When `window_seconds` is unknown (any source has `to_seconds=None`) the cap
    is ignored — we can't know in advance how long the source will be.
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
