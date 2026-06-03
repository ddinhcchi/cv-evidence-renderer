"""End-to-end tests for `render_from_jsonl`."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest

from cv_evidence_renderer.encoder.libx264 import Libx264Encoder
from cv_evidence_renderer.offline import render_from_jsonl
from cv_evidence_renderer.types import Encoder

# ---------- fixtures ----------


def _write_video(
    path: Path, width: int = 64, height: int = 64, fps: int = 10, n_frames: int = 50
) -> None:
    """Write a tiny libx264 MP4 with a frame-index marker in the top-left pixel block."""
    with Libx264Encoder(path, width=width, height=height, fps=fps) as enc:
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Make each frame visually distinct (incrementing intensity)
            frame[:] = (i * 5 % 256, (i * 7) % 256, (i * 11) % 256)
            enc.write(frame)


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(d) for d in lines) + "\n", encoding="utf-8")


def _decode_count_and_dims(path: Path) -> tuple[int, int, int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return (
            sum(1 for _ in container.decode(stream)),
            stream.codec_context.width,
            stream.codec_context.height,
        )


@pytest.fixture
def video_with_jsonl(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "in.mp4"
    dets = tmp_path / "dets.jsonl"
    _write_video(video, width=64, height=64, fps=10, n_frames=50)  # 5 seconds
    _write_jsonl(
        dets,
        [
            # Detections spanning frames 8..18 (event window will be 1.0..2.0s → frames 10..20)
            {"frame_idx": fi, "bbox": [10, 10, 40, 40], "label": "person"}
            for fi in range(8, 19)
        ],
    )
    return video, dets


# ---------- happy paths ----------


def test_renders_event_window(video_with_jsonl: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = video_with_jsonl
    out = tmp_path / "evidence.mp4"

    result = render_from_jsonl(
        video=video,
        detections_jsonl=dets,
        event_start=1.0,
        event_end=2.0,
        output=out,
        encoder=Encoder.LIBX264,
    )

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0

    n_frames, w, h = _decode_count_and_dims(out)
    # 1.0s window at 10 fps → 10 frames
    assert n_frames == 10
    assert (w, h) == (64, 64)


def test_auto_encoder_resolves_to_libx264(
    video_with_jsonl: tuple[Path, Path], tmp_path: Path
) -> None:
    video, dets = video_with_jsonl
    out = tmp_path / "auto.mp4"
    render_from_jsonl(
        video=video,
        detections_jsonl=dets,
        event_start=0.5,
        event_end=1.0,
        output=out,
        encoder=Encoder.AUTO,
    )
    assert out.exists()


def test_string_encoder_accepted(video_with_jsonl: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = video_with_jsonl
    out = tmp_path / "str.mp4"
    render_from_jsonl(
        video=video,
        detections_jsonl=dets,
        event_start=0.0,
        event_end=0.5,
        output=out,
        encoder="libx264",
    )
    assert out.exists()


def test_overlays_burn_pixels(video_with_jsonl: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = video_with_jsonl
    no_overlay = tmp_path / "raw.mp4"
    with_overlay = tmp_path / "drawn.mp4"

    # Re-render the same window with no detections vs with detections.
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")

    render_from_jsonl(video, empty_jsonl, 1.0, 2.0, no_overlay)
    render_from_jsonl(video, dets, 1.0, 2.0, with_overlay)

    # The drawn version should differ from the un-drawn version (overlay changed pixels)
    assert no_overlay.read_bytes() != with_overlay.read_bytes()


def test_timestamp_keyed_jsonl(video_with_jsonl: tuple[Path, Path], tmp_path: Path) -> None:
    video, _ = video_with_jsonl
    dets = tmp_path / "ts.jsonl"
    # fps=10, so ts=1.0 → frame 10, ts=1.5 → frame 15.
    _write_jsonl(
        dets,
        [
            {"ts": 1.0, "bbox": [5, 5, 30, 30], "label": "x"},
            {"ts": 1.5, "bbox": [5, 5, 30, 30], "label": "y"},
        ],
    )
    out = tmp_path / "ts.mp4"
    render_from_jsonl(video, dets, event_start=0.5, event_end=2.0, output=out)
    assert out.exists()


def test_empty_detections_renders_clean_trim(
    video_with_jsonl: tuple[Path, Path], tmp_path: Path
) -> None:
    video, _ = video_with_jsonl
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    out = tmp_path / "no_dets.mp4"
    render_from_jsonl(video, empty, 0.0, 1.0, out)
    n_frames, _, _ = _decode_count_and_dims(out)
    assert n_frames == 10


# ---------- error paths ----------


def test_negative_event_start_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="event_start must be >= 0"):
        render_from_jsonl(tmp_path / "x.mp4", tmp_path / "x.jsonl", -0.1, 1.0, tmp_path / "o.mp4")


def test_reversed_event_window_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="event_end must be > event_start"):
        render_from_jsonl(tmp_path / "x.mp4", tmp_path / "x.jsonl", 2.0, 1.0, tmp_path / "o.mp4")


def test_zero_width_event_window_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="event_end must be > event_start"):
        render_from_jsonl(tmp_path / "x.mp4", tmp_path / "x.jsonl", 1.0, 1.0, tmp_path / "o.mp4")


def test_nvenc_not_supported_in_mvp(video_with_jsonl: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = video_with_jsonl
    with pytest.raises(NotImplementedError, match="not supported in MVP"):
        render_from_jsonl(video, dets, 0.0, 1.0, tmp_path / "o.mp4", encoder=Encoder.NVENC_H264)


def test_libx265_not_supported_in_mvp(video_with_jsonl: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = video_with_jsonl
    with pytest.raises(NotImplementedError, match="not supported in MVP"):
        render_from_jsonl(video, dets, 0.0, 1.0, tmp_path / "o.mp4", encoder=Encoder.LIBX265)


# ---------- v0.1 features ----------


def test_playback_speed_doubles_output_fps(
    video_with_jsonl: tuple[Path, Path], tmp_path: Path
) -> None:
    video, dets = video_with_jsonl
    out = tmp_path / "fast.mp4"
    render_from_jsonl(video, dets, 1.0, 2.0, out, playback_speed=2.0)

    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        # Source is 10 fps, 1 s window → 10 frames. Output keeps frame count
        # but doubles fps, so it plays back as 0.5 s.
        assert float(stream.average_rate) == pytest.approx(20.0, rel=0.01)
        n_frames = sum(1 for _ in container.decode(stream))
        assert n_frames == 10


def test_playback_speed_half_halves_output_fps(
    video_with_jsonl: tuple[Path, Path], tmp_path: Path
) -> None:
    video, dets = video_with_jsonl
    out = tmp_path / "slow.mp4"
    render_from_jsonl(video, dets, 0.0, 1.0, out, playback_speed=0.5)
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == pytest.approx(5.0, rel=0.01)


def test_playback_speed_zero_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="playback_speed must be > 0"):
        render_from_jsonl(
            tmp_path / "x.mp4",
            tmp_path / "x.jsonl",
            0.0,
            1.0,
            tmp_path / "o.mp4",
            playback_speed=0,
        )


def test_playback_speed_negative_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="playback_speed must be > 0"):
        render_from_jsonl(
            tmp_path / "x.mp4",
            tmp_path / "x.jsonl",
            0.0,
            1.0,
            tmp_path / "o.mp4",
            playback_speed=-1.0,
        )


def test_label_formatter_changes_pixels(
    video_with_jsonl: tuple[Path, Path], tmp_path: Path
) -> None:
    video, dets = video_with_jsonl
    default = tmp_path / "default.mp4"
    custom = tmp_path / "custom.mp4"

    render_from_jsonl(video, dets, 1.0, 2.0, default)
    render_from_jsonl(
        video,
        dets,
        1.0,
        2.0,
        custom,
        label_formatter=lambda d: f"!! {d.label.upper()} !!",
    )

    # Different captions → different text pixels → byte-different files.
    assert default.read_bytes() != custom.read_bytes()


def test_default_label_formatter_is_public() -> None:
    """v0.1 exposes the default formatter so users can wrap/compose it."""
    from cv_evidence_renderer.overlay import default_label_formatter
    from cv_evidence_renderer.types import Detection

    text = default_label_formatter(
        Detection(bbox=(0, 0, 10, 10), label="person", score=0.87, track_id=3)
    )
    assert text == "person #3 0.87"
