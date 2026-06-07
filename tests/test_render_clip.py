"""Tests for the lower-level `render_clip` API."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest

from cv_evidence_renderer import ClipSource, render_clip
from cv_evidence_renderer.encoder.libx264 import Libx264Encoder


def _write_video(
    path: Path, *, width: int = 64, height: int = 64, fps: int = 10, n: int = 50
) -> None:
    with Libx264Encoder(path, width=width, height=height, fps=fps) as enc:
        for i in range(n):
            frame = np.full((height, width, 3), (i * 5 % 256, 80, 120), dtype=np.uint8)
            enc.write(frame)


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(d) for d in lines) + "\n", encoding="utf-8")


def _decode_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


@pytest.fixture
def src_with_dets(tmp_path: Path) -> tuple[Path, Path]:
    v = tmp_path / "src.mp4"
    d = tmp_path / "src.jsonl"
    _write_video(v, n=50)
    _write_jsonl(d, [{"frame_idx": fi, "bbox": [10, 10, 40, 40], "label": "x"} for fi in range(50)])
    return v, d


def test_single_source_renders(src_with_dets: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = src_with_dets
    out = tmp_path / "ev.mp4"
    result = render_clip(
        sources=[ClipSource(video=video, detections=dets, from_seconds=1.0, to_seconds=2.0)],
        output=out,
    )
    assert result == out
    assert out.exists()
    assert _decode_count(out) == 10  # 1 s @ 10 fps


def test_source_without_detections(src_with_dets: tuple[Path, Path], tmp_path: Path) -> None:
    """A ClipSource with detections=None should just trim — no overlay layer."""
    video, dets = src_with_dets
    no_overlay = tmp_path / "raw.mp4"
    with_overlay = tmp_path / "drawn.mp4"
    render_clip(
        sources=[ClipSource(video=video, detections=None, from_seconds=1.0, to_seconds=2.0)],
        output=no_overlay,
    )
    render_clip(
        sources=[ClipSource(video=video, detections=dets, from_seconds=1.0, to_seconds=2.0)],
        output=with_overlay,
    )
    # Overlay must change pixels → byte-different files.
    assert no_overlay.read_bytes() != with_overlay.read_bytes()
    # Both should have the same frame count though.
    assert _decode_count(no_overlay) == _decode_count(with_overlay) == 10


def test_to_seconds_none_renders_to_eof(src_with_dets: tuple[Path, Path], tmp_path: Path) -> None:
    """`to_seconds=None` means decode until the source ends."""
    video, dets = src_with_dets
    out = tmp_path / "eof.mp4"
    render_clip(
        sources=[ClipSource(video=video, detections=dets, from_seconds=4.0, to_seconds=None)],
        output=out,
    )
    # Source is 5 s @ 10 fps = 50 frames; from_seconds=4 → start at frame 40 → 10 frames left.
    assert _decode_count(out) == 10


def test_empty_sources_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs at least one ClipSource"):
        render_clip(sources=[], output=tmp_path / "o.mp4")


# ---------- multi-source concat ----------


def test_concat_two_segments_frame_count(src_with_dets: tuple[Path, Path], tmp_path: Path) -> None:
    """Two windows from the same file → output frame count = sum of windows."""
    video, dets = src_with_dets
    out = tmp_path / "concat.mp4"
    render_clip(
        sources=[
            ClipSource(video=video, detections=dets, from_seconds=0.0, to_seconds=1.0),
            ClipSource(video=video, detections=dets, from_seconds=3.0, to_seconds=4.5),
        ],
        output=out,
    )
    # 1 s + 1.5 s @ 10 fps = 10 + 15 = 25 frames.
    assert _decode_count(out) == 25


def test_concat_two_distinct_files(tmp_path: Path) -> None:
    """Two physically separate source files with matching dims/fps concat cleanly."""
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_video(a, n=30)
    _write_video(b, n=20)
    out = tmp_path / "concat.mp4"
    render_clip(
        sources=[
            ClipSource(video=a, from_seconds=2.0, to_seconds=3.0),  # 10 frames
            ClipSource(video=b, from_seconds=0.0, to_seconds=1.5),  # 15 frames
        ],
        output=out,
    )
    assert _decode_count(out) == 25


def test_concat_width_mismatch_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_video(a, width=64, height=64, n=20)
    _write_video(b, width=128, height=64, n=20)
    with pytest.raises(ValueError, match="dimensions"):
        render_clip(
            sources=[
                ClipSource(video=a, from_seconds=0, to_seconds=1),
                ClipSource(video=b, from_seconds=0, to_seconds=1),
            ],
            output=tmp_path / "o.mp4",
        )


def test_concat_fps_mismatch_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_video(a, fps=10, n=20)
    _write_video(b, fps=30, n=20)
    with pytest.raises(ValueError, match="fps"):
        render_clip(
            sources=[
                ClipSource(video=a, from_seconds=0, to_seconds=1),
                ClipSource(video=b, from_seconds=0, to_seconds=0.5),
            ],
            output=tmp_path / "o.mp4",
        )


def test_concat_mixes_detections_with_none(tmp_path: Path) -> None:
    """One source has detections, the next has none — both render, no crash."""
    video = tmp_path / "v.mp4"
    dets = tmp_path / "d.jsonl"
    _write_video(video, n=50)
    _write_jsonl(
        dets, [{"frame_idx": fi, "bbox": [5, 5, 30, 30], "label": "x"} for fi in range(50)]
    )
    out = tmp_path / "mixed.mp4"
    render_clip(
        sources=[
            ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=1),
            ClipSource(video=video, detections=None, from_seconds=2, to_seconds=3),
        ],
        output=out,
    )
    assert _decode_count(out) == 20


def test_concat_max_duration_uses_total_window(
    src_with_dets: tuple[Path, Path], tmp_path: Path
) -> None:
    """Cap is computed against sum of all source windows, not per-source."""
    video, dets = src_with_dets
    out = tmp_path / "cap.mp4"
    # 2 s + 2 s = 4 s total, cap 1 s, framedrop → keep every 4th frame.
    render_clip(
        sources=[
            ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=2),
            ClipSource(video=video, detections=dets, from_seconds=3, to_seconds=5),
        ],
        output=out,
        max_duration_seconds=1.0,
        duration_strategy="framedrop",
    )
    # 40 input frames, stride 4 → 10 kept.
    assert _decode_count(out) == 10


def test_concat_max_duration_timelapse_scales_fps(
    src_with_dets: tuple[Path, Path], tmp_path: Path
) -> None:
    video, dets = src_with_dets
    out = tmp_path / "tl.mp4"
    # 4 s total, cap 1 s → speed 4x, fps 10 → 40.
    render_clip(
        sources=[
            ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=2),
            ClipSource(video=video, detections=dets, from_seconds=3, to_seconds=5),
        ],
        output=out,
        max_duration_seconds=1.0,
        duration_strategy="timelapse",
    )
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == pytest.approx(40.0, rel=0.02)
        assert sum(1 for _ in container.decode(stream)) == 40


def test_concat_last_source_to_eof(tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_video(a, n=20)
    _write_video(b, n=15)
    out = tmp_path / "eof.mp4"
    render_clip(
        sources=[
            ClipSource(video=a, from_seconds=1.0, to_seconds=2.0),  # 10 frames
            ClipSource(video=b, from_seconds=0.5, to_seconds=None),  # 10 frames to EOF
        ],
        output=out,
    )
    assert _decode_count(out) == 20


def test_clipsource_validates_window() -> None:
    with pytest.raises(ValueError, match="to_seconds must be > from_seconds"):
        ClipSource(video="x.mp4", from_seconds=2.0, to_seconds=1.0)
    with pytest.raises(ValueError, match="from_seconds must be >= 0"):
        ClipSource(video="x.mp4", from_seconds=-0.1)


def test_render_clip_honours_playback_speed(
    src_with_dets: tuple[Path, Path], tmp_path: Path
) -> None:
    video, dets = src_with_dets
    out = tmp_path / "fast.mp4"
    render_clip(
        sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=2)],
        output=out,
        playback_speed=2.0,
    )
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == pytest.approx(20.0, rel=0.02)


def test_render_clip_honours_max_duration(src_with_dets: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = src_with_dets
    out = tmp_path / "cap.mp4"
    render_clip(
        sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=4)],
        output=out,
        max_duration_seconds=1.0,
        duration_strategy="framedrop",
    )
    # 4 s @ 10 fps = 40 frames sampled to fit 1 s @ 10 fps = 10 frames.
    assert _decode_count(out) == 10


def test_max_duration_ignored_when_to_seconds_is_none(
    src_with_dets: tuple[Path, Path], tmp_path: Path
) -> None:
    """Without an explicit window end, we can't compute the cap ratio; pass through."""
    video, dets = src_with_dets
    out = tmp_path / "noend.mp4"
    render_clip(
        sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=None)],
        output=out,
        max_duration_seconds=1.0,
    )
    # Full 5 s rendered, cap silently ignored.
    assert _decode_count(out) == 50
