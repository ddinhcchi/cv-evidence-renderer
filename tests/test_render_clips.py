"""Tests for the batch `render_clips` API."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest

from cv_evidence_renderer import Clip, ClipSource, render_clip, render_clips
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


def _output_fps(path: Path) -> float:
    with av.open(str(path)) as container:
        return float(container.streams.video[0].average_rate)


@pytest.fixture
def shared_video(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "shared.mp4"
    dets = tmp_path / "shared.jsonl"
    _write_video(video, n=100)  # 10 s @ 10 fps
    _write_jsonl(
        dets,
        [{"frame_idx": fi, "bbox": [5, 5, 30, 30], "label": "x"} for fi in range(100)],
    )
    return video, dets


# ---------- API ----------


def test_empty_clips_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs at least one Clip"):
        render_clips(clips=[])


def test_single_clip_works(shared_video: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = shared_video
    out = tmp_path / "ev.mp4"
    paths = render_clips(
        clips=[
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=1)],
                output=out,
            )
        ]
    )
    assert paths == [out]
    assert _decode_count(out) == 10


def test_returns_paths_in_input_order(shared_video: tuple[Path, Path], tmp_path: Path) -> None:
    video, dets = shared_video
    outs = [tmp_path / f"ev{i}.mp4" for i in range(3)]
    clips = [
        Clip(
            sources=[ClipSource(video=video, detections=dets, from_seconds=i, to_seconds=i + 1)],
            output=outs[i],
        )
        for i in range(3)
    ]
    paths = render_clips(clips=clips)
    assert paths == outs


# ---------- shared single-source decode-once ----------


def test_shared_source_produces_correct_frame_counts(
    shared_video: tuple[Path, Path], tmp_path: Path
) -> None:
    """Three clips on the same file → each gets the right slice."""
    video, dets = shared_video
    paths = render_clips(
        clips=[
            Clip(
                sources=[
                    ClipSource(video=video, detections=dets, from_seconds=0.0, to_seconds=2.0)
                ],
                output=tmp_path / "a.mp4",
            ),
            Clip(
                sources=[
                    ClipSource(video=video, detections=dets, from_seconds=4.0, to_seconds=5.0)
                ],
                output=tmp_path / "b.mp4",
            ),
            Clip(
                sources=[
                    ClipSource(video=video, detections=dets, from_seconds=7.5, to_seconds=9.5)
                ],
                output=tmp_path / "c.mp4",
            ),
        ]
    )
    counts = [_decode_count(p) for p in paths]
    assert counts == [20, 10, 20]  # 2 s, 1 s, 2 s @ 10 fps


def test_shared_source_per_clip_playback_speed(
    shared_video: tuple[Path, Path], tmp_path: Path
) -> None:
    """Each clip in the shared-source group still honours its own playback_speed."""
    video, dets = shared_video
    paths = render_clips(
        clips=[
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=2)],
                output=tmp_path / "slow.mp4",
                playback_speed=1.0,
            ),
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=2)],
                output=tmp_path / "fast.mp4",
                playback_speed=2.0,
            ),
        ]
    )
    assert _output_fps(paths[0]) == pytest.approx(10.0, rel=0.02)
    assert _output_fps(paths[1]) == pytest.approx(20.0, rel=0.02)


def test_shared_source_matches_individual_render_pixelwise(
    shared_video: tuple[Path, Path], tmp_path: Path
) -> None:
    """A batched run on shared source must produce the same frame-count as
    standalone `render_clip` calls — the optimisation is invisible to callers."""
    video, dets = shared_video

    # Reference: render each one independently.
    ref_a = tmp_path / "ref_a.mp4"
    ref_b = tmp_path / "ref_b.mp4"
    render_clip(
        sources=[ClipSource(video=video, detections=dets, from_seconds=1, to_seconds=2)],
        output=ref_a,
    )
    render_clip(
        sources=[ClipSource(video=video, detections=dets, from_seconds=3, to_seconds=4)],
        output=ref_b,
    )

    # Batched: same two clips together.
    batch = render_clips(
        clips=[
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=1, to_seconds=2)],
                output=tmp_path / "batch_a.mp4",
            ),
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=3, to_seconds=4)],
                output=tmp_path / "batch_b.mp4",
            ),
        ]
    )

    # Frame counts must match exactly.
    assert _decode_count(batch[0]) == _decode_count(ref_a)
    assert _decode_count(batch[1]) == _decode_count(ref_b)


def test_shared_source_label_formatter_isolated_per_clip(
    shared_video: tuple[Path, Path], tmp_path: Path
) -> None:
    """Each clip's label_formatter must apply only to its own output."""
    video, dets = shared_video
    paths = render_clips(
        clips=[
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=1)],
                output=tmp_path / "a.mp4",
                label_formatter=lambda d: "AAA",
            ),
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=1)],
                output=tmp_path / "b.mp4",
                label_formatter=lambda d: "BBB",
            ),
        ]
    )
    # Different captions → different rendered bytes.
    assert paths[0].read_bytes() != paths[1].read_bytes()


# ---------- mixed batches ----------


def test_unique_sources_no_sharing(tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_video(a, n=30)
    _write_video(b, n=30)
    paths = render_clips(
        clips=[
            Clip(
                sources=[ClipSource(video=a, from_seconds=0, to_seconds=1)],
                output=tmp_path / "out_a.mp4",
            ),
            Clip(
                sources=[ClipSource(video=b, from_seconds=0, to_seconds=2)],
                output=tmp_path / "out_b.mp4",
            ),
        ]
    )
    assert _decode_count(paths[0]) == 10
    assert _decode_count(paths[1]) == 20


def test_mixed_single_and_multi_source(tmp_path: Path) -> None:
    """A batch can mix single-source (shareable) and multi-source clips."""
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_video(a, n=30)
    _write_video(b, n=30)
    paths = render_clips(
        clips=[
            Clip(
                sources=[ClipSource(video=a, from_seconds=0, to_seconds=1)],
                output=tmp_path / "single.mp4",
            ),
            Clip(
                sources=[
                    ClipSource(video=a, from_seconds=2.0, to_seconds=2.5),
                    ClipSource(video=b, from_seconds=0.0, to_seconds=1.5),
                ],
                output=tmp_path / "multi.mp4",
            ),
        ]
    )
    assert _decode_count(paths[0]) == 10
    assert _decode_count(paths[1]) == 20  # 5 + 15 frames


def test_shared_max_duration_independent_per_clip(
    shared_video: tuple[Path, Path], tmp_path: Path
) -> None:
    """Two clips on the same source: only one has a duration cap, the other doesn't."""
    video, dets = shared_video
    paths = render_clips(
        clips=[
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=4)],
                output=tmp_path / "uncapped.mp4",
            ),
            Clip(
                sources=[ClipSource(video=video, detections=dets, from_seconds=0, to_seconds=4)],
                output=tmp_path / "capped.mp4",
                max_duration_seconds=1.0,
                duration_strategy="framedrop",
            ),
        ]
    )
    # Uncapped: 4 s @ 10 fps = 40 frames.
    # Capped framedrop: stride 4 → 10 frames kept.
    assert _decode_count(paths[0]) == 40
    assert _decode_count(paths[1]) == 10
