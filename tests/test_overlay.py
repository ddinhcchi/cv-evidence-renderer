"""Tests for the bbox/label overlay."""

from __future__ import annotations

import numpy as np

from cv_evidence_renderer.overlay import color_for_label, draw_detections
from cv_evidence_renderer.types import Detection


def _blank(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_returns_same_frame_object() -> None:
    frame = _blank()
    assert draw_detections(frame, []) is frame


def test_empty_detections_leaves_frame_untouched() -> None:
    frame = _blank()
    before = frame.copy()
    draw_detections(frame, [])
    assert np.array_equal(frame, before)


def test_dtype_and_shape_preserved() -> None:
    frame = _blank(h=120, w=160)
    draw_detections(frame, [Detection(bbox=(10, 10, 50, 50), label="x")])
    assert frame.dtype == np.uint8
    assert frame.shape == (120, 160, 3)


def test_draws_something() -> None:
    frame = _blank()
    draw_detections(frame, [Detection(bbox=(50, 60, 200, 180), label="person")])
    assert frame.sum() > 0


def test_color_for_label_deterministic() -> None:
    assert color_for_label("person") == color_for_label("person")
    color = color_for_label("anything")
    assert isinstance(color, tuple) and len(color) == 3
    assert all(0 <= c <= 255 for c in color)


def test_custom_color_overrides_default() -> None:
    custom = (123, 45, 67)
    frame = _blank()
    draw_detections(frame, [Detection(bbox=(20, 20, 100, 100), label="x", color=custom)])
    matches = (
        (frame[..., 0] == custom[0]) & (frame[..., 1] == custom[1]) & (frame[..., 2] == custom[2])
    )
    assert matches.any(), "custom color should appear somewhere in the rendered frame"


def test_label_score_and_track_id_render_without_error() -> None:
    frame = _blank()
    draw_detections(
        frame,
        [Detection(bbox=(10, 50, 100, 150), label="person", score=0.87, track_id=7)],
    )
    assert frame.sum() > 0


def test_bbox_clamping_at_edges() -> None:
    frame = _blank(h=100, w=100)
    draw_detections(
        frame,
        [
            Detection(bbox=(0, 0, 40, 40), label="topleft"),
            Detection(bbox=(-50, -50, 30, 30), label="negative"),
            Detection(bbox=(80, 80, 500, 500), label="overflow"),
        ],
    )
    assert frame.shape == (100, 100, 3)
    assert frame.sum() > 0


def test_degenerate_bbox_skipped() -> None:
    frame = _blank()
    before = frame.copy()
    draw_detections(
        frame,
        [
            Detection(bbox=(50, 50, 50, 50), label="zero-area"),
            Detection(bbox=(60, 60, 40, 40), label="reversed"),
        ],
    )
    assert np.array_equal(frame, before)


def test_multiple_detections_all_drawn() -> None:
    frame = _blank()
    bboxes = [(10, 10, 60, 60), (80, 80, 140, 140), (200, 50, 280, 130)]
    draw_detections(
        frame,
        [Detection(bbox=b, label=f"obj{i}") for i, b in enumerate(bboxes)],
    )
    # Each bbox should have *some* coloured pixels inside its rectangle area.
    for x1, y1, x2, y2 in bboxes:
        region = frame[y1:y2, x1:x2]
        assert region.sum() > 0, f"nothing drawn for bbox {(x1, y1, x2, y2)}"


def test_label_at_top_of_frame_falls_back_inside_box() -> None:
    # bbox starts at y=0 → no room above for label, must render inside the top.
    frame = _blank(h=100, w=200)
    draw_detections(frame, [Detection(bbox=(10, 0, 100, 80), label="top")])
    # Top strip of the bbox should contain the filled label background.
    top_strip = frame[0:20, 10:100]
    assert top_strip.sum() > 0
