"""Tests for the Ultralytics YOLO adapter.

We don't depend on ultralytics directly — instead we verify the adapter
correctly forwards to `supervision.Detections.from_ultralytics` and then
back into our `FrameDetections` shape.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from cv_evidence_renderer.adapters import from_yolo_results


def test_routes_through_supervision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[object] = []

    def fake_from_ultralytics(results: object) -> sv.Detections:
        captured.append(results)
        return sv.Detections(
            xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float),
            class_id=np.array([0, 1]),
            confidence=np.array([0.9, 0.7]),
            tracker_id=np.array([10, 11]),
            data={"class_name": np.array(["person", "car"])},
        )

    monkeypatch.setattr(sv.Detections, "from_ultralytics", staticmethod(fake_from_ultralytics))

    sentinel = object()
    fd = from_yolo_results(sentinel, frame_idx=5)

    assert captured == [sentinel], "adapter must pass Results through unchanged"
    assert fd.frame_idx == 5
    assert len(fd.detections) == 2
    assert [d.label for d in fd.detections] == ["person", "car"]
    assert fd.detections[0].track_id == 10
    assert fd.detections[1].score == pytest.approx(0.7)


def test_timestamp_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sv.Detections,
        "from_ultralytics",
        staticmethod(lambda _: sv.Detections.empty()),
    )
    fd = from_yolo_results(object(), timestamp=42.0)
    assert fd.timestamp == 42.0
    assert fd.detections == []


def test_neither_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sv.Detections,
        "from_ultralytics",
        staticmethod(lambda _: sv.Detections.empty()),
    )
    with pytest.raises(ValueError, match="needs either frame_idx or timestamp"):
        from_yolo_results(object())


def test_missing_supervision_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If `supervision` is not installed the adapter should raise an actionable error."""
    # Force `import supervision` inside the adapter to fail.
    monkeypatch.setitem(sys.modules, "supervision", None)
    with pytest.raises(ImportError, match="requires supervision"):
        from_yolo_results(object(), frame_idx=0)
