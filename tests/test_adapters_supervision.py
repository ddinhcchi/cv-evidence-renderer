"""Tests for the supervision.Detections adapter."""

from __future__ import annotations

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from cv_evidence_renderer.adapters import from_supervision


def test_full_fields_parsed() -> None:
    det = sv.Detections(
        xyxy=np.array([[10, 20, 100, 80], [150, 30, 250, 120]], dtype=float),
        class_id=np.array([0, 2]),
        confidence=np.array([0.85, 0.72]),
        tracker_id=np.array([1, 7]),
        data={"class_name": np.array(["person", "car"])},
    )

    fd = from_supervision(det, frame_idx=42)

    assert fd.frame_idx == 42
    assert fd.timestamp is None
    assert len(fd.detections) == 2

    a, b = fd.detections
    assert a.bbox == (10.0, 20.0, 100.0, 80.0)
    assert a.label == "person"
    assert a.score == pytest.approx(0.85)
    assert a.track_id == 1
    assert b.label == "car"
    assert b.track_id == 7


def test_falls_back_to_class_id_when_no_class_name() -> None:
    det = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([5]),
        confidence=np.array([0.5]),
    )
    fd = from_supervision(det, frame_idx=0)
    assert fd.detections[0].label == "class_5"


def test_falls_back_to_obj_when_no_class_info() -> None:
    det = sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float))
    fd = from_supervision(det, frame_idx=0)
    assert fd.detections[0].label == "obj"


def test_empty_detections() -> None:
    fd = from_supervision(sv.Detections.empty(), frame_idx=3)
    assert fd.frame_idx == 3
    assert fd.detections == []


def test_missing_confidence_and_tracker_id() -> None:
    det = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([0]),
        data={"class_name": np.array(["thing"])},
    )
    fd = from_supervision(det, frame_idx=0)
    d = fd.detections[0]
    assert d.label == "thing"
    assert d.score is None
    assert d.track_id is None


def test_timestamp_keyed() -> None:
    det = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([0]),
    )
    fd = from_supervision(det, timestamp=1716530001.2)
    assert fd.timestamp == 1716530001.2
    assert fd.frame_idx is None


def test_both_keys_accepted() -> None:
    det = sv.Detections.empty()
    fd = from_supervision(det, frame_idx=5, timestamp=1.5)
    assert fd.frame_idx == 5
    assert fd.timestamp == 1.5


def test_neither_key_raises() -> None:
    with pytest.raises(ValueError, match="needs either frame_idx or timestamp"):
        from_supervision(sv.Detections.empty())


def test_returned_values_are_python_scalars() -> None:
    """Numpy scalars must be coerced to plain Python types so the result is
    JSON-serialisable downstream."""
    det = sv.Detections(
        xyxy=np.array([[1, 2, 3, 4]], dtype=np.float64),
        class_id=np.array([7], dtype=np.int64),
        confidence=np.array([0.9], dtype=np.float32),
        tracker_id=np.array([3], dtype=np.int32),
    )
    d = from_supervision(det, frame_idx=0).detections[0]
    assert all(isinstance(v, float) for v in d.bbox)
    assert isinstance(d.score, float)
    assert isinstance(d.track_id, int)
