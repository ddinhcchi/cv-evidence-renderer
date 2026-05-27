"""Tests for the JSONL detection loader."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from cv_evidence_renderer.adapters import from_jsonl


def _write_jsonl(tmp: Path, lines: list[dict]) -> Path:
    p = tmp / "dets.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in lines) + "\n", encoding="utf-8")
    return p


def test_empty_file_yields_nothing(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert list(from_jsonl(p)) == []


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "blanks.jsonl"
    p.write_text(
        dedent(
            """
            {"frame_idx": 0, "bbox": [0,0,10,10], "label": "x"}

            {"frame_idx": 1, "bbox": [0,0,10,10], "label": "x"}

            """
        ),
        encoding="utf-8",
    )
    out = list(from_jsonl(p))
    assert len(out) == 2


def test_single_detection(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"frame_idx": 0, "bbox": [1, 2, 3, 4], "label": "person"}])
    out = list(from_jsonl(p))
    assert len(out) == 1
    fd = out[0]
    assert fd.frame_idx == 0
    assert fd.timestamp is None
    assert len(fd.detections) == 1
    det = fd.detections[0]
    assert det.bbox == (1.0, 2.0, 3.0, 4.0)
    assert det.label == "person"
    assert det.score is None
    assert det.track_id is None
    assert det.color is None
    assert det.extra == {}


def test_groups_consecutive_same_frame(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [
            {"frame_idx": 5, "bbox": [0, 0, 10, 10], "label": "a"},
            {"frame_idx": 5, "bbox": [10, 10, 20, 20], "label": "b"},
            {"frame_idx": 5, "bbox": [20, 20, 30, 30], "label": "c"},
            {"frame_idx": 6, "bbox": [0, 0, 10, 10], "label": "d"},
        ],
    )
    out = list(from_jsonl(p))
    assert len(out) == 2
    assert out[0].frame_idx == 5
    assert [d.label for d in out[0].detections] == ["a", "b", "c"]
    assert out[1].frame_idx == 6
    assert [d.label for d in out[1].detections] == ["d"]


def test_optional_fields_parsed(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [
            {
                "frame_idx": 0,
                "bbox": [0, 0, 50, 50],
                "label": "person",
                "score": 0.91,
                "track_id": 7,
                "color": [255, 0, 0],
                "extra": {"distance_m": 1.4},
            }
        ],
    )
    det = next(iter(from_jsonl(p))).detections[0]
    assert det.score == 0.91
    assert det.track_id == 7
    assert det.color == (255, 0, 0)
    assert det.extra == {"distance_m": 1.4}


def test_timestamp_keyed(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [
            {"ts": 1716530001.2, "bbox": [0, 0, 10, 10], "label": "x"},
            {"ts": 1716530001.2, "bbox": [5, 5, 15, 15], "label": "y"},
            {"ts": 1716530002.0, "bbox": [0, 0, 10, 10], "label": "z"},
        ],
    )
    out = list(from_jsonl(p))
    assert len(out) == 2
    assert out[0].timestamp == 1716530001.2
    assert out[0].frame_idx is None
    assert len(out[0].detections) == 2
    assert out[1].timestamp == 1716530002.0


def test_non_consecutive_same_frame_yields_separate(tmp_path: Path) -> None:
    # Unsorted input → adapter does not magically re-group.
    p = _write_jsonl(
        tmp_path,
        [
            {"frame_idx": 0, "bbox": [0, 0, 10, 10], "label": "a"},
            {"frame_idx": 1, "bbox": [0, 0, 10, 10], "label": "b"},
            {"frame_idx": 0, "bbox": [5, 5, 15, 15], "label": "c"},
        ],
    )
    out = list(from_jsonl(p))
    assert len(out) == 3


def test_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"frame_idx": 0, "bbox": [0,0,10,10] BROKEN', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(from_jsonl(p))


def test_missing_bbox_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"frame_idx": 0, "label": "x"}])
    with pytest.raises(ValueError, match="'bbox' must be a list of 4"):
        list(from_jsonl(p))


def test_bbox_wrong_length_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"frame_idx": 0, "label": "x", "bbox": [1, 2, 3]}])
    with pytest.raises(ValueError, match="'bbox' must be a list of 4"):
        list(from_jsonl(p))


def test_bbox_non_numeric_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"frame_idx": 0, "label": "x", "bbox": [1, 2, 3, "four"]}])
    with pytest.raises(ValueError, match="'bbox' values must be numeric"):
        list(from_jsonl(p))


def test_missing_label_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"frame_idx": 0, "bbox": [0, 0, 10, 10]}])
    with pytest.raises(ValueError, match="'label' must be a non-empty string"):
        list(from_jsonl(p))


def test_missing_frame_key_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"bbox": [0, 0, 10, 10], "label": "x"}])
    with pytest.raises(ValueError, match="needs 'frame_idx' or 'ts'"):
        list(from_jsonl(p))


def test_invalid_color_length_raises(tmp_path: Path) -> None:
    p = _write_jsonl(
        tmp_path,
        [{"frame_idx": 0, "bbox": [0, 0, 10, 10], "label": "x", "color": [255, 0]}],
    )
    with pytest.raises(ValueError, match="'color' must be a 3-element list"):
        list(from_jsonl(p))


def test_error_message_includes_line_number(tmp_path: Path) -> None:
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        dedent(
            """\
            {"frame_idx": 0, "bbox": [0,0,10,10], "label": "ok"}
            {"frame_idx": 1, "label": "missing-bbox"}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=":2 "):
        list(from_jsonl(p))


def test_bbox_accepts_floats(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"frame_idx": 0, "bbox": [1.5, 2.5, 3.5, 4.5], "label": "x"}])
    det = next(iter(from_jsonl(p))).detections[0]
    assert det.bbox == (1.5, 2.5, 3.5, 4.5)
