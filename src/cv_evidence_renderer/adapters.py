"""Adapters from external detection formats → internal `FrameDetections`.

Supports:
    - `supervision.Detections` (optional, only if `pip install cv-evidence-renderer[supervision]`)
    - Raw JSONL: one detection per line, see SPEC.md §6.3
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cv_evidence_renderer.types import Detection, FrameDetections

if TYPE_CHECKING:
    import supervision as sv


def from_supervision(
    detections: sv.Detections,
    frame_idx: int | None = None,
    timestamp: float | None = None,
) -> FrameDetections:
    """Convert a `supervision.Detections` into a `FrameDetections`.

    Label resolution prefers `detections.data["class_name"]`; otherwise it
    falls back to `f"class_{class_id}"`; if neither is available the label
    is `"obj"`. Confidence and tracker_id are copied through when present.

    `supervision` is an optional dependency — install with
    `pip install cv-evidence-renderer[supervision]`. This function uses only
    duck-typed attribute access, so no runtime import is required here.
    """
    if frame_idx is None and timestamp is None:
        raise ValueError("from_supervision needs either frame_idx or timestamp")

    n = len(detections)
    class_names = None
    if hasattr(detections, "data") and detections.data:
        class_names = detections.data.get("class_name")

    parsed: list[Detection] = []
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in detections.xyxy[i])

        if class_names is not None:
            label = str(class_names[i])
        elif detections.class_id is not None:
            label = f"class_{int(detections.class_id[i])}"
        else:
            label = "obj"

        score = float(detections.confidence[i]) if detections.confidence is not None else None
        track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None

        parsed.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                label=label,
                score=score,
                track_id=track_id,
            )
        )

    return FrameDetections(detections=parsed, frame_idx=frame_idx, timestamp=timestamp)


def from_jsonl(path: str | Path) -> Iterator[FrameDetections]:
    """Stream detections from a JSONL file, grouped into one `FrameDetections` per frame.

    Each line must be a JSON object with:
        - `bbox`: list of 4 numbers (x1, y1, x2, y2)
        - `label`: str
        - exactly one of `frame_idx` (int) or `ts` (float) — both is also accepted
        - optional `score` (float), `track_id` (int), `color` (3-tuple BGR), `extra` (dict)

    Lines must be sorted so detections of the same frame are consecutive; otherwise
    they will be yielded as separate `FrameDetections`. Blank lines are skipped.

    Raises:
        ValueError: malformed JSON, missing required field, or wrong types.
    """
    path = Path(path)

    current_key: tuple[int | None, float | None] | None = None
    current_dets: list[Detection] = []

    with path.open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            det, key = _parse_line(line, path, line_no)
            if current_key is None or key == current_key:
                current_key = key
                current_dets.append(det)
            else:
                yield _make(current_key, current_dets)
                current_key = key
                current_dets = [det]

    if current_key is not None:
        yield _make(current_key, current_dets)


def _make(
    key: tuple[int | None, float | None],
    dets: list[Detection],
) -> FrameDetections:
    frame_idx, timestamp = key
    return FrameDetections(detections=dets, frame_idx=frame_idx, timestamp=timestamp)


def _parse_line(
    line: str,
    path: Path,
    line_no: int,
) -> tuple[Detection, tuple[int | None, float | None]]:
    try:
        obj: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no} invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path}:{line_no} expected a JSON object, got {type(obj).__name__}")

    bbox = obj.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{path}:{line_no} 'bbox' must be a list of 4 numbers")
    if not all(isinstance(v, int | float) for v in bbox):
        raise ValueError(f"{path}:{line_no} 'bbox' values must be numeric")

    label = obj.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError(f"{path}:{line_no} 'label' must be a non-empty string")

    frame_idx = obj.get("frame_idx")
    timestamp = obj.get("ts")
    if frame_idx is None and timestamp is None:
        raise ValueError(f"{path}:{line_no} needs 'frame_idx' or 'ts'")
    if frame_idx is not None and not isinstance(frame_idx, int):
        raise ValueError(f"{path}:{line_no} 'frame_idx' must be int")
    if timestamp is not None and not isinstance(timestamp, int | float):
        raise ValueError(f"{path}:{line_no} 'ts' must be numeric")

    color_raw = obj.get("color")
    color: tuple[int, int, int] | None = None
    if color_raw is not None:
        if not isinstance(color_raw, list) or len(color_raw) != 3:
            raise ValueError(f"{path}:{line_no} 'color' must be a 3-element list")
        color = (int(color_raw[0]), int(color_raw[1]), int(color_raw[2]))

    det = Detection(
        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        label=label,
        score=obj.get("score"),
        track_id=obj.get("track_id"),
        color=color,
        extra=obj.get("extra", {}),
    )
    return det, (frame_idx, float(timestamp) if timestamp is not None else None)
