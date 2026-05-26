"""Bbox burn-in on BGR frames. CPU path via cv2; visual style matches supervision."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from cv_evidence_renderer.types import Detection

# 10-colour BGR palette picked to stay readable on dark and light backgrounds.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 56, 56),  # blue-red
    (0, 165, 255),  # orange
    (0, 200, 0),  # green
    (0, 0, 255),  # red
    (255, 200, 0),  # cyan-yellow
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (148, 0, 211),  # purple
    (0, 128, 128),  # teal
    (180, 105, 255),  # pink
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2
_TEXT_COLOR = (255, 255, 255)


def color_for_label(label: str) -> tuple[int, int, int]:
    """Deterministic BGR colour for a label string."""
    digest = hashlib.md5(label.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _PALETTE[int(digest, 16) % len(_PALETTE)]


def _format_label(det: Detection) -> str:
    parts = [det.label]
    if det.track_id is not None:
        parts.append(f"#{det.track_id}")
    if det.score is not None:
        parts.append(f"{det.score:.2f}")
    return " ".join(parts)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bboxes + labels onto a BGR frame in-place. Returns the same frame for chaining.

    Bboxes outside the frame are clamped; degenerate bboxes (zero area or reversed)
    are skipped silently.
    """
    h, w = frame.shape[:2]

    for det in detections:
        color = det.color if det.color is not None else color_for_label(det.label)

        x1f, y1f, x2f, y2f = det.bbox
        x1 = max(0, min(round(x1f), w - 1))
        y1 = max(0, min(round(y1f), h - 1))
        x2 = max(0, min(round(x2f), w - 1))
        y2 = max(0, min(round(y2f), h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=_BOX_THICKNESS)

        text = _format_label(det)
        (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        bg_h = th + baseline + 2

        # Prefer label above the bbox; fall back to inside-top when there's no room.
        if y1 - bg_h >= 0:
            bg_y1 = y1 - bg_h
            text_y = y1 - 2
        else:
            bg_y1 = y1
            text_y = y1 + th + 1

        bg_x2 = min(x1 + tw + 4, w - 1)
        bg_y2 = bg_y1 + bg_h

        cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, thickness=-1)
        cv2.putText(
            frame,
            text,
            (x1 + 2, text_y),
            _FONT,
            _FONT_SCALE,
            _TEXT_COLOR,
            thickness=_FONT_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    return frame
