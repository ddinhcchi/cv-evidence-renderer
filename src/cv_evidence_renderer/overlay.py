"""Bbox burn-in on RGB/BGR frames. CPU path via cv2; matches supervision drawing style."""

from __future__ import annotations

import numpy as np

from cv_evidence_renderer.types import Detection


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bboxes + labels onto a BGR frame in-place. Returns the same frame for chaining."""
    raise NotImplementedError("MVP")
