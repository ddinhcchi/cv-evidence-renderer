"""Adapters from external detection formats → internal `FrameDetections`.

Supports:
    - `supervision.Detections` (optional, only if `pip install cv-evidence-renderer[supervision]`)
    - Raw JSONL: one detection per line, see SPEC.md §6.3
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from cv_evidence_renderer.types import FrameDetections

if TYPE_CHECKING:
    import supervision as sv


def from_supervision(
    detections: sv.Detections,
    frame_idx: int | None = None,
    timestamp: float | None = None,
) -> FrameDetections:
    raise NotImplementedError("v0.2")


def from_jsonl(path: str | Path) -> Iterator[FrameDetections]:
    raise NotImplementedError("MVP")
