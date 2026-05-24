"""`EvidenceRecorder` — live RTSP + ring buffer + event trigger (USE CASE A).

See SPEC.md §6.1 and §8 v0.2. Not implemented in MVP — placeholder API.
"""

from __future__ import annotations

from pathlib import Path

from cv_evidence_renderer.types import Encoder


class EvidenceRecorder:
    def __init__(
        self,
        source: str,
        pre_buffer_seconds: float = 5.0,
        post_buffer_seconds: float = 10.0,
        encoder: Encoder | str = Encoder.AUTO,
        output_dir: str | Path = "./evidence",
    ) -> None:
        self.source = source
        self.pre_buffer_seconds = pre_buffer_seconds
        self.post_buffer_seconds = post_buffer_seconds
        self.encoder = Encoder(encoder) if isinstance(encoder, str) else encoder
        self.output_dir = Path(output_dir)

    def start(self) -> None:
        raise NotImplementedError("v0.2 — see SPEC.md §8")

    def push(self, frame_idx: int, detections: object) -> None:
        raise NotImplementedError("v0.2")

    def trigger_event(self, event_id: str, label: str | None = None) -> Path:
        raise NotImplementedError("v0.2")

    def stop(self) -> None:
        raise NotImplementedError("v0.2")
