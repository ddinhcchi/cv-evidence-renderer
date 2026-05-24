"""`EvidenceRecorderPool` — multi-stream parallel orchestrator with shared NVENC context.

See SPEC.md §8 v0.3.
"""

from __future__ import annotations


class EvidenceRecorderPool:
    def __init__(self, max_streams: int = 4) -> None:
        self.max_streams = max_streams

    def add(self, name: str, recorder: object) -> None:
        raise NotImplementedError("v0.3")

    def stop_all(self) -> None:
        raise NotImplementedError("v0.3")
