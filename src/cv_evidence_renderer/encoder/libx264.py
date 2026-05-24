"""libx264 CPU encoder via PyAV. Always available; used as MVP default and fallback."""

from __future__ import annotations

from pathlib import Path


class Libx264Encoder:
    def __init__(self, output: str | Path, width: int, height: int, fps: float) -> None:
        self.output = Path(output)
        self.width = width
        self.height = height
        self.fps = fps

    def open(self) -> None:
        raise NotImplementedError("MVP")

    def write(self, frame_bgr: object) -> None:
        raise NotImplementedError("MVP")

    def close(self) -> None:
        raise NotImplementedError("MVP")
