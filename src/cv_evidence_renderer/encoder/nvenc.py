"""NVENC encoder via PyAV. Requires CUDA-enabled FFmpeg build."""

from __future__ import annotations

from pathlib import Path


class NvencEncoder:
    def __init__(
        self, output: str | Path, width: int, height: int, fps: float, codec: str = "h264_nvenc"
    ) -> None:
        self.output = Path(output)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec

    def open(self) -> None:
        raise NotImplementedError("v0.2")

    def write(self, frame_bgr: object) -> None:
        raise NotImplementedError("v0.2")

    def close(self) -> None:
        raise NotImplementedError("v0.2")


def is_available() -> bool:
    """Probe whether the local PyAV/FFmpeg has h264_nvenc available."""
    try:
        import av.codec
    except ImportError:
        return False
    try:
        return "h264_nvenc" in {c.name for c in av.codec.codecs_available}
    except Exception:
        return False
