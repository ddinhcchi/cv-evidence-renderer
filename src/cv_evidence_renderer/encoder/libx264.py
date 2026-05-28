"""libx264 CPU encoder via PyAV. Always available; used as MVP default and fallback."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import TracebackType

import av
import av.container
import av.video
import numpy as np


class Libx264Encoder:
    """H.264 encoder writing an MP4 file via PyAV's libx264 binding.

    Accepts BGR uint8 frames (the cv2 convention). Width and height must be even
    because yuv420p requires it; the encoder raises early if they aren't.
    """

    def __init__(
        self,
        output: str | Path,
        width: int,
        height: int,
        fps: float,
        crf: int = 23,
        preset: str = "medium",
    ) -> None:
        if width % 2 or height % 2:
            raise ValueError(f"libx264 yuv420p requires even dimensions, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        self.output = Path(output)
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.preset = preset

        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.VideoStream | None = None
        self._frame_idx = 0

    def open(self) -> None:
        if self._container is not None:
            raise RuntimeError("encoder already open")

        self.output.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(self.output), mode="w")
        # PyAV's add_stream wants a Fraction, not a float — bare floats raise
        # `AttributeError: 'float' object has no attribute 'numerator'`.
        rate = Fraction(self.fps).limit_denominator(10000)
        stream = container.add_stream("libx264", rate=rate)
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(self.crf), "preset": self.preset}

        self._container = container
        self._stream = stream

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._container is None or self._stream is None:
            raise RuntimeError("encoder not open — call open() first")
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame shape {frame_bgr.shape[:2]} does not match encoder "
                f"({self.height}, {self.width})"
            )

        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        frame.pts = self._frame_idx
        self._frame_idx += 1
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._container is None or self._stream is None:
            return
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None

    def __enter__(self) -> Libx264Encoder:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
