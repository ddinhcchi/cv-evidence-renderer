"""Tests for the libx264 CPU encoder."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from cv_evidence_renderer.encoder.libx264 import Libx264Encoder


def _gradient_frame(width: int, height: int, t: int) -> np.ndarray:
    """A deterministic BGR frame that varies with `t` so encoded video has motion."""
    x = np.linspace(0, 255, width, dtype=np.int32)
    y = np.linspace(0, 255, height, dtype=np.int32)
    bg = (x[None, :] + y[:, None]) % 256
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = bg.astype(np.uint8)
    frame[..., 1] = ((bg + t * 8) % 256).astype(np.uint8)
    frame[..., 2] = ((bg + t * 16) % 256).astype(np.uint8)
    return frame


def test_encodes_short_clip(tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    width, height, fps, n_frames = 320, 240, 30, 30  # 1 second

    with Libx264Encoder(out, width=width, height=height, fps=fps) as enc:
        for t in range(n_frames):
            enc.write(_gradient_frame(width, height, t))

    assert out.exists()
    assert out.stat().st_size > 1_000  # > 1 KB — non-trivial

    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.width == width
        assert stream.height == height
        decoded = sum(1 for _ in container.decode(stream))
        assert decoded == n_frames


def test_explicit_open_close(tmp_path: Path) -> None:
    out = tmp_path / "explicit.mp4"
    enc = Libx264Encoder(out, width=64, height=64, fps=10)
    enc.open()
    for t in range(5):
        enc.write(_gradient_frame(64, 64, t))
    enc.close()
    assert out.exists() and out.stat().st_size > 0

    # close() is idempotent
    enc.close()


def test_rejects_odd_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="even dimensions"):
        Libx264Encoder(tmp_path / "x.mp4", width=321, height=240, fps=30)


def test_rejects_non_positive_fps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        Libx264Encoder(tmp_path / "x.mp4", width=64, height=64, fps=0)


def test_accepts_float_fps(tmp_path: Path) -> None:
    """Regression: PyAV needs Fraction for rate; bare floats raised AttributeError."""
    out = tmp_path / "float_fps.mp4"
    with Libx264Encoder(out, width=64, height=64, fps=29.97) as enc:
        for t in range(5):
            enc.write(_gradient_frame(64, 64, t))
    assert out.exists() and out.stat().st_size > 0


def test_write_before_open_raises(tmp_path: Path) -> None:
    enc = Libx264Encoder(tmp_path / "x.mp4", width=64, height=64, fps=10)
    with pytest.raises(RuntimeError, match="not open"):
        enc.write(_gradient_frame(64, 64, 0))


def test_write_wrong_shape_raises(tmp_path: Path) -> None:
    with (
        Libx264Encoder(tmp_path / "x.mp4", width=64, height=64, fps=10) as enc,
        pytest.raises(ValueError, match="does not match encoder"),
    ):
        enc.write(_gradient_frame(32, 32, 0))


def test_double_open_raises(tmp_path: Path) -> None:
    enc = Libx264Encoder(tmp_path / "x.mp4", width=64, height=64, fps=10)
    enc.open()
    try:
        with pytest.raises(RuntimeError, match="already open"):
            enc.open()
    finally:
        enc.close()
