"""Golden-frame regression test.

Renders a deterministic short clip with `render_from_jsonl` and compares one
decoded frame against a stored PNG. If the encoder/overlay change in a way that
materially affects pixel output (codec swap, drawing bug), this catches it.

Regenerate the golden when an intentional pixel change lands:

    REGEN_GOLDEN=1 pytest tests/test_golden_frame.py

PSNR is benchmarked on the same machine that generated the golden PNG so
local reruns sit around 45-50 dB. Cross-platform libx264 builds (the
Homebrew bottle on macOS vs the apt build on Ubuntu) diverge by enough
that the same logical render produces ~36 dB between platforms. The
threshold below is set to catch real regressions (a broken overlay sits
below 25 dB, a swapped codec below 15 dB) while tolerating that noise
floor.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from cv_evidence_renderer.encoder.libx264 import Libx264Encoder
from cv_evidence_renderer.offline import render_from_jsonl

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_PNG = FIXTURES / "golden_frame_05.png"
PSNR_THRESHOLD_DB = 30.0

WIDTH, HEIGHT, FPS, N_FRAMES = 128, 96, 10, 50
EVENT_START, EVENT_END = 1.0, 2.0
TARGET_FRAME_IN_OUTPUT = 5  # frame index inside the rendered clip


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float((diff * diff).mean())
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _build_input_video(path: Path) -> None:
    """Deterministic input: a soft colour gradient that shifts each frame."""
    with Libx264Encoder(path, width=WIDTH, height=HEIGHT, fps=FPS) as enc:
        ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.int32)
        for t in range(N_FRAMES):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame[..., 0] = ((xs + t * 3) % 256).astype(np.uint8)
            frame[..., 1] = ((ys + t * 5) % 256).astype(np.uint8)
            frame[..., 2] = ((xs + ys + t * 7) % 256).astype(np.uint8)
            enc.write(frame)


def _write_input_jsonl(path: Path) -> None:
    lines = []
    for fi in range(10, 20):
        lines.append(
            {
                "frame_idx": fi,
                "bbox": [20 + (fi - 10) * 2, 30, 90 + (fi - 10) * 2, 80],
                "label": "walker",
                "score": 0.80 + (fi - 10) * 0.01,
                "track_id": 1,
            }
        )
    path.write_text("\n".join(json.dumps(d) for d in lines) + "\n", encoding="utf-8")


def _decode_frame(video: Path, target_index: int) -> np.ndarray:
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == target_index:
                return frame.to_ndarray(format="bgr24")
    raise AssertionError(f"video {video} has fewer than {target_index + 1} frames")


def test_golden_frame(tmp_path: Path) -> None:
    src_video = tmp_path / "src.mp4"
    dets = tmp_path / "dets.jsonl"
    out = tmp_path / "evidence.mp4"

    _build_input_video(src_video)
    _write_input_jsonl(dets)

    render_from_jsonl(
        video=src_video,
        detections_jsonl=dets,
        event_start=EVENT_START,
        event_end=EVENT_END,
        output=out,
    )

    rendered = _decode_frame(out, TARGET_FRAME_IN_OUTPUT)

    if os.environ.get("REGEN_GOLDEN") == "1":
        FIXTURES.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(GOLDEN_PNG), rendered)
        pytest.skip(f"regenerated {GOLDEN_PNG.name}; rerun without REGEN_GOLDEN to compare")

    if not GOLDEN_PNG.exists():
        pytest.skip(
            f"missing golden fixture {GOLDEN_PNG.name}; "
            f"run `REGEN_GOLDEN=1 pytest tests/test_golden_frame.py` to create it"
        )

    golden = cv2.imread(str(GOLDEN_PNG))
    assert golden.shape == rendered.shape, (
        f"shape drift: golden={golden.shape} rendered={rendered.shape}"
    )

    psnr = _psnr(golden, rendered)
    assert psnr >= PSNR_THRESHOLD_DB, (
        f"frame {TARGET_FRAME_IN_OUTPUT} drifted: PSNR {psnr:.2f} dB < {PSNR_THRESHOLD_DB} dB"
    )
