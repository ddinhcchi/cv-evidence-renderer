"""Reproducible benchmark for `render_from_jsonl`.

Builds synthetic input at several resolutions, renders a fixed event window
through the full pipeline (decode → overlay → encode), and prints a table.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --resolutions 720p,1080p --duration 10 --workers 1

Defaults are tuned to run in roughly a minute on a 2024 MacBook M-series CPU.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv_evidence_renderer.encoder.libx264 import Libx264Encoder
from cv_evidence_renderer.offline import render_from_jsonl

# Resolution catalogue. NB: dimensions stay even for yuv420p.
_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


@dataclass
class BenchResult:
    label: str
    width: int
    height: int
    fps: int
    n_frames: int
    setup_seconds: float
    render_seconds: float
    output_bytes: int

    @property
    def throughput_fps(self) -> float:
        return self.n_frames / self.render_seconds if self.render_seconds > 0 else 0.0

    @property
    def realtime_factor(self) -> float:
        clip_seconds = self.n_frames / self.fps
        return clip_seconds / self.render_seconds if self.render_seconds > 0 else 0.0


def _build_input(video: Path, dets: Path, w: int, h: int, fps: int, duration_s: int) -> None:
    n_frames = fps * duration_s
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int32)
    with Libx264Encoder(video, width=w, height=h, fps=fps) as enc:
        for t in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[..., 0] = ((xs + t * 3) % 256).astype(np.uint8)
            frame[..., 1] = ((ys + t * 5) % 256).astype(np.uint8)
            frame[..., 2] = ((xs + ys + t * 7) % 256).astype(np.uint8)
            enc.write(frame)

    # Roughly 2 detections per frame across the whole video
    bw, bh = w // 6, h // 4
    lines = []
    for fi in range(n_frames):
        ox = (fi * 4) % (w - bw - 1)
        oy = (fi * 3) % (h - bh - 1)
        lines.append(
            {"frame_idx": fi, "bbox": [ox, oy, ox + bw, oy + bh], "label": "obj_a", "track_id": 1}
        )
        lines.append(
            {
                "frame_idx": fi,
                "bbox": [w - ox - bw, h - oy - bh, w - ox, h - oy],
                "label": "obj_b",
                "track_id": 2,
            }
        )
    dets.write_text("\n".join(json.dumps(d) for d in lines) + "\n", encoding="utf-8")


def _benchmark_one(label: str, w: int, h: int, fps: int, duration_s: int) -> BenchResult:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "src.mp4"
        dets = tmp_dir / "dets.jsonl"
        out = tmp_dir / "evidence.mp4"

        t0 = time.perf_counter()
        _build_input(src, dets, w, h, fps, duration_s)
        setup = time.perf_counter() - t0

        # Render the inner 80% of the video (skips one frame off each end of the trim).
        event_start = 0.0
        event_end = float(duration_s)
        n_frames = fps * duration_s

        t1 = time.perf_counter()
        render_from_jsonl(src, dets, event_start, event_end, out)
        render = time.perf_counter() - t1

        return BenchResult(
            label=label,
            width=w,
            height=h,
            fps=fps,
            n_frames=n_frames,
            setup_seconds=setup,
            render_seconds=render,
            output_bytes=out.stat().st_size,
        )


def _print_table(results: list[BenchResult]) -> None:
    headers = ("res", "fps", "frames", "render s", "throughput fps", "x realtime", "out MB")
    rows = [
        (
            r.label,
            str(r.fps),
            str(r.n_frames),
            f"{r.render_seconds:.2f}",
            f"{r.throughput_fps:.1f}",
            f"{r.realtime_factor:.2f}",
            f"{r.output_bytes / 1024 / 1024:.2f}",
        )
        for r in results
    ]
    widths = [
        max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)
    ]

    def fmt(cols: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(fmt(headers))
    print(fmt(tuple("-" * w for w in widths)))
    for row in rows:
        print(fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        default="480p,720p,1080p",
        help="comma-separated subset of: " + ", ".join(_RESOLUTIONS),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=int, default=5, help="seconds of source video")
    args = parser.parse_args()

    labels = [s.strip() for s in args.resolutions.split(",") if s.strip()]
    unknown = [label for label in labels if label not in _RESOLUTIONS]
    if unknown:
        parser.error(f"unknown resolution(s): {unknown}; choose from {list(_RESOLUTIONS)}")

    print(f"Benchmark: fps={args.fps}, duration={args.duration}s, encoder=libx264 (CPU)")
    print()

    results: list[BenchResult] = []
    for label in labels:
        w, h = _RESOLUTIONS[label]
        print(f"running {label} ({w}x{h}) ...", flush=True)
        results.append(_benchmark_one(label, w, h, args.fps, args.duration))

    print()
    _print_table(results)


if __name__ == "__main__":
    main()
