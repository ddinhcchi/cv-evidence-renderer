"""Generate `demo/demo.gif` from a synthetic scene rendered through the library.

This is the README demo. It deliberately uses synthetic footage (two coloured
blobs moving across a slate background) rather than real surveillance video,
so the demo can be regenerated reproducibly without depending on any external
dataset or licence.

Run from project root:

    python scripts/make_demo_gif.py

Output: `demo/demo.gif`, ~5 seconds, 320x240 at 15 fps.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from cv_evidence_renderer.encoder.libx264 import Libx264Encoder
from cv_evidence_renderer.offline import render_from_jsonl

WIDTH = 320
HEIGHT = 240
FPS = 15
N_FRAMES = 75  # 5 seconds


def _scene_frame(t: int) -> np.ndarray:
    """A slate background with two moving rectangles (person-like proportions)."""
    frame = np.full((HEIGHT, WIDTH, 3), (54, 50, 46), dtype=np.uint8)  # dark slate

    # Subtle floor band — gives the scene depth
    cv2.rectangle(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT), (74, 70, 66), thickness=-1)

    # Person A: walks left to right at mid height
    progress_a = t / (N_FRAMES - 1)
    ax = int(20 + progress_a * (WIDTH - 80))
    ay = 110
    cv2.rectangle(frame, (ax, ay), (ax + 28, ay + 70), (210, 210, 220), thickness=-1)
    cv2.circle(frame, (ax + 14, ay - 8), 11, (200, 200, 215), thickness=-1)

    # Person B: walks right to left, slightly delayed
    progress_b = ((t + 25) % N_FRAMES) / (N_FRAMES - 1)
    bx = int(WIDTH - 50 - progress_b * (WIDTH - 80))
    by = 95
    cv2.rectangle(frame, (bx, by), (bx + 26, by + 65), (180, 200, 240), thickness=-1)
    cv2.circle(frame, (bx + 13, by - 8), 10, (170, 190, 230), thickness=-1)

    return frame


def _detections_for_frame(t: int) -> list[dict[str, object]]:
    """Bboxes that follow the two people in `_scene_frame`."""
    progress_a = t / (N_FRAMES - 1)
    ax = int(20 + progress_a * (WIDTH - 80))
    progress_b = ((t + 25) % N_FRAMES) / (N_FRAMES - 1)
    bx = int(WIDTH - 50 - progress_b * (WIDTH - 80))
    return [
        {
            "frame_idx": t,
            "bbox": [ax - 2, 100, ax + 30, 182],
            "label": "person",
            "score": 0.92,
            "track_id": 1,
        },
        {
            "frame_idx": t,
            "bbox": [bx - 2, 85, bx + 28, 162],
            "label": "person",
            "score": 0.87,
            "track_id": 2,
        },
    ]


def _build_input(video: Path, detections_jsonl: Path) -> None:
    with Libx264Encoder(video, width=WIDTH, height=HEIGHT, fps=FPS) as enc:
        for t in range(N_FRAMES):
            enc.write(_scene_frame(t))

    lines = []
    for t in range(N_FRAMES):
        lines.extend(_detections_for_frame(t))
    detections_jsonl.write_text("\n".join(json.dumps(d) for d in lines) + "\n", encoding="utf-8")


def _mp4_to_gif(mp4_path: Path, gif_path: Path) -> None:
    frames: list[Image.Image] = []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE))

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_gif = project_root / "demo" / "demo.gif"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "scene.mp4"
        dets = tmp_dir / "dets.jsonl"
        rendered = tmp_dir / "rendered.mp4"

        _build_input(src, dets)
        render_from_jsonl(
            video=src,
            detections_jsonl=dets,
            event_start=0.0,
            event_end=N_FRAMES / FPS,
            output=rendered,
        )
        _mp4_to_gif(rendered, out_gif)

    size_kb = out_gif.stat().st_size / 1024
    print(
        f"Wrote {out_gif.relative_to(project_root)} ({size_kb:.1f} KB, {N_FRAMES} frames @ {FPS} fps)"
    )


if __name__ == "__main__":
    main()
