# 🎬 cv-evidence-renderer

The missing **evidence-clip layer** between your detector and storage. Point it at a video + a detection JSONL → get a trimmed MP4 with bounding boxes already burned in, encoded in pure Python — no DeepStream required.

![demo](demo/demo.gif)

> The GIF above is rendered through `cv-evidence-renderer` itself — synthetic scene, synthetic detections, real output. Regenerate any time with `python scripts/make_demo_gif.py`.

> **Status: v0.1 — offline mode works end-to-end** with cross-file concat, batch rendering, playback speed, duration cap, and per-clip caption customisation. Live RTSP recording, ring buffer, and NVENC encoding are designed but not yet implemented (see Roadmap).

---

## Why this project

Every CV team shipping to production hits the same wall: the detector fires, and now you need a **short MP4 clip of the event with bounding boxes drawn on it**, to attach to an alert, archive for compliance, or replay during QA. The existing options are all incomplete:

- [`supervision`](https://github.com/roboflow/supervision) (39k ⭐) owns Python drawing — but its `VideoSink` is `cv2.VideoWriter` with `mp4v` hard-coded. No NVENC, no event window, no pre/post buffer.
- DeepStream's **Smart Record** is the official NVIDIA answer — but it has [no Python bindings](https://forums.developer.nvidia.com/t/how-to-use-smart-record-in-deepstream-6-1-python/231682) (NVIDIA staff confirmed), and bbox burn-in [has been broken since 6.4](https://forums.developer.nvidia.com/t/deepstream-6-4-smart-record-video-issue-with-bbox-enabled/290732).
- The canonical [PyImageSearch KeyClipWriter](https://pyimagesearch.com/2016/02/29/saving-key-event-video-clips-with-opencv/) ring-buffer pattern is detection-agnostic, OpenCV-only, and isn't a library.

So every team hand-rolls OpenCV + an FFmpeg subprocess, ships the bug to prod, and writes it again on the next project. This repo is the library version of that pattern — done once, done right, with GPU encoding included.

---

## What it does (and doesn't)

✅ **Working today (v0.1):**
- **Offline mode**: re-render evidence from a saved video + detections JSONL with event-window trim
- **Multi-source events**: one event window spanning several source files (NVR / CCTV segmented recordings cut at hour boundaries) — see `render_clip` with multiple `ClipSource` segments
- **Batch rendering**: `render_clips` writes N evidence files in one call; clips that share a source video are decoded *once*, dispatched to each clip's encoder
- **Bounding-box / label burn-in** before encode (so the evidence file *is* the annotated version)
- **Per-clip caption customisation** via `label_formatter` callable; default formatter exposed for composition
- **Playback speed control** (`playback_speed`) and **output duration cap** (`max_duration_seconds`) with `timelapse` or `framedrop` strategy
- **libx264** CPU encoding via PyAV — works on Mac, Linux, Windows with no GPU
- **First-class interop** with [`supervision.Detections`](https://supervision.roboflow.com/) and Ultralytics YOLO `Results` — also accepts raw JSONL
- Python library + Typer CLI (`cv-evidence render ...` with `--playback-speed`, `--max-duration-seconds`, `--duration-strategy`)

🚧 **Designed, not yet implemented (see Roadmap):**
- **NVENC** GPU encoding (H.264 / H.265) — v0.2
- **Live mode**: threaded RTSP reader → ring buffer → `trigger_event()` flushes evidence — v0.2
- **Multi-stream** parallel via shared encoder pool — v0.3
- **Plugin overlays** for custom lines, points, anchors, distance vectors, zone polygons — v0.3

🚫 **Does not (by design):**
- Detection / tracking — bring your own (YOLO, Detectron2, anything that produces bboxes)
- Live video streaming output — output is an MP4 file, not an RTSP relay
- Alerting / Telegram / email — pair with [realtime-object-detection-alert](https://github.com/ddinhcchi/realtime-object-detection-alert) for that
- Web UI — CLI + Python library only

---

## Quick start

```bash
pip install cv-evidence-renderer

# Optional: install with supervision interop
pip install cv-evidence-renderer[supervision]
```

### Use case A: offline batch from JSONL (working today)

```python
from cv_evidence_renderer import render_from_jsonl

render_from_jsonl(
    video="incidents/raw_001.mp4",
    detections_jsonl="incidents/raw_001.detections.jsonl",
    event_start=12.5,            # seconds
    event_end=22.0,
    output="evidence/event_001.mp4",
    encoder="libx264",           # NVENC ships in v0.2
)
```

### Use case B: CLI (working today)

```bash
cv-evidence render \
  --input street.mp4 \
  --detections detections.jsonl \
  --event-start 12.5 --event-end 22.0 \
  --output evidence.mp4 \
  --encoder libx264
```

### Use case C: route detections from Ultralytics YOLO or supervision

```python
import supervision as sv
from ultralytics import YOLO
from cv_evidence_renderer.adapters import from_yolo_results, from_supervision

model = YOLO("yolov8n.pt")
results = model("incidents/raw_001.mp4")

# Either: stream Ultralytics Results directly
frame_detections = [
    from_yolo_results(r, frame_idx=i) for i, r in enumerate(results)
]

# Or: pre-converted supervision.Detections
det = sv.Detections.from_ultralytics(results[0])
frame_detections = [from_supervision(det, frame_idx=0)]
```

Both adapters require the optional `[supervision]` extra (`pip install cv-evidence-renderer[supervision]`).

### Use case D: one event spanning two NVR files (cross-file concat)

```python
from cv_evidence_renderer import ClipSource, render_clip

# A violation at 22:59:30 lives across two hour-segmented recordings.
render_clip(
    sources=[
        ClipSource(video="cam01_22-00.mp4", detections="cam01_22-00.jsonl",
                   from_seconds=1770, to_seconds=1800),  # last 30 s of file A
        ClipSource(video="cam01_23-00.mp4", detections="cam01_23-00.jsonl",
                   from_seconds=0, to_seconds=90),       # first 90 s of file B
    ],
    output="evidence/violation_cross_file.mp4",
    label_formatter=lambda d: f"{d.label.upper()} #{d.track_id}",
)
```

All sources must share width, height, and (within 1%) fps; otherwise the call
raises a clear `ValueError`. The output is one continuous MP4 with detections
overlaid on each segment from its own JSONL.

### Use case E: batch render with playback speed and duration cap

```python
from cv_evidence_renderer import Clip, ClipSource, render_clips

# Ten events from the same 4-hour recording. The source is decoded once.
events = [(60, 75), (340, 360), (812, 830), ...]  # (start, end) per event

render_clips(
    clips=[
        Clip(
            sources=[ClipSource(video="day_03.mp4", detections="day_03.jsonl",
                                from_seconds=start, to_seconds=end)],
            output=f"evidence/event_{i:03d}.mp4",
            playback_speed=1.0,
            max_duration_seconds=15,        # cap each clip at 15 s
            duration_strategy="timelapse",  # auto fast-forward if longer
        )
        for i, (start, end) in enumerate(events)
    ],
)
```

### Use case F: live RTSP recorder (v0.2 — not yet implemented)

```python
# This is the planned API. EvidenceRecorder currently raises NotImplementedError.
from cv_evidence_renderer import EvidenceRecorder

recorder = EvidenceRecorder(
    source="rtsp://camera.local/stream",
    pre_buffer_seconds=5,
    post_buffer_seconds=10,
    encoder="nvenc_h264",  # NVENC also v0.2
)
recorder.start()
# ... see SPEC.md for the full design.
```

---

## Benchmark — Apple M4 (CPU libx264 baseline)

5-second source video, 30 fps, two detections burned in every frame, full decode → overlay → encode pipeline through `render_from_jsonl`. Reproduce with `python scripts/benchmark.py`.

| Resolution | Render time | Throughput | × realtime | Output |
|---|---:|---:|---:|---:|
| 480p (854×480) | 0.53 s | **282 fps** | 9.4× | 0.42 MB |
| 720p (1280×720) | 0.89 s | 168 fps | 5.6× | 0.70 MB |
| 1080p (1920×1080) | 1.70 s | 88 fps | 2.95× | 1.34 MB |

CPU-only libx264 on M4 already runs faster than realtime up to 1080p; NVENC on a discrete GPU (v0.2) will be added to this table side by side once the encoder lands.

---

## Architecture

```
                                ┌──────────────────────────────────────┐
                                │  Your detector loop                  │
                                │  (YOLO / Detectron2 / anything)      │
                                └──────────────┬───────────────────────┘
                                               │ frame_idx, sv.Detections
                                               ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐
│ Video source │ → │ Threaded reader  │ → │ Ring buffer      │ → │ NVENC encoder│ → evidence.mp4
│ RTSP / MP4   │    │ (auto-reconnect) │    │ (pre-buffer N s) │    │ (PyAV)       │
└──────────────┘    └──────────────────┘    └────────▲─────────┘    └──────────────┘
                                                     │
                                          on trigger_event():
                                          flush pre-buffer +
                                          record N more seconds
                                          with bbox burn-in
```

### Code layout

| File | Responsibility |
|---|---|
| `src/cv_evidence_renderer/recorder.py` | `EvidenceRecorder` — live RTSP + ring buffer + trigger API |
| `src/cv_evidence_renderer/offline.py` | `render_from_jsonl()` — re-render from saved video |
| `src/cv_evidence_renderer/buffer.py` | Ring buffer with keyframe-aware seek |
| `src/cv_evidence_renderer/encoder/nvenc.py` | PyAV NVENC wrapper |
| `src/cv_evidence_renderer/encoder/libx264.py` | Fallback CPU encode |
| `src/cv_evidence_renderer/overlay.py` | Bbox burn-in (cv2; supervision-compatible) |
| `src/cv_evidence_renderer/io/rtsp.py` | Threaded RTSP reader, auto-reconnect |
| `src/cv_evidence_renderer/adapters.py` | `sv.Detections` ↔ internal format ↔ raw JSONL |
| `src/cv_evidence_renderer/pool.py` | Multi-stream encoder pool |
| `src/cv_evidence_renderer/cli.py` | Typer entrypoint |

---

## Design notes

- **Why a ring buffer instead of just trimming after the fact?** Because in live mode you don't *have* the future — when an event fires at frame N, you need to have already been keeping frames N-150 to N. A naive trim-after-event approach only works for saved video files (see "use case B"), and most teams hit the live case first.
- **Why bbox burn-in instead of metadata sidecar?** Evidence clips get sent to non-technical operators (compliance, ops) who open them in QuickTime / VLC. A sidecar JSON they can't read is useless. The burn-in is the point.
- **Why PyAV instead of `subprocess` to FFmpeg?** Race conditions when multiple streams write to the same FFmpeg subprocess pool. PyAV gives a clean Python object per stream and re-uses the libav encoder context.
- **Why interop with supervision instead of re-implementing drawing?** Because supervision (39.5k ⭐) does it better than we ever will, and "ride the ecosystem" is faster than "compete for drawing API mindshare". Our moat is the event-clip pipeline, not the rectangle drawing.
- **Why keyframe-aware seek for pre-buffer?** Because seeking to a non-keyframe in libav gives you garbage frames until the next IDR. The buffer indexes keyframes and snaps pre-buffer start to the nearest one.

---

## Roadmap

- DeepStream sink integration (close the [no-Python-binding](https://forums.developer.nvidia.com/t/how-to-use-smart-record-in-deepstream-6-1-python/231682) gap)
- Overlapping recordings on the same stream ([feature DeepStream explicitly doesn't support](https://forums.developer.nvidia.com/t/need-parallel-overlap-recording-on-the-same-stream/337137))
- Per-event metadata sidecar (JSON + MKV chapters)
- ONVIF event trigger input

---

## Comparison to similar tools

See [COMPETITORS.md](COMPETITORS.md) for the full research write-up.

| | cv-evidence-renderer | supervision | DeepStream Smart Record | KeyClipWriter |
|---|---|---|---|---|
| Python-only install | ✅ | ✅ | ❌ (needs DeepStream SDK) | ✅ |
| Event-window trim (offline) | ✅ | ❌ | ✅ (C only) | ✅ |
| **Cross-file event concat** (NVR-style split files) | ✅ | ❌ | ❌ | ❌ |
| **Decode-once batch** for N events on one source | ✅ | ❌ | ❌ | ❌ |
| Output duration cap (timelapse / framedrop) | ✅ | ❌ | ❌ | ❌ |
| Bbox burn-in into evidence | ✅ | ✅ (excellent) | ⚠️ (bug since 6.4) | ❌ |
| supervision interop | ✅ | — | ❌ | ❌ |
| Ultralytics YOLO adapter | ✅ | ✅ (via `from_ultralytics`) | ❌ | ❌ |
| NVENC encode | 🚧 v0.2 | ❌ | ✅ | ❌ |
| Live RTSP ring buffer | 🚧 v0.2 | ❌ | ✅ (C only) | ✅ |
| Multi-stream pool | 🚧 v0.3 | ❌ | ✅ | ❌ |

---

## License

MIT
