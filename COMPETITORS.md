# Competitor Research — cv-evidence-renderer

> Research date: 2026-05-24. Method: GitHub search + WebFetch on top hits + NVIDIA DeepStream forum + PyImageSearch + discuss.python.org.

## Direct competitors

| Repo | ⭐ | Last commit | GPU? | Trim/record? | Multi-stream? | Gap vs cv-evidence-renderer |
|---|---|---|---|---|---|---|
| [roboflow/supervision](https://github.com/roboflow/supervision) | ~39.5k | Very active 2026 | No (CPU OpenCV) | Partial — `VideoSink` is OpenCV `mp4v` only, no event-window trim | No | No NVENC, no RTSP-trigger record, no pre/post buffer event clips, codec stuck on `mp4v` |
| [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps) | ~1.8k | Sep 2025 | Yes (NVENC/CUDA via GStreamer) | Smart Record exists in C, **not exposed in Python bindings** | Yes | DeepStream-locked, very steep learning curve, no clean Python API for smart-record |
| [prominenceai/deepstream-services-library (DSL)](https://github.com/prominenceai/deepstream-services-library) | ~341 | Active | Yes | Yes (Record Tap + ODE Actions) | Yes | DeepStream/GStreamer-locked, C++17 lib with Python C-bindings, niche audience, high friction install |
| [abhiTronix/vidgear](https://github.com/abhiTronix/vidgear) | ~7k | Active | Partial (NVENC via FFmpeg user-params) | WriteGear writes streams, no event-window logic | Yes (multi-thread) | Generic video framework — no detection/bbox concept, user must hand-build event trimming + overlay |
| [obss/sahi](https://github.com/obss/sahi) | ~5.3k | Sep 2025 | No | No (CPU video predict only) | No | Inference-focused; bbox render is incidental |
| [shoumikchow/bbox-visualizer](https://github.com/shoumikchow/bbox-visualizer) | ~400 | Low activity | No | No | No | Just image-level drawing utilities |

## Adjacent tools

| Repo | ⭐ | One-liner | Overlap |
|---|---|---|---|
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 35k+ | YOLO trainer; `results.save()` writes annotated MP4 | Auto-annotates whole video, no event-window/trim, OpenCV writer |
| [tryolabs/norfair](https://github.com/tryolabs/norfair) | ~2.6k | Lightweight MOT tracker, has `Video` helper | Drawing helpers, no record/trim |
| [voxel51/fiftyone](https://github.com/voxel51/fiftyone) | ~10k+ | Dataset curation + viz, `to_clips()` for temporal labels | Clip *views* on datasets, not encoded MP4 evidence; dataset tool, not renderer |
| [facebookresearch/pytorchvideo](https://github.com/facebookresearch/pytorchvideo) | ~3k | Video understanding lib, has `VideoVisualizer` | Bbox/action overlay only |
| [kkroening/ffmpeg-python](https://github.com/kkroening/ffmpeg-python) | ~10k | FFmpeg bindings — used to build NVENC pipelines | Low-level building block, not detection-aware |
| [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | ~3k | Scene cut detection + split | Event-window split paradigm parallel, not detection-driven |
| [PyImageSearch KeyClipWriter (tutorial)](https://pyimagesearch.com/2016/02/29/saving-key-event-video-clips-with-opencv/) | — | Ring-buffer pre/post event clip writer in OpenCV | **This is the canonical DIY pattern users currently copy.** No library form, no GPU. |

## Deep dive — 3 closest

### roboflow/supervision (~39.5k ⭐)
- Dominant Python CV utility library. Comprehensive `Annotator` classes (Box/Label/Mask/Trace/Polygon etc.), `ByteTrack`, dataset I/O.
- Video API intentionally thin: `VideoSink` is a context manager wrapping `cv2.VideoWriter` with `mp4v` codec hard-coded ([issue #380](https://github.com/roboflow/supervision/issues/380) is open requesting fourcc customization).
- **No NVENC, no RTSP-as-source, no event-triggered trim, no pre/post buffer, no multi-stream orchestration.** GPU only used for inference upstream; encoding is CPU.
- Feature requests like [#357 real-time streaming](https://github.com/roboflow/supervision/issues/357) and [#1909 GPU not accelerated](https://github.com/roboflow/supervision/issues/1909) show users repeatedly hitting the encoding ceiling.

### NVIDIA-AI-IOT/deepstream_python_apps (~1.8k ⭐)
- Python bindings (pyds) for DeepStream SDK. Sample apps like deepstream-test5 demo smart-record, but **the Smart Record APIs themselves have no Python bindings**. NVIDIA confirmed: *"There is no python binding for smart recording APIs. It is not supported with pyds now."*
- Requires DeepStream SDK install, NVIDIA hardware, GStreamer fluency. High barrier; community treats it as last-resort.

### obss/sahi (~5.3k ⭐)
- Slicing-aided inference for small objects. Has `predict` CLI for video, bbox visualization is purely for debug. **No record/trim/RTSP/GPU-encode.** Not a real competitor; orthogonal concern.

## Evidence of user demand

### DeepStream forum (active 2024–2026)
- [Deepstream 6.4 Smart Record Video Issue with bbox enabled](https://forums.developer.nvidia.com/t/deepstream-6-4-smart-record-video-issue-with-bbox-enabled/290732) — *"When we use the smart record bin at the sink end of the pipeline, after the encoder, the resulting video is glitchy with randomly occurring gray/green dots."* **Closed without resolution.**
- [How to use smart record in deepstream 6.1 python](https://forums.developer.nvidia.com/t/how-to-use-smart-record-in-deepstream-6-1-python/231682) — NVIDIA staff reply: *"There is no python binding for smart recording APIs. It is not supported with pyds now."*
- [Deepstream smart record implementation](https://forums.developer.nvidia.com/t/deepstream-smart-record-implementation/362581) (2026) — user struggling to trigger smart-record from cloud events; required undocumented config patches.
- [Need parallel/overlap recording on the same stream](https://forums.developer.nvidia.com/t/need-parallel-overlap-recording-on-the-same-stream/337137) (2025) — DeepStream explicitly does not support overlapping recordings on a stream.
- [Overlapping smart record feature](https://forums.developer.nvidia.com/t/overapping-smart-record-feature/327050) (March 2025) — user asking for timeline; NVIDIA non-committal.

**Pattern:** real users want event-clip-with-bbox today; the only "official" answer (DeepStream Smart Record) is C-only, buggy with bbox, no Python.

### Other channels
- [discuss.python.org — "OpenCV: Constant video stream, write file 1 minute before and after an event"](https://discuss.python.org/t/opencv-constant-video-stream-write-file-1-minute-before-and-after-an-event/32385) — exactly the use case.
- PyImageSearch KeyClipWriter tutorial pageviews + linked-from blog posts — canonical DIY pattern that has zero library equivalent.

**Caveat:** r/computervision search was unreliable (Reddit search not well-indexed externally). Treat user-demand evidence as **DeepStream-forum-strong, Reddit-weak**.

---

## Verdict

**Niche empty?** Mostly yes. The exact combination — *Python-first library*, *detection-JSON-in / MP4-evidence-out*, *NVENC-accelerated*, *event-window trim with pre/post buffer*, *multi-stream parallel*, *no DeepStream dependency* — is not covered by any single tool. Supervision owns the drawing API; DeepStream owns the GPU pipeline; nobody bridges them cleanly for non-DeepStream users. The DIY pattern is hand-rolled OpenCV+FFmpeg subprocess.

### Top 3 differentiators (REAL)

1. **Python-native NVENC encoding with bbox burn-in** — supervision is CPU-only, DeepStream needs C; genuine gap.
2. **Event-window trim with pre/post buffer driven by detection JSON** — supervision has no concept of events. KeyClipWriter exists but is detection-agnostic. **Strongest differentiator.**
3. **Decoupled from DeepStream/GStreamer while still GPU-accelerated** — addresses the large pool of users who want NVIDIA GPUs but not the DeepStream learning curve.

### Top 2 differentiators (WEAK — must reposition)

1. ~~**"Bbox/label overlay rendering"** alone is commodity.~~ supervision crushes this. **Do NOT position the project here.**
2. ~~**"Multi-stream parallel"**~~ — VidGear, DeepStream, even bare `multiprocessing+ffmpeg` cover this. Table-stakes, not a moat. Mention as feature, never as headline.

### Recommendation: **BUILD, but reposition**

- **New positioning:** *"the missing evidence-clip layer between your detector and storage"* — not a visualization library (supervision wins), not a GStreamer pipeline (DeepStream/DSL win).
- **Headline features:** (a) event-window trim with pre-buffer (b) NVENC out of the box (c) Python-only install, no DeepStream.
- **Strongly consider interop with supervision's annotators** (accept `sv.Detections`) — ride the ecosystem rather than re-implement drawing. Dramatically reduces scope and pitches you as the "clip layer" complementing supervision.
- 6 weeks reasonable IF drawing is thin wrapper over supervision/OpenCV and budget goes to FFmpeg/NVENC pipeline + ring-buffer event logic + multi-stream orchestration. If you try to out-draw supervision, **you will lose.**
