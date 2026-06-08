"""cv-evidence-renderer — the missing evidence-clip layer between detector and storage."""

from cv_evidence_renderer._version import __version__
from cv_evidence_renderer.offline import render_clip, render_clips, render_from_jsonl
from cv_evidence_renderer.recorder import EvidenceRecorder
from cv_evidence_renderer.types import Clip, ClipSource

__all__ = [
    "Clip",
    "ClipSource",
    "EvidenceRecorder",
    "__version__",
    "render_clip",
    "render_clips",
    "render_from_jsonl",
]
