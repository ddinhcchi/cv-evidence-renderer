"""cv-evidence-renderer — the missing evidence-clip layer between detector and storage."""

from cv_evidence_renderer._version import __version__
from cv_evidence_renderer.offline import render_from_jsonl
from cv_evidence_renderer.recorder import EvidenceRecorder

__all__ = [
    "EvidenceRecorder",
    "__version__",
    "render_from_jsonl",
]
