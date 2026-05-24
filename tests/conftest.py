"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from cv_evidence_renderer.encoder import nvenc


def gpu_available() -> bool:
    return nvenc.is_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests marked `gpu` when no NVENC is available."""
    if gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="NVENC not available on this runner")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
