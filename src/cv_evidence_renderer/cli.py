"""Typer CLI entry point — `cv-evidence render` and `cv-evidence batch`."""

from __future__ import annotations

from pathlib import Path

import typer

from cv_evidence_renderer._version import __version__
from cv_evidence_renderer.offline import render_from_jsonl
from cv_evidence_renderer.types import Encoder

app = typer.Typer(
    name="cv-evidence",
    help="Render bbox-burned evidence clips from detector output. See https://github.com/ddinhcchi/cv-evidence-renderer",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print version and exit."""
    typer.echo(__version__)


@app.command()
def render(
    input: Path = typer.Option(..., "--input", "-i", help="Input video file"),
    detections: Path = typer.Option(..., "--detections", "-d", help="JSONL detections file"),
    event_start: float = typer.Option(..., "--event-start", help="Event start, seconds"),
    event_end: float = typer.Option(..., "--event-end", help="Event end, seconds"),
    output: Path = typer.Option(..., "--output", "-o", help="Output MP4 path"),
    encoder: Encoder = typer.Option(Encoder.AUTO, "--encoder", help="Video encoder"),
) -> None:
    """Render a single evidence clip from a saved video + detections JSONL."""
    out = render_from_jsonl(
        video=input,
        detections_jsonl=detections,
        event_start=event_start,
        event_end=event_end,
        output=output,
        encoder=encoder,
    )
    typer.echo(f"Wrote {out}")


@app.command()
def batch(
    inputs: Path = typer.Option(..., "--inputs", help="Directory of input videos"),
    events: Path = typer.Option(..., "--events", help="events.jsonl"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers"),
) -> None:
    """Batch-render many evidence clips from a directory of videos + events.jsonl. (v0.3)"""
    raise NotImplementedError("v0.3 — see SPEC.md §8")


if __name__ == "__main__":
    app()
