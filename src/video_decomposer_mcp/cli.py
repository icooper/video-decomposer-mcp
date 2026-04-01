import asyncio
import logging
from pathlib import Path

import typer

from . import configure_logging
from .tools.analyze import do_analyze
from .tools.download import do_download
from .tools.frames import do_extract_frame
from .tools.transcribe import do_transcribe, preload_whisper_model
from .video_store import VideoStore

app = typer.Typer()

# use a local directory for the video store in the CLI
store = VideoStore(Path("./video_store"))


@app.command()
def preload(whisper_model: str = "turbo") -> None:
    """Preload a Whisper model into the cache. Useful for warming up before handling requests."""
    preload_whisper_model(whisper_model)
    typer.echo(f"Model '{whisper_model}' preloaded into cache.")


@app.command()
def download(url: str) -> None:
    """Download a video and print its video_id."""
    video_id = asyncio.run(do_download(store, url))
    typer.echo(video_id)


@app.command()
def transcribe(
    video_id: str,
    whisper_model: str = typer.Option("turbo", help="Whisper model size: turbo, base, small, medium, or large"),
    diarize_speakers: bool = typer.Option(False, "--diarize-speakers/--no-diarize-speakers", help="Identify speakers"),
    align_language: str = typer.Option("auto", help="Language code for alignment (e.g. en, fr) or auto to detect"),
) -> None:
    """Transcribe a downloaded video."""
    result = asyncio.run(do_transcribe(store, video_id, whisper_model, diarize_speakers, align_language))
    typer.echo(result["text"])


@app.command()
def extract_frame(
    video_id: str,
    timestamp: float,
    output_dir: Path = typer.Option(Path("./frames"), help="Directory to save the frame"),
) -> None:
    """Extract a single frame at a specific timestamp from a downloaded video."""
    result = asyncio.run(do_extract_frame(store, video_id, timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"frame_{int(timestamp * 1000):08d}.jpg"
    path = output_dir / filename
    path.write_bytes(result.data)
    typer.echo(f"Saved frame at {timestamp}s to {path}")


@app.command()
def analyze(
    url: str,
    whisper_model: str = typer.Option("turbo", help="Whisper model size: turbo, base, small, medium, or large"),
    diarize_speakers: bool = typer.Option(False, "--diarize-speakers/--no-diarize-speakers", help="Identify speakers"),
    align_language: str = typer.Option("auto", help="Language code for alignment (e.g. en, fr) or auto to detect"),
) -> None:
    """Download and transcribe a video in one step."""
    result = asyncio.run(do_analyze(store, url, whisper_model, diarize_speakers, align_language))
    typer.echo(f"Video ID: {result['video_id']}")
    typer.echo(f"Transcript: {result['transcript']['text']}")


def main():
    configure_logging(logging.ERROR)  # can be overridden by LOG_LEVEL environment variable
    store.cleanup()
    app()
