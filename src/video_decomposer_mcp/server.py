import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP, Image
from pydantic import Field

from . import configure_logging
from .tools.analyze import do_analyze
from .tools.download import do_download
from .tools.frames import do_extract_frame
from .tools.transcribe import do_transcribe, preload_align_model, preload_diarization_pipeline, preload_whisper_model
from .video_store import VideoStore

logger = logging.getLogger(__name__)
configure_logging(logging.INFO)

WHISPER_MODEL_DESCRIPTION = (
    'Whisper model size: "turbo" (fast, good quality), "base", "small", "medium", or "large" (slow, best quality)'
)
ALIGN_LANGUAGE_DESCRIPTION = (
    'Language code for alignment (e.g. "en", "fr", "de", "ja"). The default is to auto-detect from the transcription.'
)

# get configuration from environment variables
store_path = Path(os.environ.get("VIDEO_STORE_PATH", "./video_store"))  # in the container, this is /app/video_store
video_store_ttl_seconds = int(os.environ.get("VIDEO_STORE_TTL_SECONDS", "14400"))
video_store_cleanup_interval_seconds = int(os.environ.get("VIDEO_STORE_CLEANUP_INTERVAL_SECONDS", "600"))
default_whisper_model = os.environ.get("WHISPER_MODEL", "turbo")
default_align_language = os.environ.get("PRELOAD_ALIGN_LANGUAGE", "en")
hf_token = os.environ.get("HF_TOKEN", "")

# create video store
store = VideoStore(store_path, ttl_seconds=video_store_ttl_seconds)


@asynccontextmanager
async def lifespan(app):
    # preload models on startup to reduce the first-request latency
    preload_whisper_model(default_whisper_model)
    logger.info("WhisperX '%s' model preloaded", default_whisper_model)
    preload_align_model(default_align_language)
    logger.info("Alignment model for '%s' preloaded", default_align_language)
    if len(hf_token) > 0:
        preload_diarization_pipeline()
        logger.info("Diarization pipeline preloaded")

    # start the cleanup loop to remove expired videos from the store every 10 minutes
    task = asyncio.create_task(_cleanup_loop())

    try:
        yield

    finally:
        # stop the cleanup loop on shutdown
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _cleanup_loop():
    while True:
        await asyncio.sleep(video_store_cleanup_interval_seconds)
        try:
            count = await store.async_cleanup()
            if count > 0:
                logger.info("Cleaned up %d expired videos", count)
        except Exception:
            logger.exception("Error during video store cleanup")


mcp = FastMCP(
    "Video Decomposer",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8000")),
    lifespan=lifespan,
)


@mcp.tool()
async def download_video(
    url: Annotated[
        str, Field(description="URL of the video to download (YouTube, Facebook, Instagram, or other supported site)")
    ],
    ctx: Context | None = None,
) -> str:
    """Download a video from a URL without transcribing or extracting frames.
    Use this when you only need to download now and process later, or when
    you want to transcribe and extract frames separately.

    Supports YouTube, Facebook, Instagram, and most video hosting sites.

    Returns a video_id string. Pass this ID to transcribe_video or
    extract_frame to process the video. Downloaded videos expire after
    4 hours."""
    if ctx:
        await ctx.info("Downloading video...")
    return await do_download(store, url)


@mcp.tool()
async def transcribe_video(
    video_id: Annotated[str, Field(description="ID returned by download_video or analyze_video")],
    whisper_model: Annotated[
        str,
        Field(description=WHISPER_MODEL_DESCRIPTION),
    ] = default_whisper_model,
    diarize_speakers: Annotated[
        bool,
        Field(description=("Identify who is speaking in each segment.")),
    ] = False,
    align_language: Annotated[
        str,
        Field(description=ALIGN_LANGUAGE_DESCRIPTION),
    ] = "auto",
    ctx: Context | None = None,
) -> dict:
    """Transcribe the audio of a previously downloaded video to text using
    WhisperX (faster-whisper). Requires a video_id from download_video or
    analyze_video.

    The default "turbo" model works well for English and most major languages.
    Whisper supports 99 languages total, but accuracy varies — English is the
    strongest, and low-resource languages may have reduced accuracy. For best
    results on non-English audio, consider using the "large" model at the cost
    of slower processing.

    Each segment has start/end times in seconds and the spoken text. To add
    speaker labels to the segments and the full text, set diarize_speakers=true.

    Returns a dict with "text" (the full transcript) and "segments" (a list
    of {start, end, text, speaker?} objects with timestamps in seconds). Use
    the segments to correlate speech with frames from extract_frame."""
    return await do_transcribe(store, video_id, whisper_model, diarize_speakers, align_language, ctx=ctx)


@mcp.tool()
async def extract_frame(
    video_id: Annotated[str, Field(description="ID returned by download_video or analyze_video")],
    timestamp: Annotated[
        float,
        Field(
            description=(
                "Timestamp in seconds of the frame to extract. "
                "Use timestamps from analyze_video or transcribe_video transcript segments"
            )
        ),
    ],
    max_dimension: Annotated[int, Field(description="Maximum pixel size on the longest edge (default 768px)")] = 768,
    quality: Annotated[int, Field(description="JPEG compression quality (1-100)")] = 75,
) -> Image:
    """Extract a single video frame at a specific timestamp as a JPEG image.
    Use this to see what was shown on screen at a particular moment.

    Requires a video_id from download_video or analyze_video. Use the timestamps
    from transcript segments (returned by analyze_video or transcribe_video) to
    choose which moments to visualize.

    The image is resized so its longest edge is at most max_dimension pixels
    (default 768), preserving aspect ratio."""
    return await do_extract_frame(
        store,
        video_id,
        timestamp,
        max_dimension=max_dimension,
        quality=quality,
    )


@mcp.tool()
async def analyze_video(
    url: Annotated[
        str, Field(description="URL of the video to analyze (YouTube, Facebook, Instagram, or other supported site)")
    ],
    whisper_model: Annotated[
        str,
        Field(description=WHISPER_MODEL_DESCRIPTION),
    ] = default_whisper_model,
    diarize_speakers: Annotated[
        bool,
        Field(description=("Identify who is speaking in each segment.")),
    ] = False,
    align_language: Annotated[
        str,
        Field(description=ALIGN_LANGUAGE_DESCRIPTION),
    ] = "auto",
    ctx: Context | None = None,
) -> dict:
    """Analyze a video from a URL — downloads it and transcribes the audio
    with optional speaker diarization.

    This is the best starting point when a user asks you to watch, review,
    look at, summarize, or understand a video.

    Supports YouTube, Facebook, Instagram, and most video hosting sites.

    Returns a video_id (for follow-up calls) and the transcript with timestamped
    segments. Each segment has start/end times in seconds and the spoken text. To
    add speaker labels to the segments and the full text, set diarize_speakers=true.

    To see what was shown at a specific moment, use extract_frame with a
    timestamp from the transcript segments."""
    return await do_analyze(
        store,
        url,
        whisper_model,
        diarize_speakers,
        align_language,
        ctx=ctx,
    )


def main():
    mcp.run(transport="sse")
