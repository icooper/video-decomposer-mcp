import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from . import configure_logging
from .tools.analyze import do_analyze
from .tools.download import do_download
from .tools.frames import do_extract_frame
from .tools.transcribe import do_transcribe
from .video_store import VideoStore

logger = logging.getLogger(__name__)

WHISPER_MODEL_DESCRIPTION = (
    'Whisper model size: "turbo" (fast, good quality), "base", "small", "medium", or "large" (slow, best quality)'
)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _cleanup_loop():
    while True:
        await asyncio.sleep(600)
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

# get video store path from environment variable, defaulting to ./video_store if not set
store_path = Path(os.environ.get("VIDEO_STORE_PATH", "./video_store"))
store = VideoStore(store_path)


@mcp.tool()
async def download_video(
    url: Annotated[
        str, Field(description="URL of the video to download (YouTube, Facebook, Instagram, or other supported site)")
    ],
) -> str:
    """Download a video from a URL without transcribing or extracting frames.
    Use this when you only need to download now and process later, or when
    you want to transcribe and extract frames separately.

    Supports YouTube, Facebook, Instagram, and most video hosting sites.

    Returns a video_id string. Pass this ID to transcribe_video or
    extract_frames to process the video. Downloaded videos expire after
    4 hours."""
    return await do_download(store, url)


@mcp.tool()
async def transcribe_video(
    video_id: Annotated[str, Field(description="ID returned by download_video or analyze_video")],
    model: Annotated[
        str,
        Field(description=WHISPER_MODEL_DESCRIPTION),
    ] = "turbo",
) -> dict:
    """Transcribe the audio of a previously downloaded video to text using
    OpenAI Whisper. Requires a video_id from download_video or analyze_video.

    The default "turbo" model works well for English and most major languages.
    Whisper supports 99 languages total, but accuracy varies — English is the
    strongest, and low-resource languages may have reduced accuracy. For best
    results on non-English audio, consider using the "large" model at the cost
    of slower processing.

    Returns a dict with "text" (the full transcript) and "segments" (a list
    of {start, end, text} objects with timestamps in seconds). Use the
    segments to correlate speech with frames from extract_frames."""
    return await do_transcribe(store, video_id, model)


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
    max_dimension: Annotated[int, Field(description="Maximum pixel size on the longest edge")] = 768,
    quality: Annotated[int, Field(description="JPEG compression quality (1-100)")] = 75,
) -> dict:
    """Extract a single video frame at a specific timestamp as a base64-encoded
    JPEG image. Use this to see what was shown on screen at a particular moment.

    Requires a video_id from download_video or analyze_video. Use the timestamps
    from transcript segments (returned by analyze_video or transcribe_video) to
    choose which moments to visualize.

    Returns a dict with the base64-encoded JPEG image and the timestamp."""
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
    ] = "turbo",
) -> dict:
    """Analyze a video from a URL — downloads it and transcribes the audio.
    This is the best starting point when a user asks you to watch, review,
    look at, summarize, or understand a video.

    Supports YouTube, Facebook, Instagram, and most video hosting sites.

    Returns a video_id (for follow-up calls) and the transcript with timestamped
    segments. Each segment has start/end times in seconds and the spoken text.

    To see what was shown at a specific moment, use extract_frame with a
    timestamp from the transcript segments."""
    return await do_analyze(
        store,
        url,
        whisper_model,
    )


def main():
    configure_logging(logging.INFO)
    mcp.run(transport="sse")
