import logging

from mcp.server.fastmcp import Context

from ..video_store import VideoStore
from .download import do_download
from .transcribe import do_transcribe

logger = logging.getLogger(__name__)


async def do_analyze(  # noqa: PLR0913
    store: VideoStore,
    url: str,
    whisper_model: str = "turbo",
    diarize_speakers: bool = True,
    align_language: str = "auto",
    *,
    ctx: Context | None = None,
) -> dict:
    logger.info("Analyzing video url=%s", url)
    if ctx:
        await ctx.info("Downloading video...")
    video_id = await do_download(store, url)
    if ctx:
        await ctx.info("Download complete, starting transcription...")
    transcript = await do_transcribe(store, video_id, whisper_model, diarize_speakers, align_language, ctx=ctx)
    logger.info("Analysis complete video_id=%s", video_id)
    return {
        "video_id": video_id,
        "transcript": transcript,
    }
