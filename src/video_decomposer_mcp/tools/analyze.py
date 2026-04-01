import logging

from ..video_store import VideoStore
from .download import do_download
from .transcribe import do_transcribe

logger = logging.getLogger(__name__)


async def do_analyze(
    store: VideoStore,
    url: str,
    whisper_model: str = "turbo",
    diarize_speakers: bool = True,
    align_language: str = "auto",
) -> dict:
    logger.info("Analyzing video url=%s", url)
    logger.debug("Step: download")
    video_id = await do_download(store, url)
    logger.debug("Step: transcribe")
    transcript = await do_transcribe(store, video_id, whisper_model, diarize_speakers, align_language)
    logger.info("Analysis complete video_id=%s", video_id)
    return {
        "video_id": video_id,
        "transcript": transcript,
    }
