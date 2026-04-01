import asyncio
import logging
import threading
from functools import partial

import torch
import whisper

from ..video_store import VideoStore

logger = logging.getLogger(__name__)

_model_cache: dict[str, whisper.Whisper] = {}
_model_lock = threading.Lock()


def _get_whisper_model(whisper_model: str) -> whisper.Whisper:
    with _model_lock:
        if whisper_model not in _model_cache:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading Whisper model '%s' on %s", whisper_model, device)
            _model_cache[whisper_model] = whisper.load_model(whisper_model, device=device)
        else:
            logger.debug("Using cached Whisper model '%s'", whisper_model)
        return _model_cache[whisper_model]


def preload_whisper_model(whisper_model: str) -> None:
    """Preload a Whisper model into the cache. Useful for warming up before handling requests."""
    _get_whisper_model(whisper_model)


def _transcribe(file_path: str, whisper_model: str) -> dict:
    model = _get_whisper_model(whisper_model)
    result: dict = model.transcribe(file_path)
    segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result.get("segments", [])]
    return {"text": result["text"], "segments": segments}


async def do_transcribe(store: VideoStore, video_id: str, whisper_model: str = "turbo") -> dict:
    logger.info("Transcribing video_id=%s model=%s", video_id, whisper_model)
    record = store.get(video_id)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, partial(_transcribe, str(record.file_path), whisper_model))
    logger.debug("Transcription complete video_id=%s length=%d chars", video_id, len(result["text"]))
    return result
