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


def _get_model(model_name: str) -> whisper.Whisper:
    with _model_lock:
        if model_name not in _model_cache:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading Whisper model '%s' on %s", model_name, device)
            _model_cache[model_name] = whisper.load_model(model_name, device=device)
        else:
            logger.debug("Using cached Whisper model '%s'", model_name)
        return _model_cache[model_name]


def preload_model(model_name: str) -> None:
    """Preload a Whisper model into the cache. Useful for warming up before handling requests."""
    _get_model(model_name)


def _transcribe(file_path: str, model_name: str) -> dict:
    model = _get_model(model_name)
    result = model.transcribe(file_path)
    segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result.get("segments", [])]
    return {"text": result["text"], "segments": segments}


async def do_transcribe(store: VideoStore, video_id: str, model_name: str = "turbo") -> dict:
    logger.info("Transcribing video_id=%s model=%s", video_id, model_name)
    record = store.get(video_id)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, partial(_transcribe, str(record.file_path), model_name))
    logger.debug("Transcription complete video_id=%s length=%d chars", video_id, len(result["text"]))
    return result
