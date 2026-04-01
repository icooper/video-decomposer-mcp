import asyncio
import logging
import os
import threading
from functools import partial
from typing import Any

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from ..video_store import VideoStore

logger = logging.getLogger(__name__)

_whisper_cache: dict[str, Any] = {}
_align_cache: dict[str, Any] = {}
_diarize_cache: dict[str, DiarizationPipeline] = {}
_model_lock = threading.Lock()


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_whisper_model(whisper_model: str) -> Any:
    with _model_lock:
        if whisper_model not in _whisper_cache:
            device = _get_device()
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info("Loading WhisperX model '%s' on %s (%s)", whisper_model, device, compute_type)
            _whisper_cache[whisper_model] = whisperx.load_model(whisper_model, device=device, compute_type=compute_type)
        else:
            logger.debug("Using cached WhisperX model '%s'", whisper_model)
        return _whisper_cache[whisper_model]


def _get_align_model(language_code: str) -> Any:
    with _model_lock:
        if language_code not in _align_cache:
            device = _get_device()
            logger.info("Loading alignment model for language '%s' on %s", language_code, device)
            align_model = whisperx.load_align_model(language_code=language_code, device=device)
            _align_cache[language_code] = align_model
        else:
            logger.debug("Using cached alignment model for language '%s'", language_code)
        return _align_cache[language_code]


def _get_diarization_pipeline() -> DiarizationPipeline | None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "HF_TOKEN environment variable is required for speaker diarization. "
            "Accept the pyannote model conditions at https://hf.co/pyannote/speaker-diarization-3.1 "
            "and set HF_TOKEN to a Hugging Face access token."
        )
        return None
    with _model_lock:
        if "pipeline" not in _diarize_cache:
            device = _get_device()
            logger.info("Loading diarization pipeline on %s", device)
            pipeline = DiarizationPipeline(token=hf_token, device=device)
            _diarize_cache["pipeline"] = pipeline
        else:
            logger.debug("Using cached diarization pipeline")
        return _diarize_cache["pipeline"]


def preload_whisper_model(whisper_model: str) -> None:
    """Preload a WhisperX model into the cache."""
    _get_whisper_model(whisper_model)


def preload_align_model(language_code: str) -> None:
    """Preload an alignment model into the cache."""
    _get_align_model(language_code)


def preload_diarization_pipeline() -> None:
    """Preload the diarization pipeline. Logs a warning and returns if HF_TOKEN is not set."""
    _get_diarization_pipeline()


def _build_annotated_text(segments: list[dict]) -> str:
    """Build speaker-annotated text from diarized segments, consolidating adjacent segments from the same speaker."""
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg["text"].strip()
        if lines and lines[-1][0] == speaker:
            lines[-1] = (speaker, f"{lines[-1][1]} {text}")
        else:
            lines.append((speaker, text))
    return "\n".join(f"{speaker}: {text}" for speaker, text in lines)


def _transcribe(file_path: str, whisper_model: str, diarize_speakers: bool, align_language: str = "auto") -> dict:
    audio = whisperx.load_audio(file_path)
    model = _get_whisper_model(whisper_model)
    result = model.transcribe(audio)
    pipeline = _get_diarization_pipeline() if diarize_speakers else None

    if pipeline is None:
        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result.get("segments", [])
        ]
        text = "".join(seg["text"] for seg in segments)
        return {"text": text, "segments": segments}

    # Alignment: get word-level timestamps
    language = result.get("language", "en") if align_language == "auto" else align_language
    model_a, metadata = _get_align_model(language)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device=_get_device())

    # Diarization: identify speakers
    diarize_segments = pipeline(audio)

    # Assign speakers to words and segments
    diarized = whisperx.assign_word_speakers(diarize_segments, aligned)

    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"], "speaker": seg.get("speaker", "UNKNOWN")}
        for seg in diarized.get("segments", [])
    ]
    text = _build_annotated_text(segments)
    return {"text": text, "segments": segments}


async def do_transcribe(
    store: VideoStore,
    video_id: str,
    whisper_model: str = "turbo",
    diarize_speakers: bool = True,
    align_language: str = "auto",
) -> dict:
    logger.info(
        "Transcribing video_id=%s model=%s diarize=%s lang=%s",
        video_id,
        whisper_model,
        diarize_speakers,
        align_language,
    )
    record = store.get(video_id)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, partial(_transcribe, str(record.file_path), whisper_model, diarize_speakers, align_language)
    )
    logger.debug("Transcription complete video_id=%s length=%d chars", video_id, len(result["text"]))
    return result
