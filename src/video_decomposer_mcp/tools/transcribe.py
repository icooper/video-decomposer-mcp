import asyncio
import json
import logging
import os
import tempfile
import threading
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import whisperx
from mcp.server.fastmcp import Context
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


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types from WhisperX output."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _read_cache(path: Path) -> Any:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _write_cache(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, cls=_NumpyEncoder)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


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


def _transcribe_stage(file_path: str, video_dir_path: Path, whisper_model: str) -> tuple[dict, Any]:
    """Run or load cached transcription. Returns (result_dict, audio_or_None)."""
    transcription_cache_path = video_dir_path / f"transcription_cache_{whisper_model}.json"
    result = _read_cache(transcription_cache_path)
    if result is not None:
        logger.info("Loaded cached transcription for model '%s'", whisper_model)
        return result, None

    audio = whisperx.load_audio(file_path)
    model = _get_whisper_model(whisper_model)
    result = model.transcribe(audio)
    _write_cache(transcription_cache_path, result)
    logger.info("Cached transcription for model '%s'", whisper_model)
    return result, audio


def _align_stage(  # noqa: PLR0913
    file_path: str, video_dir_path: Path, whisper_model: str, result: dict, audio: Any, align_language: str
) -> tuple[dict, Any]:
    """Run or load cached alignment. Returns (aligned_dict, audio_or_None)."""
    language = result.get("language", "en") if align_language == "auto" else align_language
    alignment_cache_path = video_dir_path / f"alignment_cache_{whisper_model}_{language}.json"
    aligned = _read_cache(alignment_cache_path)
    if aligned is not None:
        logger.info("Loaded cached alignment for model '%s' language '%s'", whisper_model, language)
        return aligned, audio

    if audio is None:
        audio = whisperx.load_audio(file_path)
    model_a, metadata = _get_align_model(language)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device=_get_device())
    _write_cache(alignment_cache_path, aligned)
    logger.info("Cached alignment for model '%s' language '%s'", whisper_model, language)
    return aligned, audio


def _diarize_stage(
    file_path: str, video_dir_path: Path, whisper_model: str, pipeline: DiarizationPipeline, audio: Any
) -> tuple[Any, Any]:
    """Run or load cached diarization. Returns (diarize_segments_df, audio_or_None)."""

    # diarization shouldn't depend on the language itself, but we include the model in the cache key since different
    # models may produce different segmentations that could affect diarization quality
    diarization_cache_path = video_dir_path / f"diarization_cache_{whisper_model}.json"

    cached = _read_cache(diarization_cache_path)
    if cached is not None:
        logger.info("Loaded cached diarization for model '%s'", whisper_model)
        return pd.DataFrame(cached), audio

    if audio is None:
        audio = whisperx.load_audio(file_path)
    diarize_segments = pipeline(audio)
    cache_list = [
        {"start": row["start"], "end": row["end"], "speaker": row["speaker"]} for _, row in diarize_segments.iterrows()
    ]
    _write_cache(diarization_cache_path, cache_list)
    logger.info("Cached diarization for model '%s'", whisper_model)
    return diarize_segments, audio


def _assign_speakers_stage(diarize_segments: Any, aligned: dict) -> dict:
    """Assign speakers to segments and build final result."""
    diarized = whisperx.assign_word_speakers(diarize_segments, aligned)
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"], "speaker": seg.get("speaker", "UNKNOWN")}
        for seg in diarized.get("segments", [])
    ]
    text = _build_annotated_text(segments)
    return {"text": text, "segments": segments}


async def do_transcribe(  # noqa: PLR0913
    store: VideoStore,
    video_id: str,
    whisper_model: str = "turbo",
    diarize_speakers: bool = True,
    align_language: str = "auto",
    *,
    ctx: Context | None = None,
) -> dict:
    logger.info(
        "Transcribing video_id=%s model=%s diarize=%s lang=%s",
        video_id,
        whisper_model,
        diarize_speakers,
        align_language,
    )
    record = store.get(video_id)
    video_dir_path = record.file_path.parent
    file_path = str(record.file_path)
    loop = asyncio.get_running_loop()

    # Stage 1: Transcription
    if ctx:
        await ctx.report_progress(0, 4, "Transcribing audio...")
    result, audio = await loop.run_in_executor(
        None, partial(_transcribe_stage, file_path, video_dir_path, whisper_model)
    )

    pipeline = _get_diarization_pipeline() if diarize_speakers else None
    if pipeline is None:
        if ctx:
            await ctx.report_progress(4, 4, "Transcription complete")
        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result.get("segments", [])
        ]
        text = "".join(seg["text"] for seg in segments)
        return {"text": text, "segments": segments}

    # Stage 2: Alignment
    if ctx:
        await ctx.report_progress(1, 4, "Aligning transcript...")
    aligned, audio = await loop.run_in_executor(
        None, partial(_align_stage, file_path, video_dir_path, whisper_model, result, audio, align_language)
    )

    # Stage 3: Diarization
    if ctx:
        await ctx.report_progress(2, 4, "Identifying speakers...")
    diarize_segments, audio = await loop.run_in_executor(
        None, partial(_diarize_stage, file_path, video_dir_path, whisper_model, pipeline, audio)
    )

    # Stage 4: Speaker assignment
    if ctx:
        await ctx.report_progress(3, 4, "Assigning speakers to segments...")
    final = await loop.run_in_executor(None, partial(_assign_speakers_stage, diarize_segments, aligned))

    if ctx:
        await ctx.report_progress(4, 4, "Transcription complete")
    logger.debug("Transcription complete video_id=%s length=%d chars", video_id, len(final["text"]))
    return final
