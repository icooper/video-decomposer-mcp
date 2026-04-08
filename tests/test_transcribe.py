import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from video_decomposer_mcp.tools.transcribe import (
    _align_cache,
    _align_stage,
    _assign_speakers_stage,
    _build_annotated_text,
    _diarize_cache,
    _diarize_stage,
    _get_diarization_pipeline,
    _NumpyEncoder,
    _read_cache,
    _transcribe_stage,
    _whisper_cache,
    _write_cache,
    do_transcribe,
    preload_align_model,
    preload_diarization_pipeline,
    preload_whisper_model,
)


@pytest.fixture(autouse=True)
def clear_caches():
    _whisper_cache.clear()
    _align_cache.clear()
    _diarize_cache.clear()
    yield
    _whisper_cache.clear()
    _align_cache.clear()
    _diarize_cache.clear()


TRANSCRIBE_MODULE = "video_decomposer_mcp.tools.transcribe"


# --- WhisperX model loading ---


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_transcribe_stage_cuda(mock_torch, mock_whisperx, tmp_path):
    mock_torch.cuda.is_available.return_value = True
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 2.5, "text": " Hello world"}],
    }
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()

    result, audio = _transcribe_stage("/fake/path.mp4", tmp_path, "turbo")

    assert result["segments"][0]["text"] == " Hello world"
    assert audio is not None
    mock_whisperx.load_model.assert_called_once_with("turbo", device="cuda", compute_type="float16")
    mock_whisperx.load_audio.assert_called_once_with("/fake/path.mp4")


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_transcribe_stage_cpu(mock_torch, mock_whisperx, tmp_path):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": " No GPU"}],
    }
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()

    result, audio = _transcribe_stage("/fake/path.mp4", tmp_path, "base")

    assert result["segments"][0]["text"] == " No GPU"
    mock_whisperx.load_model.assert_called_once_with("base", device="cpu", compute_type="int8")


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_whisper_model_caching(mock_torch, mock_whisperx, tmp_path):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"language": "en", "segments": []}
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()

    dir1 = tmp_path / "vid1"
    dir1.mkdir()
    dir2 = tmp_path / "vid2"
    dir2.mkdir()
    _transcribe_stage("/fake/1.mp4", dir1, "turbo")
    _transcribe_stage("/fake/2.mp4", dir2, "turbo")

    mock_whisperx.load_model.assert_called_once()


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_preload_whisper_model(mock_torch, mock_whisperx):
    mock_torch.cuda.is_available.return_value = False
    mock_whisperx.load_model.return_value = MagicMock()

    preload_whisper_model("base")

    mock_whisperx.load_model.assert_called_once_with("base", device="cpu", compute_type="int8")
    assert "base" in _whisper_cache


# --- Alignment model loading ---


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_preload_align_model(mock_torch, mock_whisperx):
    mock_torch.cuda.is_available.return_value = False
    mock_model_a = MagicMock()
    mock_metadata = MagicMock()
    mock_whisperx.load_align_model.return_value = (mock_model_a, mock_metadata)

    preload_align_model("en")

    mock_whisperx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
    assert "en" in _align_cache


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_align_model_caching(mock_torch, mock_whisperx):
    mock_torch.cuda.is_available.return_value = False
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())

    preload_align_model("en")
    preload_align_model("en")

    mock_whisperx.load_align_model.assert_called_once()


# --- Diarization pipeline loading ---


@patch(f"{TRANSCRIBE_MODULE}.DiarizationPipeline")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_preload_diarization_pipeline(mock_torch, mock_pipeline_cls, monkeypatch):
    mock_torch.cuda.is_available.return_value = False
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    mock_pipeline_cls.return_value = MagicMock()

    preload_diarization_pipeline()

    mock_pipeline_cls.assert_called_once_with(token="hf_test_token", device="cpu")
    assert "pipeline" in _diarize_cache


@patch(f"{TRANSCRIBE_MODULE}.DiarizationPipeline")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_preload_diarization_pipeline_caching(mock_torch, mock_pipeline_cls, monkeypatch):
    mock_torch.cuda.is_available.return_value = False
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    mock_pipeline_cls.return_value = MagicMock()

    preload_diarization_pipeline()
    preload_diarization_pipeline()

    mock_pipeline_cls.assert_called_once()


def test_preload_diarization_pipeline_missing_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # Should not raise, just log a warning and return
    preload_diarization_pipeline()

    assert "pipeline" not in _diarize_cache


@patch(f"{TRANSCRIBE_MODULE}.DiarizationPipeline")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_get_diarization_pipeline_missing_token_raises(mock_torch, mock_pipeline_cls, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mock_torch.cuda.is_available.return_value = False

    assert _get_diarization_pipeline() is None


# --- Full diarization pipeline ---


@patch(f"{TRANSCRIBE_MODULE}.DiarizationPipeline")
@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_all_stages_with_diarization(mock_torch, mock_whisperx, mock_pipeline_cls, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    mock_torch.cuda.is_available.return_value = False

    # Mock transcription
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello everyone"},
            {"start": 2.8, "end": 5.1, "text": " Thanks for having me"},
        ],
    }
    mock_whisperx.load_model.return_value = mock_model
    mock_audio = MagicMock()
    mock_whisperx.load_audio.return_value = mock_audio

    # Mock alignment
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx.align.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello everyone", "words": []},
            {"start": 2.8, "end": 5.1, "text": " Thanks for having me", "words": []},
        ],
    }

    # Mock diarization
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_diarize_df = pd.DataFrame(
        [
            {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
            {"start": 2.8, "end": 5.1, "speaker": "SPEAKER_01"},
        ]
    )
    mock_pipeline.return_value = mock_diarize_df

    # Mock speaker assignment
    mock_whisperx.assign_word_speakers.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello everyone", "speaker": "SPEAKER_00"},
            {"start": 2.8, "end": 5.1, "text": " Thanks for having me", "speaker": "SPEAKER_01"},
        ],
    }

    # Run all stages
    result, audio = _transcribe_stage("/fake/path.mp4", tmp_path, "turbo")
    aligned, audio = _align_stage("/fake/path.mp4", tmp_path, "turbo", result, audio, "auto")
    pipeline = _get_diarization_pipeline()
    diarize_segments, audio = _diarize_stage("/fake/path.mp4", tmp_path, "turbo", pipeline, audio)
    final = _assign_speakers_stage(diarize_segments, aligned)

    # Verify pipeline stages called in order
    mock_whisperx.load_audio.assert_called_once_with("/fake/path.mp4")
    mock_model.transcribe.assert_called_once_with(mock_audio)
    mock_whisperx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
    mock_whisperx.align.assert_called_once()
    mock_pipeline.assert_called_once_with(mock_audio)
    mock_whisperx.assign_word_speakers.assert_called_once()

    # Verify output format
    assert len(final["segments"]) == 2
    assert final["segments"][0]["speaker"] == "SPEAKER_00"
    assert final["segments"][1]["speaker"] == "SPEAKER_01"
    assert "SPEAKER_00:" in final["text"]
    assert "SPEAKER_01:" in final["text"]


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
def test_assign_speakers_stage_defaults_unknown_speaker(mock_whisperx):
    # assign_word_speakers returns segment without speaker key
    mock_whisperx.assign_word_speakers.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": " Some text"}],
    }

    diarize_segments = pd.DataFrame([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}])
    aligned = {"segments": [{"start": 0.0, "end": 2.0, "text": " Some text"}]}

    result = _assign_speakers_stage(diarize_segments, aligned)

    assert result["segments"][0]["speaker"] == "UNKNOWN"
    assert "UNKNOWN:" in result["text"]


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_align_stage_with_explicit_language(mock_torch, mock_whisperx, tmp_path):
    mock_torch.cuda.is_available.return_value = False
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx.align.return_value = {"segments": [{"start": 0.0, "end": 2.0, "text": " Bonjour"}]}

    result = {"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": " Bonjour"}]}
    audio = MagicMock()

    _align_stage("/fake/path.mp4", tmp_path, "turbo", result, audio, "fr")

    # Should use the explicit language "fr" instead of auto-detected "en"
    mock_whisperx.load_align_model.assert_called_once_with(language_code="fr", device="cpu")


# --- Annotated text building ---


def test_build_annotated_text():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Hello everyone", "speaker": "SPEAKER_00"},
        {"start": 2.8, "end": 5.1, "text": " Thanks for having me", "speaker": "SPEAKER_01"},
    ]
    text = _build_annotated_text(segments)
    assert text == "SPEAKER_00: Hello everyone\nSPEAKER_01: Thanks for having me"


def test_build_annotated_text_consolidates_same_speaker():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Hello everyone", "speaker": "SPEAKER_00"},
        {"start": 2.8, "end": 5.1, "text": " welcome to the show", "speaker": "SPEAKER_00"},
        {"start": 5.5, "end": 7.0, "text": " Thanks", "speaker": "SPEAKER_01"},
        {"start": 7.2, "end": 9.0, "text": " for having me", "speaker": "SPEAKER_01"},
        {"start": 9.5, "end": 11.0, "text": " Of course", "speaker": "SPEAKER_00"},
    ]
    text = _build_annotated_text(segments)
    assert text == (
        "SPEAKER_00: Hello everyone welcome to the show\nSPEAKER_01: Thanks for having me\nSPEAKER_00: Of course"
    )


def test_build_annotated_text_empty():
    assert _build_annotated_text([]) == ""


# --- Async wrapper ---


@patch(f"{TRANSCRIBE_MODULE}._get_diarization_pipeline")
@patch(f"{TRANSCRIBE_MODULE}._transcribe_stage")
async def test_do_transcribe_no_diarization(mock_stage, mock_get_pipeline, store_with_video):
    store, video_id, video_file = store_with_video
    mock_stage.return_value = (
        {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": " Hello"}]},
        None,
    )
    mock_get_pipeline.return_value = None

    result = await do_transcribe(store, video_id, "turbo", diarize_speakers=False)

    assert result["text"] == " Hello"
    assert len(result["segments"]) == 1
    mock_stage.assert_called_once()


@patch(f"{TRANSCRIBE_MODULE}._assign_speakers_stage")
@patch(f"{TRANSCRIBE_MODULE}._diarize_stage")
@patch(f"{TRANSCRIBE_MODULE}._align_stage")
@patch(f"{TRANSCRIBE_MODULE}._get_diarization_pipeline")
@patch(f"{TRANSCRIBE_MODULE}._transcribe_stage")
async def test_do_transcribe_with_diarization(  # noqa: PLR0913
    mock_transcribe_s, mock_get_pipeline, mock_align_s, mock_diarize_s, mock_assign_s, store_with_video
):
    store, video_id, _ = store_with_video
    mock_transcribe_s.return_value = ({"language": "en", "segments": []}, None)
    mock_get_pipeline.return_value = MagicMock()
    mock_align_s.return_value = ({"segments": []}, None)
    mock_diarize_s.return_value = (pd.DataFrame(), None)
    mock_assign_s.return_value = {"text": "SPEAKER_00: Hi", "segments": []}

    result = await do_transcribe(store, video_id, "turbo")

    assert result["text"] == "SPEAKER_00: Hi"
    mock_transcribe_s.assert_called_once()
    mock_align_s.assert_called_once()
    mock_diarize_s.assert_called_once()
    mock_assign_s.assert_called_once()


@patch(f"{TRANSCRIBE_MODULE}._assign_speakers_stage")
@patch(f"{TRANSCRIBE_MODULE}._diarize_stage")
@patch(f"{TRANSCRIBE_MODULE}._align_stage")
@patch(f"{TRANSCRIBE_MODULE}._get_diarization_pipeline")
@patch(f"{TRANSCRIBE_MODULE}._transcribe_stage")
async def test_do_transcribe_reports_progress(  # noqa: PLR0913
    mock_transcribe_s, mock_get_pipeline, mock_align_s, mock_diarize_s, mock_assign_s, store_with_video
):
    store, video_id, _ = store_with_video
    mock_transcribe_s.return_value = ({"language": "en", "segments": []}, None)
    mock_get_pipeline.return_value = MagicMock()
    mock_align_s.return_value = ({"segments": []}, None)
    mock_diarize_s.return_value = (pd.DataFrame(), None)
    mock_assign_s.return_value = {"text": "Hi", "segments": []}

    ctx = MagicMock()
    ctx.report_progress = AsyncMock()

    await do_transcribe(store, video_id, "turbo", ctx=ctx)

    # Verify progress was reported for each stage
    calls = ctx.report_progress.call_args_list
    assert len(calls) == 5
    assert calls[0].args == (0, 4, "Transcribing audio...")
    assert calls[1].args == (1, 4, "Aligning transcript...")
    assert calls[2].args == (2, 4, "Identifying speakers...")
    assert calls[3].args == (3, 4, "Assigning speakers to segments...")
    assert calls[4].args == (4, 4, "Transcription complete")


@patch(f"{TRANSCRIBE_MODULE}._get_diarization_pipeline")
@patch(f"{TRANSCRIBE_MODULE}._transcribe_stage")
async def test_do_transcribe_no_diarization_reports_progress(mock_stage, mock_get_pipeline, store_with_video):
    store, video_id, _ = store_with_video
    mock_stage.return_value = ({"language": "en", "segments": []}, None)
    mock_get_pipeline.return_value = None

    ctx = MagicMock()
    ctx.report_progress = AsyncMock()

    await do_transcribe(store, video_id, "turbo", diarize_speakers=False, ctx=ctx)

    calls = ctx.report_progress.call_args_list
    assert len(calls) == 2
    assert calls[0].args == (0, 4, "Transcribing audio...")
    assert calls[1].args == (4, 4, "Transcription complete")


@patch(f"{TRANSCRIBE_MODULE}._get_diarization_pipeline")
@patch(f"{TRANSCRIBE_MODULE}._transcribe_stage")
async def test_do_transcribe_works_without_ctx(mock_stage, mock_get_pipeline, store_with_video):
    store, video_id, _ = store_with_video
    mock_stage.return_value = ({"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": " Hi"}]}, None)
    mock_get_pipeline.return_value = None

    # No ctx passed — should not raise
    result = await do_transcribe(store, video_id, "turbo", diarize_speakers=False)
    assert result["text"] == " Hi"


# --- NumpyEncoder ---


def test_numpy_encoder_float():
    import numpy as np

    data = {"score": np.float32(0.95), "val": np.float64(1.5)}
    result = json.loads(json.dumps(data, cls=_NumpyEncoder))
    assert result["score"] == pytest.approx(0.95, abs=1e-5)
    assert result["val"] == 1.5


def test_numpy_encoder_int():
    import numpy as np

    data = {"count": np.int64(42)}
    result = json.loads(json.dumps(data, cls=_NumpyEncoder))
    assert result["count"] == 42


def test_numpy_encoder_array():
    import numpy as np

    data = {"values": np.array([1.0, 2.0, 3.0])}
    result = json.loads(json.dumps(data, cls=_NumpyEncoder))
    assert result["values"] == [1.0, 2.0, 3.0]


def test_numpy_encoder_unsupported_type():
    with pytest.raises(TypeError):
        json.dumps({"bad": object()}, cls=_NumpyEncoder)


# --- Cache read/write helpers ---


def test_read_cache_missing(tmp_path):
    assert _read_cache(tmp_path / "nonexistent.json") is None


def test_read_cache_corrupt_json(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{bad json content")
    assert _read_cache(path) is None


def test_write_and_read_cache(tmp_path):
    path = tmp_path / "test_cache.json"
    data = {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}]}
    _write_cache(path, data)
    loaded = _read_cache(path)
    assert loaded == data


def test_write_cache_atomic(tmp_path):
    """Verify no .tmp files are left behind after successful write."""
    path = tmp_path / "test_cache.json"
    _write_cache(path, {"key": "value"})
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


# --- Transcription cache integration ---


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_transcription_cache_hit_skips_model(mock_torch, mock_whisperx, tmp_path):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": " Cached"}],
    }
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()

    # First call: populates cache
    result1, audio1 = _transcribe_stage("/fake/path.mp4", tmp_path, "turbo")
    assert mock_model.transcribe.call_count == 1
    assert mock_whisperx.load_audio.call_count == 1
    assert audio1 is not None

    # Second call: should use cache, skip transcribe and load_audio
    result2, audio2 = _transcribe_stage("/fake/path.mp4", tmp_path, "turbo")
    assert mock_model.transcribe.call_count == 1  # not called again
    assert mock_whisperx.load_audio.call_count == 1  # not called again
    assert audio2 is None  # no audio loaded on cache hit
    assert result1 == result2

    # Verify cache file exists
    assert (tmp_path / "transcription_cache_turbo.json").exists()


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_alignment_cache_hit_skips_align(mock_torch, mock_whisperx, tmp_path):
    mock_torch.cuda.is_available.return_value = False
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx.align.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": " Hello", "words": []}],
    }

    result = {"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": " Hello"}]}
    audio = MagicMock()

    # First call: populates cache
    _align_stage("/fake/path.mp4", tmp_path, "turbo", result, audio, "auto")
    assert mock_whisperx.align.call_count == 1

    # Second call: should use cache
    _align_stage("/fake/path.mp4", tmp_path, "turbo", result, audio, "auto")
    assert mock_whisperx.align.call_count == 1  # not called again

    # Verify cache file exists
    assert (tmp_path / "alignment_cache_turbo_en.json").exists()


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
def test_diarize_stage_cache_hit(mock_whisperx, tmp_path):
    """When diarization cache exists, audio is not loaded and pipeline is not called."""
    _write_cache(
        tmp_path / "diarization_cache_turbo.json",
        [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        ],
    )

    pipeline = MagicMock()
    diarize_segments, audio = _diarize_stage("/fake/path.mp4", tmp_path, "turbo", pipeline, None)

    pipeline.assert_not_called()
    mock_whisperx.load_audio.assert_not_called()
    assert len(diarize_segments) == 1
    assert diarize_segments.iloc[0]["speaker"] == "SPEAKER_00"


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
def test_diarize_stage_loads_audio_when_needed(mock_whisperx, tmp_path):
    """When no cache and audio is None, audio is loaded."""
    mock_whisperx.load_audio.return_value = MagicMock()
    pipeline = MagicMock()
    pipeline.return_value = pd.DataFrame([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}])

    _diarize_stage("/fake/path.mp4", tmp_path, "turbo", pipeline, None)

    mock_whisperx.load_audio.assert_called_once_with("/fake/path.mp4")
    pipeline.assert_called_once()
    assert (tmp_path / "diarization_cache_turbo.json").exists()


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_align_stage_loads_audio_when_needed(mock_torch, mock_whisperx, tmp_path):
    """When audio is None (transcription was cached), align_stage loads audio."""
    mock_torch.cuda.is_available.return_value = False
    mock_whisperx.load_audio.return_value = MagicMock()
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx.align.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": " Hello", "words": []}],
    }

    result = {"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": " Hello"}]}

    _align_stage("/fake/path.mp4", tmp_path, "turbo", result, None, "auto")

    mock_whisperx.load_audio.assert_called_once_with("/fake/path.mp4")
    mock_whisperx.align.assert_called_once()
    assert (tmp_path / "alignment_cache_turbo_en.json").exists()


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_auto_language_resolves_in_alignment_cache_filename(mock_torch, mock_whisperx, tmp_path):
    """When align_language='auto', the resolved language appears in the alignment cache filename."""
    mock_torch.cuda.is_available.return_value = False
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx.align.return_value = {"segments": [{"start": 0.0, "end": 1.0, "text": " Bonjour"}]}

    result = {"language": "fr", "segments": [{"start": 0.0, "end": 1.0, "text": " Bonjour"}]}
    audio = MagicMock()

    _align_stage("/fake/path.mp4", tmp_path, "turbo", result, audio, "auto")

    # Should use the auto-detected language "fr" in the cache filename
    assert (tmp_path / "alignment_cache_turbo_fr.json").exists()
    assert not (tmp_path / "alignment_cache_turbo_en.json").exists()


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_different_whisper_models_get_separate_caches(mock_torch, mock_whisperx, tmp_path):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"language": "en", "segments": []}
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()

    _transcribe_stage("/fake/path.mp4", tmp_path, "turbo")
    _transcribe_stage("/fake/path.mp4", tmp_path, "base")

    assert (tmp_path / "transcription_cache_turbo.json").exists()
    assert (tmp_path / "transcription_cache_base.json").exists()
    assert mock_model.transcribe.call_count == 2


def test_write_cache_cleans_up_on_error(tmp_path):
    """Verify temp file is cleaned up if json.dump fails."""
    path = tmp_path / "bad_cache.json"
    bad_data = {"bad": object()}  # not serializable even with _NumpyEncoder
    with pytest.raises(TypeError):
        _write_cache(path, bad_data)
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
