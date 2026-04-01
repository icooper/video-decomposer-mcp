from unittest.mock import MagicMock, patch

import pytest

from video_decomposer_mcp.tools.transcribe import (
    _align_cache,
    _build_annotated_text,
    _diarize_cache,
    _get_diarization_pipeline,
    _transcribe,
    _whisper_cache,
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
def test_transcribe_without_diarization_cuda(mock_torch, mock_whisperx):
    mock_torch.cuda.is_available.return_value = True
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 2.5, "text": " Hello world"}],
    }
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()

    result = _transcribe("/fake/path.mp4", "turbo", diarize_speakers=False)

    assert result["text"] == " Hello world"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 0.0
    assert "speaker" not in result["segments"][0]
    mock_whisperx.load_model.assert_called_once_with("turbo", device="cuda", compute_type="float16")
    mock_whisperx.load_audio.assert_called_once_with("/fake/path.mp4")


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_transcribe_without_diarization_cpu(mock_torch, mock_whisperx):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": " No GPU"}],
    }
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()

    result = _transcribe("/fake/path.mp4", "base", diarize_speakers=False)

    assert result["text"] == " No GPU"
    mock_whisperx.load_model.assert_called_once_with("base", device="cpu", compute_type="int8")


@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_whisper_model_caching(mock_torch, mock_whisperx):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"language": "en", "segments": []}
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()

    _transcribe("/fake/1.mp4", "turbo", diarize_speakers=False)
    _transcribe("/fake/2.mp4", "turbo", diarize_speakers=False)

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
def test_transcribe_with_diarization(mock_torch, mock_whisperx, mock_pipeline_cls, monkeypatch):
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
    mock_diarize_result = MagicMock()
    mock_pipeline.return_value = mock_diarize_result

    # Mock speaker assignment
    mock_whisperx.assign_word_speakers.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello everyone", "speaker": "SPEAKER_00"},
            {"start": 2.8, "end": 5.1, "text": " Thanks for having me", "speaker": "SPEAKER_01"},
        ],
    }

    result = _transcribe("/fake/path.mp4", "turbo", diarize_speakers=True)

    # Verify pipeline stages called in order
    mock_whisperx.load_audio.assert_called_once_with("/fake/path.mp4")
    mock_model.transcribe.assert_called_once_with(mock_audio)
    mock_whisperx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
    mock_whisperx.align.assert_called_once()
    mock_pipeline.assert_called_once_with(mock_audio)
    mock_whisperx.assign_word_speakers.assert_called_once_with(mock_diarize_result, mock_whisperx.align.return_value)

    # Verify output format
    assert len(result["segments"]) == 2
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][1]["speaker"] == "SPEAKER_01"
    assert "SPEAKER_00:" in result["text"]
    assert "SPEAKER_01:" in result["text"]


@patch(f"{TRANSCRIBE_MODULE}.DiarizationPipeline")
@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_transcribe_diarization_defaults_unknown_speaker(mock_torch, mock_whisperx, mock_pipeline_cls, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    mock_torch.cuda.is_available.return_value = False

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 2.0, "text": " Some text"}],
    }
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx.align.return_value = {"segments": [{"start": 0.0, "end": 2.0, "text": " Some text"}]}
    mock_pipeline_cls.return_value = MagicMock()

    # assign_word_speakers returns segment without speaker key
    mock_whisperx.assign_word_speakers.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": " Some text"}],
    }

    result = _transcribe("/fake/path.mp4", "turbo", diarize_speakers=True)

    assert result["segments"][0]["speaker"] == "UNKNOWN"
    assert "UNKNOWN:" in result["text"]


@patch(f"{TRANSCRIBE_MODULE}.DiarizationPipeline")
@patch(f"{TRANSCRIBE_MODULE}.whisperx")
@patch(f"{TRANSCRIBE_MODULE}.torch")
def test_transcribe_with_explicit_align_language(mock_torch, mock_whisperx, mock_pipeline_cls, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    mock_torch.cuda.is_available.return_value = False

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 2.0, "text": " Bonjour"}],
    }
    mock_whisperx.load_model.return_value = mock_model
    mock_whisperx.load_audio.return_value = MagicMock()
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx.align.return_value = {"segments": [{"start": 0.0, "end": 2.0, "text": " Bonjour"}]}
    mock_pipeline_cls.return_value = MagicMock()
    mock_whisperx.assign_word_speakers.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": " Bonjour", "speaker": "SPEAKER_00"}],
    }

    _transcribe("/fake/path.mp4", "turbo", diarize_speakers=True, align_language="fr")

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


@patch(f"{TRANSCRIBE_MODULE}._transcribe")
async def test_do_transcribe(mock_transcribe, store_with_video):
    store, video_id, _ = store_with_video
    mock_transcribe.return_value = {"text": "Transcribed text", "segments": []}

    result = await do_transcribe(store, video_id, "turbo")

    assert result["text"] == "Transcribed text"
    mock_transcribe.assert_called_once()
    # Verify diarize_speakers=True is the default
    call_args = mock_transcribe.call_args
    assert call_args[0][2] is True  # diarize_speakers positional arg
    assert call_args[0][3] == "auto"  # align_language positional arg


@patch(f"{TRANSCRIBE_MODULE}._transcribe")
async def test_do_transcribe_passes_diarize_false(mock_transcribe, store_with_video):
    store, video_id, _ = store_with_video
    mock_transcribe.return_value = {"text": "Text", "segments": []}

    await do_transcribe(store, video_id, "turbo", diarize_speakers=False)

    call_args = mock_transcribe.call_args
    assert call_args[0][2] is False


@patch(f"{TRANSCRIBE_MODULE}._transcribe")
async def test_do_transcribe_passes_align_language(mock_transcribe, store_with_video):
    store, video_id, _ = store_with_video
    mock_transcribe.return_value = {"text": "Text", "segments": []}

    await do_transcribe(store, video_id, "turbo", align_language="fr")

    call_args = mock_transcribe.call_args
    assert call_args[0][3] == "fr"
