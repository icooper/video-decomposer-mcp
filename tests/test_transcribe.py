from unittest.mock import MagicMock, patch

import pytest

from video_decomposer_mcp.tools.transcribe import _model_cache, _transcribe, do_transcribe, preload_model


@pytest.fixture(autouse=True)
def clear_model_cache():
    _model_cache.clear()
    yield
    _model_cache.clear()


@patch("video_decomposer_mcp.tools.transcribe.whisper")
@patch("video_decomposer_mcp.tools.transcribe.torch")
def test_transcribe_with_cuda(mock_torch, mock_whisper):
    mock_torch.cuda.is_available.return_value = True
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "Hello world",
        "segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}],
    }
    mock_whisper.load_model.return_value = mock_model

    result = _transcribe("/fake/path.mp4", "turbo")

    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 0.0
    mock_whisper.load_model.assert_called_once_with("turbo", device="cuda")
    mock_model.transcribe.assert_called_once_with("/fake/path.mp4")


@patch("video_decomposer_mcp.tools.transcribe.whisper")
@patch("video_decomposer_mcp.tools.transcribe.torch")
def test_transcribe_without_cuda(mock_torch, mock_whisper):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "No GPU",
        "segments": [{"start": 0.0, "end": 1.0, "text": "No GPU"}],
    }
    mock_whisper.load_model.return_value = mock_model

    result = _transcribe("/fake/path.mp4", "base")

    assert result["text"] == "No GPU"
    mock_whisper.load_model.assert_called_once_with("base", device="cpu")


@patch("video_decomposer_mcp.tools.transcribe.whisper")
@patch("video_decomposer_mcp.tools.transcribe.torch")
def test_model_caching(mock_torch, mock_whisper):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "cached", "segments": []}
    mock_whisper.load_model.return_value = mock_model

    _transcribe("/fake/1.mp4", "turbo")
    _transcribe("/fake/2.mp4", "turbo")

    # Should only load once
    mock_whisper.load_model.assert_called_once()


@patch("video_decomposer_mcp.tools.transcribe.whisper")
@patch("video_decomposer_mcp.tools.transcribe.torch")
def test_preload_model(mock_torch, mock_whisper):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    preload_model("base")

    mock_whisper.load_model.assert_called_once_with("base", device="cpu")
    assert "base" in _model_cache


@patch("video_decomposer_mcp.tools.transcribe._transcribe")
async def test_do_transcribe(mock_transcribe, store_with_video):
    store, video_id, _ = store_with_video
    mock_transcribe.return_value = {"text": "Transcribed text", "segments": []}

    result = await do_transcribe(store, video_id, "turbo")

    assert result["text"] == "Transcribed text"
    mock_transcribe.assert_called_once()
