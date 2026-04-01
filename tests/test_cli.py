import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from video_decomposer_mcp.cli import app

runner = CliRunner()


@patch("video_decomposer_mcp.cli.do_download", new_callable=AsyncMock)
def test_download(mock_download):
    mock_download.return_value = "abc123def456"
    result = runner.invoke(app, ["download", "https://example.com/v"])
    assert result.exit_code == 0
    assert "abc123def456" in result.output


@patch("video_decomposer_mcp.cli.do_transcribe", new_callable=AsyncMock)
def test_transcribe(mock_transcribe):
    mock_transcribe.return_value = {"text": "Hello world transcript", "segments": []}
    result = runner.invoke(app, ["transcribe", "vid123", "--whisper-model", "base"])
    assert result.exit_code == 0
    assert "Hello world transcript" in result.output


@patch("video_decomposer_mcp.cli.do_extract_frame", new_callable=AsyncMock)
def test_extract_frame(mock_extract, tmp_path: Path):
    frame_data = base64.b64encode(b"fake jpeg").decode()
    mock_extract.return_value = {
        "type": "image",
        "data": frame_data,
        "mimeType": "image/jpeg",
        "timestamp": 5.0,
    }
    output_dir = tmp_path / "out"
    result = runner.invoke(app, ["extract-frame", "vid123", "5.0", "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert "Saved frame at 5.0s" in result.output
    assert (output_dir / "frame_00005000.jpg").read_bytes() == b"fake jpeg"


@patch("video_decomposer_mcp.cli.do_analyze", new_callable=AsyncMock)
def test_analyze(mock_analyze):
    mock_analyze.return_value = {
        "video_id": "vid999",
        "transcript": {"text": "Analyzed transcript", "segments": []},
    }
    result = runner.invoke(app, ["analyze", "https://example.com/v"])
    assert result.exit_code == 0
    assert "Video ID: vid999" in result.output
    assert "Analyzed transcript" in result.output


@patch("video_decomposer_mcp.cli.do_analyze", new_callable=AsyncMock)
def test_analyze_custom_model(mock_analyze):
    mock_analyze.return_value = {
        "video_id": "vid999",
        "transcript": {"text": "text", "segments": []},
    }
    result = runner.invoke(app, ["analyze", "https://example.com/v", "--whisper-model", "large"])
    assert result.exit_code == 0
    mock_analyze.assert_called_once()
    call_args = mock_analyze.call_args
    assert call_args[0][2] == "large"


@patch("video_decomposer_mcp.cli.preload_whisper_model")
def test_preload(mock_preload):
    result = runner.invoke(app, ["preload", "--whisper-model", "base"])
    assert result.exit_code == 0
    assert "base" in result.output
    mock_preload.assert_called_once_with("base")


@patch("video_decomposer_mcp.cli.app")
@patch("video_decomposer_mcp.cli.store")
def test_main(mock_store, mock_app):
    from video_decomposer_mcp.cli import main

    main()
    mock_store.cleanup.assert_called_once()
    mock_app.assert_called_once()
