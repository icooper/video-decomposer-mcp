from unittest.mock import AsyncMock, patch

from video_decomposer_mcp.tools.analyze import do_analyze


@patch("video_decomposer_mcp.tools.analyze.do_transcribe", new_callable=AsyncMock)
@patch("video_decomposer_mcp.tools.analyze.do_download", new_callable=AsyncMock)
async def test_do_analyze(mock_download, mock_transcribe, store_with_video):
    store, _, _ = store_with_video
    mock_download.return_value = "vid123"
    mock_transcribe.return_value = {"text": "Hello world", "segments": []}

    result = await do_analyze(store, "https://example.com/v", "turbo")

    assert result["video_id"] == "vid123"
    assert result["transcript"]["text"] == "Hello world"
    assert "frames" not in result

    mock_download.assert_called_once_with(store, "https://example.com/v")
    mock_transcribe.assert_called_once_with(store, "vid123", "turbo")


@patch("video_decomposer_mcp.tools.analyze.do_transcribe", new_callable=AsyncMock)
@patch("video_decomposer_mcp.tools.analyze.do_download", new_callable=AsyncMock)
async def test_do_analyze_custom_model(mock_download, mock_transcribe, store_with_video):
    store, _, _ = store_with_video
    mock_download.return_value = "vid123"
    mock_transcribe.return_value = {"text": "Hello", "segments": []}

    await do_analyze(store, "https://example.com/v", "large")

    mock_transcribe.assert_called_once_with(store, "vid123", "large")
