from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from video_decomposer_mcp.server import (
    _cleanup_loop,
    analyze_video,
    download_video,
    extract_frame,
    main,
    mcp,
    store,
    transcribe_video,
)


def test_server_has_tools():
    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert "download_video" in tool_names
    assert "transcribe_video" in tool_names
    assert "extract_frame" in tool_names
    assert "analyze_video" in tool_names


def test_server_config():
    assert mcp.settings.host == "127.0.0.1"
    assert mcp.settings.port == 8000
    assert mcp.settings.stateless_http is not True


@patch("video_decomposer_mcp.server.do_download", new_callable=AsyncMock)
async def test_download_video_tool(mock_do):
    mock_do.return_value = "abc123"
    result = await download_video("https://example.com/v")
    assert result == "abc123"
    mock_do.assert_called_once_with(store, "https://example.com/v")


@patch("video_decomposer_mcp.server.do_download", new_callable=AsyncMock)
async def test_download_video_tool_with_ctx(mock_do):
    mock_do.return_value = "abc123"
    ctx = MagicMock()
    ctx.info = AsyncMock()
    result = await download_video("https://example.com/v", ctx=ctx)
    assert result == "abc123"
    ctx.info.assert_called_once_with("Downloading video...")


@patch("video_decomposer_mcp.server.do_transcribe", new_callable=AsyncMock)
async def test_transcribe_video_tool(mock_do):
    mock_do.return_value = {"text": "Hello", "segments": []}
    result = await transcribe_video("vid1", "turbo")
    assert result["text"] == "Hello"
    mock_do.assert_called_once_with(store, "vid1", "turbo", False, "auto", ctx=None)


@patch("video_decomposer_mcp.server.do_transcribe", new_callable=AsyncMock)
async def test_transcribe_video_tool_with_diarization(mock_do):
    mock_do.return_value = {"text": "SPEAKER_00: Hello", "segments": []}
    result = await transcribe_video("vid1", "turbo", diarize_speakers=True)
    assert result["text"] == "SPEAKER_00: Hello"
    mock_do.assert_called_once_with(store, "vid1", "turbo", True, "auto", ctx=None)


@patch("video_decomposer_mcp.server.do_transcribe", new_callable=AsyncMock)
async def test_transcribe_video_tool_explicit_language(mock_do):
    mock_do.return_value = {"text": "Bonjour", "segments": []}
    result = await transcribe_video("vid1", "turbo", align_language="fr")
    assert result["text"] == "Bonjour"
    mock_do.assert_called_once_with(store, "vid1", "turbo", False, "fr", ctx=None)


@patch("video_decomposer_mcp.server.do_extract_frame", new_callable=AsyncMock)
async def test_extract_frame_tool(mock_do):
    from mcp.server.fastmcp import Image

    mock_do.return_value = Image(data=b"jpeg", format="jpeg")
    result = await extract_frame("vid1", 5.0)
    assert isinstance(result, Image)
    mock_do.assert_called_once_with(store, "vid1", 5.0, max_dimension=768, quality=75)


@patch("video_decomposer_mcp.server.do_extract_frame", new_callable=AsyncMock)
async def test_extract_frame_tool_custom_params(mock_do):
    from mcp.server.fastmcp import Image

    mock_do.return_value = Image(data=b"jpeg", format="jpeg")
    await extract_frame("vid1", 10.5, max_dimension=512, quality=60)
    mock_do.assert_called_once_with(store, "vid1", 10.5, max_dimension=512, quality=60)


@patch("video_decomposer_mcp.server.do_analyze", new_callable=AsyncMock)
async def test_analyze_video_tool(mock_do):
    mock_do.return_value = {
        "video_id": "v1",
        "transcript": {"text": "SPEAKER_00: Hi", "segments": []},
    }
    result = await analyze_video("https://example.com/v", "turbo")
    assert result["video_id"] == "v1"
    mock_do.assert_called_once_with(
        store,
        "https://example.com/v",
        "turbo",
        False,
        "auto",
        ctx=None,
    )


@patch("video_decomposer_mcp.server.do_analyze", new_callable=AsyncMock)
async def test_analyze_video_tool_custom_model(mock_do):
    mock_do.return_value = {"video_id": "v1", "transcript": {"text": "", "segments": []}}
    await analyze_video("https://example.com/v", "large")
    mock_do.assert_called_once_with(
        store,
        "https://example.com/v",
        "large",
        False,
        "auto",
        ctx=None,
    )


@patch("video_decomposer_mcp.server.do_analyze", new_callable=AsyncMock)
async def test_analyze_video_tool_with_diarization(mock_do):
    mock_do.return_value = {"video_id": "v1", "transcript": {"text": "SPEAKER_00: Hi", "segments": []}}
    await analyze_video("https://example.com/v", "turbo", diarize_speakers=True)
    mock_do.assert_called_once_with(
        store,
        "https://example.com/v",
        "turbo",
        True,
        "auto",
        ctx=None,
    )


@patch("video_decomposer_mcp.server.do_analyze", new_callable=AsyncMock)
async def test_analyze_video_tool_explicit_language(mock_do):
    mock_do.return_value = {"video_id": "v1", "transcript": {"text": "Bonjour", "segments": []}}
    await analyze_video("https://example.com/v", "turbo", align_language="fr")
    mock_do.assert_called_once_with(
        store,
        "https://example.com/v",
        "turbo",
        False,
        "fr",
        ctx=None,
    )


@patch("video_decomposer_mcp.server.mcp")
@patch("video_decomposer_mcp.server.threading.Thread")
@patch("video_decomposer_mcp.tools.transcribe.release_models")
@patch("video_decomposer_mcp.tools.transcribe.preload_diarization_pipeline")
@patch("video_decomposer_mcp.tools.transcribe.preload_align_model")
@patch("video_decomposer_mcp.tools.transcribe.preload_whisper_model")
def test_main(mock_preload_whisper, mock_preload_align, mock_preload_diarize, mock_release, mock_thread_cls, mock_mcp):  # noqa: PLR0913
    mock_thread = MagicMock()
    mock_thread_cls.return_value = mock_thread

    main()

    mock_preload_whisper.assert_called_once()
    mock_preload_align.assert_called_once()
    mock_release.assert_called_once()
    mock_thread_cls.assert_called_once_with(target=_cleanup_loop, daemon=True)
    mock_thread.start.assert_called_once()
    mock_mcp.run.assert_called_once_with(transport="sse")


@patch("video_decomposer_mcp.server.mcp")
@patch("video_decomposer_mcp.server.threading.Thread")
@patch("video_decomposer_mcp.tools.transcribe.release_models")
@patch("video_decomposer_mcp.tools.transcribe.preload_diarization_pipeline")
@patch("video_decomposer_mcp.tools.transcribe.preload_align_model")
@patch("video_decomposer_mcp.tools.transcribe.preload_whisper_model")
def test_main_preloads_diarization_when_hf_token_set(  # noqa: PLR0913
    mock_preload_whisper, mock_preload_align, mock_preload_diarize, mock_release, mock_thread_cls, mock_mcp
):
    mock_thread_cls.return_value = MagicMock()
    with patch("video_decomposer_mcp.server.hf_token", "test-token"):
        main()
    mock_preload_diarize.assert_called_once()


@patch("video_decomposer_mcp.server.store")
def test_cleanup_loop(mock_store):
    mock_store.cleanup.return_value = 2

    call_count = 0

    def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise SystemExit
        assert seconds == 600

    with patch("video_decomposer_mcp.server.time.sleep", side_effect=fake_sleep):
        with pytest.raises(SystemExit):
            _cleanup_loop()

    mock_store.cleanup.assert_called_once()


@patch("video_decomposer_mcp.server.store")
def test_cleanup_loop_handles_exception(mock_store):
    mock_store.cleanup.side_effect = RuntimeError("boom")

    call_count = 0

    def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise SystemExit

    with patch("video_decomposer_mcp.server.time.sleep", side_effect=fake_sleep):
        with pytest.raises(SystemExit):
            _cleanup_loop()

    mock_store.cleanup.assert_called_once()
