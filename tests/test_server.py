import asyncio
from unittest.mock import AsyncMock, patch

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


@patch("video_decomposer_mcp.server.do_transcribe", new_callable=AsyncMock)
async def test_transcribe_video_tool(mock_do):
    mock_do.return_value = {"text": "Hello", "segments": []}
    result = await transcribe_video("vid1", "turbo")
    assert result["text"] == "Hello"
    mock_do.assert_called_once_with(store, "vid1", "turbo", False, "auto")


@patch("video_decomposer_mcp.server.do_transcribe", new_callable=AsyncMock)
async def test_transcribe_video_tool_with_diarization(mock_do):
    mock_do.return_value = {"text": "SPEAKER_00: Hello", "segments": []}
    result = await transcribe_video("vid1", "turbo", diarize_speakers=True)
    assert result["text"] == "SPEAKER_00: Hello"
    mock_do.assert_called_once_with(store, "vid1", "turbo", True, "auto")


@patch("video_decomposer_mcp.server.do_transcribe", new_callable=AsyncMock)
async def test_transcribe_video_tool_explicit_language(mock_do):
    mock_do.return_value = {"text": "Bonjour", "segments": []}
    result = await transcribe_video("vid1", "turbo", align_language="fr")
    assert result["text"] == "Bonjour"
    mock_do.assert_called_once_with(store, "vid1", "turbo", False, "fr")


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
    )


@patch("video_decomposer_mcp.server.mcp")
def test_main(mock_mcp):
    main()
    mock_mcp.run.assert_called_once_with(transport="sse")


def test_server_has_lifespan():
    assert mcp.settings.lifespan is not None


@patch("video_decomposer_mcp.server.store")
async def test_cleanup_loop(mock_store):
    mock_store.async_cleanup = AsyncMock(return_value=2)

    call_count = 0

    async def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError
        assert seconds == 600

    with patch("video_decomposer_mcp.server.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await _cleanup_loop()

    mock_store.async_cleanup.assert_called_once()


@patch("video_decomposer_mcp.server.store")
async def test_cleanup_loop_handles_exception(mock_store):
    mock_store.async_cleanup = AsyncMock(side_effect=RuntimeError("boom"))

    call_count = 0

    async def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError

    with patch("video_decomposer_mcp.server.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await _cleanup_loop()

    mock_store.async_cleanup.assert_called_once()


async def test_lifespan():
    from video_decomposer_mcp.server import lifespan

    created_tasks = []
    original_create_task = asyncio.create_task

    def capture_create_task(*args, **kwargs):
        task = original_create_task(*args, **kwargs)
        created_tasks.append(task)
        return task

    with patch("video_decomposer_mcp.server.preload_whisper_model"):
        with patch("video_decomposer_mcp.server.preload_align_model"):
            with patch("video_decomposer_mcp.server.preload_diarization_pipeline"):
                with patch("video_decomposer_mcp.server._cleanup_loop", new_callable=AsyncMock):
                    with patch(
                        "video_decomposer_mcp.server.asyncio.create_task", side_effect=capture_create_task
                    ) as mock_create:
                        async with lifespan(mcp):
                            mock_create.assert_called_once()
                            assert len(created_tasks) == 1
                            assert not created_tasks[0].cancelled()
                        # After exiting lifespan, the task should be cancelled
                        assert created_tasks[0].cancelled()


@pytest.mark.asyncio
async def test_lifespan_preloads_diarization_when_hf_token_set():
    from video_decomposer_mcp.server import lifespan

    with patch("video_decomposer_mcp.server.preload_whisper_model"):
        with patch("video_decomposer_mcp.server.preload_align_model"):
            with patch("video_decomposer_mcp.server.preload_diarization_pipeline") as mock_diarize:
                with patch("video_decomposer_mcp.server._cleanup_loop", new_callable=AsyncMock):
                    with patch("video_decomposer_mcp.server.hf_token", "test-token"):
                        async with lifespan(mcp):
                            mock_diarize.assert_called_once()
