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
    mock_do.assert_called_once_with(store, "vid1", "turbo")


@patch("video_decomposer_mcp.server.do_extract_frame", new_callable=AsyncMock)
async def test_extract_frame_tool(mock_do):
    mock_do.return_value = {"type": "image", "data": "abc", "mimeType": "image/jpeg", "timestamp": 5.0}
    result = await extract_frame("vid1", 5.0)
    assert result["timestamp"] == 5.0
    mock_do.assert_called_once_with(store, "vid1", 5.0, max_dimension=768, quality=75)


@patch("video_decomposer_mcp.server.do_extract_frame", new_callable=AsyncMock)
async def test_extract_frame_tool_custom_params(mock_do):
    mock_do.return_value = {"type": "image", "data": "abc", "mimeType": "image/jpeg", "timestamp": 10.5}
    await extract_frame("vid1", 10.5, max_dimension=512, quality=60)
    mock_do.assert_called_once_with(store, "vid1", 10.5, max_dimension=512, quality=60)


@patch("video_decomposer_mcp.server.do_analyze", new_callable=AsyncMock)
async def test_analyze_video_tool(mock_do):
    mock_do.return_value = {
        "video_id": "v1",
        "transcript": {"text": "Hi", "segments": []},
    }
    result = await analyze_video("https://example.com/v", "turbo")
    assert result["video_id"] == "v1"
    mock_do.assert_called_once_with(
        store,
        "https://example.com/v",
        "turbo",
    )


@patch("video_decomposer_mcp.server.do_analyze", new_callable=AsyncMock)
async def test_analyze_video_tool_custom_model(mock_do):
    mock_do.return_value = {"video_id": "v1", "transcript": {"text": "", "segments": []}}
    await analyze_video("https://example.com/v", "large")
    mock_do.assert_called_once_with(
        store,
        "https://example.com/v",
        "large",
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

    with patch("video_decomposer_mcp.server.preload_model"):
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
