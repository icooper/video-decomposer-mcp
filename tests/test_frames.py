from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mcp.server.fastmcp import Image

from video_decomposer_mcp.tools.frames import (
    _extract_frame_at,
    do_extract_frame,
)


@patch("video_decomposer_mcp.tools.frames.cv2")
@patch("video_decomposer_mcp.tools.frames.av")
def test_extract_frame_at(mock_av, mock_cv2):
    mock_frame = MagicMock()
    mock_frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_stream = MagicMock()
    mock_stream.time_base = 1 / 1000

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.decode.return_value = iter([mock_frame])
    mock_av.open.return_value.__enter__ = MagicMock(return_value=mock_container)
    mock_av.open.return_value.__exit__ = MagicMock(return_value=False)

    mock_cv2.IMWRITE_JPEG_QUALITY = 1
    mock_cv2.imencode.return_value = (True, np.frombuffer(b"jpeg bytes", dtype=np.uint8))

    result = _extract_frame_at("/fake/video.mp4", 5.0, 768, 75)

    assert result == b"jpeg bytes"
    mock_av.open.assert_called_once_with("/fake/video.mp4")
    mock_container.seek.assert_called_once()
    mock_frame.to_ndarray.assert_called_once_with(format="bgr24")
    # No resize needed since 640 < 768
    mock_cv2.resize.assert_not_called()


@patch("video_decomposer_mcp.tools.frames.cv2")
@patch("video_decomposer_mcp.tools.frames.av")
def test_extract_frame_at_resizes_large_frame(mock_av, mock_cv2):
    mock_frame = MagicMock()
    mock_frame.to_ndarray.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)

    mock_stream = MagicMock()
    mock_stream.time_base = 1 / 1000

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.decode.return_value = iter([mock_frame])
    mock_av.open.return_value.__enter__ = MagicMock(return_value=mock_container)
    mock_av.open.return_value.__exit__ = MagicMock(return_value=False)

    resized = np.zeros((432, 768, 3), dtype=np.uint8)
    mock_cv2.resize.return_value = resized
    mock_cv2.INTER_AREA = 3
    mock_cv2.IMWRITE_JPEG_QUALITY = 1
    mock_cv2.imencode.return_value = (True, np.frombuffer(b"resized", dtype=np.uint8))

    result = _extract_frame_at("/fake/video.mp4", 10.0, 768, 75)

    assert result == b"resized"
    mock_cv2.resize.assert_called_once()


@patch("video_decomposer_mcp.tools.frames.av")
def test_extract_frame_at_no_time_base(mock_av):
    mock_stream = MagicMock()
    mock_stream.time_base = None

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_av.open.return_value.__enter__ = MagicMock(return_value=mock_container)
    mock_av.open.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="Video stream has no time_base"):
        _extract_frame_at("/fake/video.mp4", 5.0, 768, 75)


@patch("video_decomposer_mcp.tools.frames.av")
def test_extract_frame_at_no_frames(mock_av):
    mock_stream = MagicMock()
    mock_stream.time_base = 1 / 1000

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.decode.return_value = iter([])
    mock_av.open.return_value.__enter__ = MagicMock(return_value=mock_container)
    mock_av.open.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(StopIteration):
        _extract_frame_at("/fake/video.mp4", 999.0, 768, 75)


@patch("video_decomposer_mcp.tools.frames.cv2")
@patch("video_decomposer_mcp.tools.frames.av")
def test_extract_frame_at_encode_failure(mock_av, mock_cv2):
    mock_frame = MagicMock()
    mock_frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_stream = MagicMock()
    mock_stream.time_base = 1 / 1000

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.decode.return_value = iter([mock_frame])
    mock_av.open.return_value.__enter__ = MagicMock(return_value=mock_container)
    mock_av.open.return_value.__exit__ = MagicMock(return_value=False)

    mock_cv2.IMWRITE_JPEG_QUALITY = 1
    mock_cv2.imencode.return_value = (False, None)

    with pytest.raises(RuntimeError, match="Failed to encode frame as JPEG"):
        _extract_frame_at("/fake/video.mp4", 5.0, 768, 75)


@patch("video_decomposer_mcp.tools.frames._extract_frame_at")
async def test_do_extract_frame(mock_extract, store_with_video):
    store, video_id, _ = store_with_video
    mock_extract.return_value = b"jpeg data"

    result = await do_extract_frame(store, video_id, 5.0)

    assert isinstance(result, Image)
    assert result.data == b"jpeg data"
    assert result._format == "jpeg"
    mock_extract.assert_called_once()


@patch("video_decomposer_mcp.tools.frames._extract_frame_at")
async def test_do_extract_frame_cache_hit(mock_extract, store_with_video):
    store, video_id, _ = store_with_video
    frames_dir = store.frames_dir(video_id)

    # Pre-create cached frame
    cache_path = frames_dir / "5000.jpg"
    cache_path.write_bytes(b"cached jpeg")

    result = await do_extract_frame(store, video_id, 5.0)

    assert isinstance(result, Image)
    assert result.data == b"cached jpeg"
    # Should not call _extract_frame_at since cache exists
    mock_extract.assert_not_called()


@patch("video_decomposer_mcp.tools.frames._extract_frame_at")
async def test_do_extract_frame_writes_cache(mock_extract, store_with_video):
    store, video_id, _ = store_with_video
    mock_extract.return_value = b"new jpeg"

    await do_extract_frame(store, video_id, 2.5)

    # Should have written cache file
    frames_dir = store.frames_dir(video_id)
    cache_path = frames_dir / "2500.jpg"
    assert cache_path.exists()
    assert cache_path.read_bytes() == b"new jpeg"
