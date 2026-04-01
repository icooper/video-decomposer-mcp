from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_decomposer_mcp.tools.download import _download, do_download


@pytest.fixture
def mock_ydl():
    with patch("video_decomposer_mcp.tools.download.yt_dlp.YoutubeDL") as mock_cls:
        ydl_instance = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=ydl_instance)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_cls, ydl_instance


def test_download_sync(tmp_path: Path, mock_ydl):
    mock_cls, ydl_instance = mock_ydl
    video_dir = tmp_path / "video"
    video_dir.mkdir()

    # Simulate yt-dlp creating a merged mp4
    output_file = video_dir / "abc123.mp4"
    output_file.touch()

    ydl_instance.extract_info.return_value = {"id": "abc123", "ext": "mp4"}
    ydl_instance.prepare_filename.return_value = str(output_file)

    result = _download(video_dir, "https://example.com/v")
    assert result == output_file

    # Verify opts
    call_args = mock_cls.call_args[0][0]
    assert call_args["merge_output_format"] == "mp4"
    assert call_args["quiet"] is True


def test_download_non_mp4_fallback(tmp_path: Path, mock_ydl):
    mock_cls, ydl_instance = mock_ydl
    video_dir = tmp_path / "video"
    video_dir.mkdir()

    # The original file exists but .mp4 doesn't
    output_file = video_dir / "abc123.webm"
    output_file.touch()

    ydl_instance.extract_info.return_value = {"id": "abc123", "ext": "webm"}
    ydl_instance.prepare_filename.return_value = str(output_file)

    result = _download(video_dir, "https://example.com/v")
    assert result == output_file


async def test_do_download(store_with_video, mock_ydl):
    store, _, _ = store_with_video
    mock_cls, ydl_instance = mock_ydl

    # We need to set up the mock to create a file in the right place
    def fake_extract(url, download=True):
        # Find the video dir that was created
        dirs = [d for d in store.base_dir.iterdir() if d.is_dir()]
        newest = max(dirs, key=lambda d: d.stat().st_ctime)
        (newest / "test_id.mp4").touch()
        return {"id": "test_id", "ext": "mp4"}

    ydl_instance.extract_info.side_effect = fake_extract
    ydl_instance.prepare_filename.return_value = "placeholder"

    # We need a fresh store so do_download creates its own entry
    from video_decomposer_mcp.video_store import VideoStore

    fresh_store = VideoStore(base_dir=store.base_dir / "fresh")

    # Override prepare_filename to return the actual path
    def fake_prepare(info):
        dirs = [d for d in fresh_store.base_dir.iterdir() if d.is_dir()]
        newest = max(dirs, key=lambda d: d.stat().st_ctime)
        return str(newest / f"{info['id']}.{info['ext']}")

    ydl_instance.prepare_filename.side_effect = fake_prepare

    video_id = await do_download(fresh_store, "https://example.com/v")
    assert len(video_id) == 12
    record = fresh_store.get(video_id)
    assert record.url == "https://example.com/v"


async def test_do_download_dedup_skips_redownload(store_with_video):
    store, video_id, _ = store_with_video
    result = await do_download(store, "https://example.com/video")
    assert result == video_id
