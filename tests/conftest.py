from pathlib import Path

import pytest

from video_decomposer_mcp.video_store import VideoStore


@pytest.fixture
def store(tmp_path: Path) -> VideoStore:
    return VideoStore(base_dir=tmp_path / "videos")


@pytest.fixture
def store_with_video(store: VideoStore) -> tuple[VideoStore, str, Path]:
    """A store with a pre-registered video."""
    video_id, video_dir = store.create_entry("https://example.com/video")
    video_file = video_dir / "test.mp4"
    video_file.write_bytes(b"fake video content")
    store.register(video_id, "https://example.com/video", video_file)
    return store, video_id, video_file
