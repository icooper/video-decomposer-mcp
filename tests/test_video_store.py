import json
import shutil
import time
from pathlib import Path

import pytest

from video_decomposer_mcp.video_store import VideoStore


def test_create_entry(store: VideoStore):
    video_id, video_dir = store.create_entry("https://example.com/v")
    assert len(video_id) == 12
    assert video_dir.exists()
    assert video_dir.parent == store.base_dir


def test_register_and_get(store: VideoStore):
    video_id, video_dir = store.create_entry("https://example.com/v")
    file_path = video_dir / "video.mp4"
    file_path.touch()
    record = store.register(video_id, "https://example.com/v", file_path)
    assert record.video_id == video_id
    assert record.url == "https://example.com/v"
    assert record.file_path == file_path
    assert record.downloaded_at > 0

    retrieved = store.get(video_id)
    assert retrieved is record


def test_get_nonexistent(store: VideoStore):
    with pytest.raises(KeyError, match="Video not found"):
        store.get("nonexistent")


def test_frames_dir(store: VideoStore):
    video_id, video_dir = store.create_entry("https://example.com/v")
    file_path = video_dir / "video.mp4"
    file_path.touch()
    store.register(video_id, "https://example.com/v", file_path)

    frames = store.frames_dir(video_id)
    assert frames.exists()
    assert frames.name == "frames"
    assert frames.parent == video_dir

    # Calling again should not raise (exist_ok=True)
    frames2 = store.frames_dir(video_id)
    assert frames2 == frames


def test_default_base_dir():
    store = VideoStore()
    assert store.base_dir.exists()
    assert "video-decomposer-" in store.base_dir.name


def test_custom_base_dir(tmp_path: Path):
    custom = tmp_path / "custom" / "nested"
    store = VideoStore(base_dir=custom)
    assert store.base_dir == custom
    assert custom.exists()


def test_ttl_default(tmp_path: Path):
    s = VideoStore(base_dir=tmp_path / "v")
    assert s.ttl_seconds == 4 * 3600


def test_ttl_custom(tmp_path: Path):
    s = VideoStore(base_dir=tmp_path / "v", ttl_seconds=60)
    assert s.ttl_seconds == 60


def test_get_expired_raises(tmp_path: Path):
    s = VideoStore(base_dir=tmp_path / "v", ttl_seconds=1)
    video_id, video_dir = s.create_entry("https://example.com/v")
    video_file = video_dir / "video.mp4"
    video_file.write_bytes(b"data")
    record = s.register(video_id, "https://example.com/v", video_file)
    record.downloaded_at = time.time() - 5

    with pytest.raises(KeyError, match="Video not found"):
        s.get(video_id)

    assert not video_dir.exists()
    assert video_id not in s._videos


def test_get_unexpired_returns(tmp_path: Path):
    s = VideoStore(base_dir=tmp_path / "v", ttl_seconds=3600)
    video_id, video_dir = s.create_entry("https://example.com/v")
    video_file = video_dir / "video.mp4"
    video_file.touch()
    s.register(video_id, "https://example.com/v", video_file)

    record = s.get(video_id)
    assert record.video_id == video_id


def test_cleanup_removes_expired(tmp_path: Path):
    s = VideoStore(base_dir=tmp_path / "v", ttl_seconds=100)
    # Video 1: expired
    vid1, dir1 = s.create_entry("https://example.com/v1")
    f1 = dir1 / "v.mp4"
    f1.write_bytes(b"data")
    rec1 = s.register(vid1, "https://example.com/v1", f1)
    rec1.downloaded_at = time.time() - 200

    # Video 2: fresh
    vid2, dir2 = s.create_entry("https://example.com/v2")
    f2 = dir2 / "v.mp4"
    f2.write_bytes(b"data")
    s.register(vid2, "https://example.com/v2", f2)

    count = s.cleanup()
    assert count == 1
    assert not dir1.exists()
    assert vid1 not in s._videos
    assert dir2.exists()
    assert s.get(vid2).video_id == vid2


def test_cleanup_no_expired(store_with_video):
    s, video_id, _ = store_with_video
    count = s.cleanup()
    assert count == 0
    assert s.get(video_id).video_id == video_id


async def test_async_cleanup(tmp_path: Path):
    s = VideoStore(base_dir=tmp_path / "v", ttl_seconds=100)
    vid, vdir = s.create_entry("https://example.com/v")
    f = vdir / "v.mp4"
    f.write_bytes(b"data")
    rec = s.register(vid, "https://example.com/v", f)
    rec.downloaded_at = time.time() - 200

    count = await s.async_cleanup()
    assert count == 1
    assert not vdir.exists()
    assert vid not in s._videos


def test_evict_nonexistent_video(tmp_path: Path):
    """Calling _evict on an ID not in the store is a no-op."""
    s = VideoStore(base_dir=tmp_path / "v")
    s._evict("nonexistent")  # should not raise


def test_evict_missing_directory(tmp_path: Path):
    s = VideoStore(base_dir=tmp_path / "v", ttl_seconds=1)
    vid, vdir = s.create_entry("https://example.com/v")
    f = vdir / "v.mp4"
    f.write_bytes(b"data")
    rec = s.register(vid, "https://example.com/v", f)
    rec.downloaded_at = time.time() - 5

    # Remove directory before eviction
    shutil.rmtree(vdir)

    with pytest.raises(KeyError, match="Video not found"):
        s.get(vid)

    assert vid not in s._videos


def test_scan_existing_restores_videos(tmp_path: Path):
    base = tmp_path / "v"
    s1 = VideoStore(base_dir=base)
    vid, vdir = s1.create_entry("https://example.com/v")
    f = vdir / "v.mp4"
    f.write_bytes(b"data")
    s1.register(vid, "https://example.com/v", f)

    # New instance should discover the video from disk
    s2 = VideoStore(base_dir=base)
    record = s2.get(vid)
    assert record.video_id == vid
    assert record.file_path == f


def test_scan_existing_skips_dirs_without_video_files(tmp_path: Path):
    base = tmp_path / "v"
    base.mkdir(parents=True)
    # Create a directory with only a JSON file (no video files)
    (base / "empty_dir").mkdir()
    (base / "empty_dir" / "manifest.json").write_text("{}")

    s = VideoStore(base_dir=base)
    assert len(s._videos) == 0


def test_scan_existing_discovers_non_mp4_video(tmp_path: Path):
    base = tmp_path / "v"
    base.mkdir(parents=True)
    vdir = base / "abc123456789"
    vdir.mkdir()
    (vdir / "video.webm").write_bytes(b"webm data")

    s = VideoStore(base_dir=base)
    record = s.get("abc123456789")
    assert record.file_path.suffix == ".webm"


def test_scan_existing_skips_files(tmp_path: Path):
    base = tmp_path / "v"
    base.mkdir(parents=True)
    # Create a plain file (not a directory) in base_dir
    (base / "stray_file.txt").write_text("not a dir")

    s = VideoStore(base_dir=base)
    assert len(s._videos) == 0


def test_register_writes_manifest(store: VideoStore):
    video_id, video_dir = store.create_entry("https://example.com/v")
    file_path = video_dir / "video.mp4"
    file_path.touch()
    store.register(video_id, "https://example.com/v", file_path)

    manifest_path = video_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["url"] == "https://example.com/v"
    assert manifest["video_id"] == video_id
    assert manifest["downloaded_at"] > 0


def test_scan_existing_restores_url_from_manifest(tmp_path: Path):
    base = tmp_path / "v"
    s1 = VideoStore(base_dir=base)
    vid, vdir = s1.create_entry("https://example.com/v")
    f = vdir / "v.mp4"
    f.write_bytes(b"data")
    s1.register(vid, "https://example.com/v", f)

    s2 = VideoStore(base_dir=base)
    record = s2.get(vid)
    assert record.url == "https://example.com/v"


def test_scan_existing_handles_missing_manifest(tmp_path: Path):
    base = tmp_path / "v"
    base.mkdir(parents=True)
    vdir = base / "abc123456789"
    vdir.mkdir()
    (vdir / "video.mp4").write_bytes(b"data")
    # No manifest.json

    s = VideoStore(base_dir=base)
    record = s.get("abc123456789")
    assert record.url == ""


def test_scan_existing_handles_corrupt_manifest(tmp_path: Path):
    base = tmp_path / "v"
    base.mkdir(parents=True)
    vdir = base / "abc123456789"
    vdir.mkdir()
    (vdir / "video.mp4").write_bytes(b"data")
    (vdir / "manifest.json").write_text("not json{{{")

    s = VideoStore(base_dir=base)
    record = s.get("abc123456789")
    assert record.url == ""


def test_find_by_url_returns_record(store: VideoStore):
    video_id, video_dir = store.create_entry("https://example.com/v")
    file_path = video_dir / "video.mp4"
    file_path.touch()
    record = store.register(video_id, "https://example.com/v", file_path)

    found = store.find_by_url("https://example.com/v")
    assert found is record


def test_find_by_url_returns_none_for_unknown(store: VideoStore):
    assert store.find_by_url("https://example.com/nope") is None


def test_find_by_url_skips_expired(tmp_path: Path):
    s = VideoStore(base_dir=tmp_path / "v", ttl_seconds=1)
    vid, vdir = s.create_entry("https://example.com/v")
    f = vdir / "v.mp4"
    f.write_bytes(b"data")
    rec = s.register(vid, "https://example.com/v", f)
    rec.downloaded_at = time.time() - 5

    assert s.find_by_url("https://example.com/v") is None
