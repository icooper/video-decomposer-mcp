import asyncio
import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VideoRecord:
    video_id: str
    url: str
    file_path: Path
    downloaded_at: float


class VideoStore:
    def __init__(self, base_dir: Path | None = None, ttl_seconds: float = 4 * 3600):
        if base_dir is not None:
            self.base_dir = base_dir
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.base_dir = Path(tempfile.mkdtemp(prefix="video-decomposer-"))
        self.ttl_seconds = ttl_seconds
        self._videos: dict[str, VideoRecord] = {}
        self._scan_existing()
        logger.debug("Initialized store at %s", self.base_dir)

    def _scan_existing(self) -> None:
        """Rebuild the in-memory registry from video directories on disk."""
        for video_dir in self.base_dir.iterdir():
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            video_files = list(video_dir.glob("*.mp4"))
            if not video_files:
                continue
            file_path = video_files[0]
            self._videos[video_id] = VideoRecord(
                video_id=video_id,
                url="",
                file_path=file_path,
                downloaded_at=file_path.stat().st_mtime,
            )

    def create_entry(self, url: str) -> tuple[str, Path]:
        video_id = uuid.uuid4().hex[:12]
        video_dir = self.base_dir / video_id
        video_dir.mkdir()
        logger.debug("Created entry video_id=%s", video_id)
        return video_id, video_dir

    def register(self, video_id: str, url: str, file_path: Path) -> VideoRecord:
        record = VideoRecord(
            video_id=video_id,
            url=url,
            file_path=file_path,
            downloaded_at=time.time(),
        )
        self._videos[video_id] = record
        logger.debug("Registered video video_id=%s path=%s", video_id, file_path)
        return record

    def get(self, video_id: str) -> VideoRecord:
        if video_id not in self._videos:
            raise KeyError(f"Video not found: {video_id}")
        record = self._videos[video_id]
        if self._is_expired(record):
            self._evict(video_id)
            raise KeyError(f"Video not found: {video_id}")
        return record

    def frames_dir(self, video_id: str) -> Path:
        record = self.get(video_id)
        frames = record.file_path.parent / "frames"
        frames.mkdir(exist_ok=True)
        return frames

    def _is_expired(self, record: VideoRecord) -> bool:
        return (time.time() - record.downloaded_at) >= self.ttl_seconds

    def _evict(self, video_id: str) -> None:
        record = self._videos.pop(video_id, None)
        if record is None:
            return
        video_dir = record.file_path.parent
        if video_dir.exists():
            shutil.rmtree(video_dir)
        logger.info("Evicted expired video video_id=%s", video_id)

    def cleanup(self) -> int:
        now = time.time()
        expired_ids = [vid for vid, rec in self._videos.items() if (now - rec.downloaded_at) >= self.ttl_seconds]
        for vid in expired_ids:
            self._evict(vid)
        return len(expired_ids)

    async def async_cleanup(self) -> int:
        loop = asyncio.get_running_loop()
        now = time.time()
        expired_ids = [vid for vid, rec in self._videos.items() if (now - rec.downloaded_at) >= self.ttl_seconds]
        for vid in expired_ids:
            record = self._videos.pop(vid)
            video_dir = record.file_path.parent
            if video_dir.exists():
                await loop.run_in_executor(None, shutil.rmtree, video_dir)
            logger.info("Evicted expired video video_id=%s", vid)
        return len(expired_ids)
