import asyncio
import logging
import shutil
from functools import partial
from pathlib import Path

import yt_dlp

from ..video_store import VideoStore

logger = logging.getLogger(__name__)

_url_locks: dict[str, asyncio.Lock] = {}


def _download(video_dir: Path, url: str) -> Path:
    ydl_opts = {
        "outtmpl": str(video_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # After merge, extension may differ from original
        merged = Path(filename).with_suffix(".mp4")
        if merged.exists():
            logger.debug("Using merged .mp4 path: %s", merged)
            return merged
        logger.debug("Using original filename: %s", filename)
        return Path(filename)


async def do_download(store: VideoStore, url: str) -> str:
    if url not in _url_locks:
        _url_locks[url] = asyncio.Lock()
    async with _url_locks[url]:
        existing = store.find_by_url(url)
        if existing is not None:
            logger.info("Video already downloaded video_id=%s url=%s", existing.video_id, url)
            return existing.video_id
        logger.info("Downloading video url=%s", url)
        video_id, video_dir = store.create_entry(url)
        loop = asyncio.get_running_loop()
        try:
            file_path = await loop.run_in_executor(None, partial(_download, video_dir, url))
            store.register(video_id, url, file_path)
        except Exception:
            shutil.rmtree(video_dir, ignore_errors=True)
            raise
        logger.info("Download complete video_id=%s path=%s", video_id, file_path)
        return video_id
