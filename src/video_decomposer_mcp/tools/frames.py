import asyncio
import base64
import logging
from functools import partial

import av
import cv2

from ..video_store import VideoStore

logger = logging.getLogger(__name__)


def _extract_frame_at(video_path: str, timestamp: float, max_dimension: int, quality: int) -> bytes:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        target_pts = int(timestamp / stream.time_base)
        container.seek(target_pts, stream=stream)
        frame = next(container.decode(stream))
    img = frame.to_ndarray(format="bgr24")
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


async def do_extract_frame(
    store: VideoStore,
    video_id: str,
    timestamp: float,
    *,
    max_dimension: int = 768,
    quality: int = 75,
) -> dict:
    logger.info("Extracting frame video_id=%s timestamp=%.3f", video_id, timestamp)
    record = store.get(video_id)
    frames_dir = store.frames_dir(video_id)

    cache_key = f"{int(timestamp * 1000)}.jpg"
    cache_path = frames_dir / cache_key

    if cache_path.exists():
        logger.debug("Cache hit for frame at %.3fs", timestamp)
        data = cache_path.read_bytes()
    else:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            None,
            partial(
                _extract_frame_at,
                str(record.file_path),
                timestamp,
                max_dimension,
                quality,
            ),
        )
        cache_path.write_bytes(data)

    encoded = base64.b64encode(data).decode("utf-8")
    return {
        "type": "image",
        "data": encoded,
        "mimeType": "image/jpeg",
        "timestamp": timestamp,
    }
