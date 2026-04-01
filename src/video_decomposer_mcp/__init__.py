import logging
import os


def configure_logging(default_level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Uses LOG_LEVEL environment variable if set, otherwise falls back to default_level.
    """
    level_name = os.environ.get("LOG_LEVEL", "").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = default_level
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
