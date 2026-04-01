import logging
import os
import warnings

# Suppress noisy third-party warnings before they are triggered by imports
warnings.filterwarnings("ignore", message="TensorFloat-32.*", category=UserWarning)
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom", category=UserWarning)


def configure_logging(default_level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Uses LOG_LEVEL environment variable if set, otherwise falls back to default_level.
    Overrides third-party logger levels and formats (whisperx, lightning).
    """
    level_name = os.environ.get("LOG_LEVEL", "").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = default_level
    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    log_datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=log_datefmt,
    )
    # whisperx and lightning configure their own loggers with handlers, override them
    formatter = logging.Formatter(log_format, datefmt=log_datefmt)
    for name in ("whisperx", "lightning", "lightning.pytorch"):
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(level)
        for handler in lib_logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)

    # Patch uvicorn's LOGGING_CONFIG so that when uvicorn calls dictConfig at
    # startup it uses our format instead of its default.
    import uvicorn.config

    uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["fmt"] = log_format
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["datefmt"] = log_datefmt
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["fmt"] = log_format
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["datefmt"] = log_datefmt
