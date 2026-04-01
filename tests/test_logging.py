import logging
import warnings
from unittest.mock import patch

from video_decomposer_mcp import configure_logging


@patch("logging.basicConfig")
def test_configure_logging_default(mock_basic):
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("LOG_LEVEL", None)
        configure_logging(logging.INFO)
    mock_basic.assert_called_once_with(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@patch("logging.basicConfig")
def test_configure_logging_env_override(mock_basic):
    with patch.dict("os.environ", {"LOG_LEVEL": "DEBUG"}):
        configure_logging(logging.INFO)
    mock_basic.assert_called_once_with(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@patch("logging.basicConfig")
def test_configure_logging_env_case_insensitive(mock_basic):
    with patch.dict("os.environ", {"LOG_LEVEL": "warning"}):
        configure_logging(logging.INFO)
    mock_basic.assert_called_once_with(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@patch("logging.basicConfig")
def test_configure_logging_invalid_env(mock_basic):
    with patch.dict("os.environ", {"LOG_LEVEL": "notavalidlevel"}):
        configure_logging(logging.INFO)
    mock_basic.assert_called_once_with(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def test_warning_filters_set_at_module_level():
    """Warning filters are set at import time so they apply before third-party imports."""
    import importlib

    import video_decomposer_mcp

    importlib.reload(video_decomposer_mcp)
    filter_patterns = [f[1].pattern if hasattr(f[1], "pattern") else "" for f in warnings.filters]
    assert any("TensorFloat-32" in p for p in filter_patterns)
    assert any("degrees of freedom" in p for p in filter_patterns)


def test_configure_logging_sets_third_party_logger_levels():
    configure_logging(logging.ERROR)
    for name in ("whisperx", "lightning", "lightning.pytorch"):
        lib_logger = logging.getLogger(name)
        assert lib_logger.level == logging.ERROR
        for handler in lib_logger.handlers:
            assert handler.level == logging.ERROR


def test_configure_logging_patches_uvicorn_config():
    import uvicorn.config

    configure_logging(logging.INFO)
    assert (
        uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["fmt"]
        == "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    assert uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["datefmt"] == "%Y-%m-%dT%H:%M:%S"
    assert (
        uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["fmt"] == "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    assert uvicorn.config.LOGGING_CONFIG["formatters"]["access"]["datefmt"] == "%Y-%m-%dT%H:%M:%S"
