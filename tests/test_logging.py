import logging
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
