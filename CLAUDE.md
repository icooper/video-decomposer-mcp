# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP server for video decomposition and analysis: downloads videos (yt-dlp), transcribes audio (Whisper with CUDA), and extracts key frames (PySceneDetect). Runs as an HTTP MCP server on a remote machine with an NVIDIA GPU.

## Commands

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests (enforces 100% coverage)
uv run server              # Start MCP server on port 8000          
docker compose up --build  # Run in Docker with GPU passthrough
```

## Architecture

- **`server.py`** — FastMCP server (streamable-http on 0.0.0.0:8000, stateless). Registers four tools that delegate to tool modules. Uses a module-level `VideoStore` instance shared across requests.
- **`video_store.py`** — Manages downloaded videos in a temp directory. Videos are keyed by short UUID IDs. In-memory dict of `video_id -> VideoRecord` (file path, URL, timestamp).
- **`tools/download.py`** — Embeds yt-dlp as a library (not subprocess). Runs in `run_in_executor`.
- **`tools/transcribe.py`** — Whisper model loaded once and cached in a module-level dict. CUDA auto-detected. Runs in `run_in_executor`.
- **`tools/frames.py`** — PySceneDetect `ContentDetector` for scene boundaries, saves one frame per scene as JPEG, returns base64-encoded image dicts. Runs in `run_in_executor`.
- **`tools/analyze.py`** — Orchestrates download → transcribe + extract_frames.

## Key Patterns

- All blocking operations (yt-dlp, whisper, scenedetect) run via `asyncio.get_running_loop().run_in_executor()` to avoid blocking the async event loop.
- Tests mock all external libraries (yt-dlp, whisper, torch, scenedetect) — no network or GPU needed to run tests.
- Python 3.12 is pinned via pyenv (PyTorch compatibility). PyTorch uses CUDA 12.4 index configured in `[tool.uv]`.
