# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP server for video decomposition and analysis: downloads videos (yt-dlp), transcribes audio (WhisperX with CUDA), identifies speakers (pyannote.audio), and extracts key frames (PyAV + OpenCV). Runs as an HTTP MCP server on a remote machine with an NVIDIA GPU.

## Commands

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests (enforces 100% coverage)
uv run server              # Start MCP server on port 8000          
docker compose up --build  # Run in Docker with GPU passthrough
```

## Architecture

- **`server.py`** — FastMCP server (SSE on 127.0.0.1:8000). Registers four tools that delegate to tool modules. Uses a module-level `VideoStore` instance shared across requests.
- **`video_store.py`** — Manages downloaded videos in a temp directory. Videos are keyed by short UUID IDs. In-memory dict of `video_id -> VideoRecord` (file path, URL, timestamp).
- **`tools/download.py`** — Embeds yt-dlp as a library (not subprocess). Runs in `run_in_executor`.
- **`tools/transcribe.py`** — WhisperX transcription with optional speaker diarization (pyannote.audio). Caches WhisperX model, alignment model, and diarization pipeline in module-level dicts. CUDA auto-detected. Runs in `run_in_executor`.
- **`tools/frames.py`** — PyAV + OpenCV frame extraction at specific timestamps, returns base64-encoded JPEG image dicts with caching. Runs in `run_in_executor`.
- **`tools/analyze.py`** — Orchestrates download → transcribe (with diarization).

## Release Notes

When making user-facing changes (new features, API changes, bug fixes), add a short description to `RELEASE_NOTES.md` at the root of the repo. Create the file if it doesn't exist. Don't create or add any headers. Do keep entries concise — one bullet per change.

## Documentation

When making user-facing or architectural changes (new features, API changes, or underlying design changes), update `README.md` at the root of the repo to keep it current with the implementation.

## Key Patterns

- All blocking operations (yt-dlp, whisperx, pyannote, av/opencv) run via `asyncio.get_running_loop().run_in_executor()` to avoid blocking the async event loop.
- Tests mock all calls to external libraries (whisperx, pyannote, torch, yt-dlp, av, opencv) so no network, GPU, or video files are needed at test time. The libraries themselves must still be installed since source modules import them at the module level.
- Python 3.12 is pinned via pyenv (PyTorch compatibility). PyTorch uses CUDA 12.8 index configured in `[tool.uv]`.
