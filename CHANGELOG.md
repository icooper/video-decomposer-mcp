# CHANGELOG

<!-- version list -->

## v1.2.0 (2026-04-08)

### Chores

- Fixes per code review
  ([`00d1270`](https://github.com/icooper/video-decomposer-mcp/commit/00d1270c0e6b88a34291062fcea3b63384785749))

### Features

- Caching for transcription, alignment, and diarization
  ([`7c11964`](https://github.com/icooper/video-decomposer-mcp/commit/7c119649b6e6c0de0f7eeb93c9630cc617d0c641))


## v1.1.0 (2026-04-01)

- Added optional speaker diarization: `diarize_speakers` parameter (default: false) on `transcribe_video` and `analyze_video` identifies who said what. Each segment includes a `speaker` field and the full text is annotated with speaker labels. Requires `HF_TOKEN` environment variable.
- `extract_frame` now returns a native MCP image content block instead of a base64-encoded dict, improving performance for LLM clients.

## v1.0.0 (2026-04-01)

- Initial Release
