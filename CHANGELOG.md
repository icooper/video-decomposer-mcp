# CHANGELOG

<!-- version list -->

## v1.2.0 (2026-04-08)

- Cache transcription, alignment, and diarization results to disk per video. Repeated transcriptions skip expensive computation entirely when all caches hit.
- Report progress during transcription via MCP progress notifications, reducing the occurrence of client timeouts on long videos.

## v1.1.0 (2026-04-01)

- Added optional speaker diarization: `diarize_speakers` parameter (default: false) on `transcribe_video` and `analyze_video` identifies who said what. Each segment includes a `speaker` field and the full text is annotated with speaker labels. Requires `HF_TOKEN` environment variable.
- `extract_frame` now returns a native MCP image content block instead of a base64-encoded dict, improving performance for LLM clients.

## v1.0.0 (2026-04-01)

- Initial Release
