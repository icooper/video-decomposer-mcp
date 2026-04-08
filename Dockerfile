ARG BASE_IMAGE=ghcr.io/icooper/video-decomposer-base:ffmpeg8.0.1-av17.0.0-cu128
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.description="MCP server for video decomposition: download, transcribe, and extract key frames"

WORKDIR /app

# Override the torch index URL (defaults to CUDA 12.8, can be set to CPU)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV UV_INDEX="pytorch-cu128=${TORCH_INDEX_URL}" \
    UV_INDEX_STRATEGY=unsafe-best-match

# Install all Python deps (torch, whisperx, etc.) into the base venv.
# av is pre-installed from the base image and will be skipped.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

# Copy source
COPY src/ src/
COPY README.md ./
RUN uv sync

# Application configuration (defaults match .env.example)
ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    VIDEO_STORE_PATH=/app/video_store \
    VIDEO_STORE_TTL_SECONDS=14400 \
    VIDEO_STORE_CLEANUP_INTERVAL_SECONDS=600 \
    WHISPER_MODEL=turbo \
    PRELOAD_ALIGN_LANGUAGE=en \
    LOG_LEVEL=INFO \
    HF_HOME=/app/hf_cache \
    NLTK_DATA=/app/nltk_data \
    PYTHONWARNINGS="ignore::UserWarning:pyannote.audio.core.io"

CMD ["uv", "run", "server"]
