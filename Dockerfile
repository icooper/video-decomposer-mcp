ARG BASE_BUILD=nvidia/cuda:12.8.1-devel-ubuntu22.04
ARG BASE_RUNTIME=nvidia/cuda:12.8.1-runtime-ubuntu22.04

FROM ${BASE_BUILD} AS ffmpeg-builder

ARG FFMPEG_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    nasm \
    pkg-config \
    xz-utils \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libopus-dev \
    libdav1d-dev \
    && rm -rf /var/lib/apt/lists/*

# Install nv-codec-headers for NVDEC/NVENC support (CUDA only)
RUN if [ "$FFMPEG_CUDA" = "1" ]; then \
        git clone --depth 1 --branch n12.2.72.0 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
        && cd nv-codec-headers && make install && cd .. && rm -rf nv-codec-headers; \
    fi

# Build FFmpeg (with NVDEC/NVENC when CUDA is enabled)
RUN CUDA_FLAGS="" && \
    if [ "$FFMPEG_CUDA" = "1" ]; then \
        CUDA_FLAGS="--enable-nonfree --enable-cuvid --enable-nvdec --enable-nvenc \
            --extra-cflags=-I/usr/local/cuda/include \
            --extra-ldflags=-L/usr/local/cuda/lib64"; \
    fi && \
    curl -sL https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz | tar xJ -C /tmp \
    && cd /tmp/ffmpeg-8.0.1 \
    && ./configure \
        --prefix=/usr/local \
        --enable-gpl \
        $CUDA_FLAGS \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libopus \
        --enable-libdav1d \
        --enable-shared \
        --disable-static \
        --disable-doc \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/ffmpeg-8.0.1

# ---

ARG BASE_RUNTIME
FROM ${BASE_RUNTIME}

LABEL org.opencontainers.image.description="MCP server for video decomposition: download, transcribe, and extract key frames"

# Copy FFmpeg from builder
COPY --from=ffmpeg-builder /usr/local/bin/ff* /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/lib/lib*.so* /usr/local/lib/
COPY --from=ffmpeg-builder /usr/local/lib/pkgconfig/ /usr/local/lib/pkgconfig/
COPY --from=ffmpeg-builder /usr/local/include/ /usr/local/include/
RUN ldconfig

# Runtime deps for FFmpeg codec libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx264-163 \
    libx265-199 \
    libvpx7 \
    libopus0 \
    libdav1d5 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

# Install Python 3.12 via uv
RUN uv python install 3.12

WORKDIR /app

# Override the torch index URL (defaults to CUDA 12.8, can be set to CPU)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV UV_INDEX="pytorch-cu128=${TORCH_INDEX_URL}" \
    UV_INDEX_STRATEGY=unsafe-best-match

# Install deps first (cache layer)
# Build av from source against our FFmpeg — needs build tools in a single layer
COPY pyproject.toml uv.lock ./
ENV UV_NO_BINARY_PACKAGE=av
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    build-essential \
    && uv sync --no-install-project \
    && apt-get purge -y build-essential pkg-config \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY src/ src/
COPY README.md ./
RUN uv sync

# Application configuration (defaults match .env.example)
ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    VIDEO_STORE=/app/video_store \
    VIDEO_STORE_TTL_SECONDS=14400 \
    VIDEO_STORE_CLEANUP_INTERVAL_SECONDS=600 \
    WHISPER_MODEL=turbo \
    PRELOAD_ALIGN_LANGUAGE=en \
    LOG_LEVEL=INFO \
    HF_TOKEN= \
    HF_HOME=/app/hf_cache \
    NLTK_DATA=/app/nltk_data \
    PYTHONWARNINGS="ignore::UserWarning:pyannote.audio.core.io"

CMD ["uv", "run", "server"]
