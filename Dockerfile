FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS ffmpeg-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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

# Install nv-codec-headers for NVDEC/NVENC support
RUN git clone --depth 1 --branch n12.2.72.0 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
    && cd nv-codec-headers && make install && cd .. && rm -rf nv-codec-headers

# Build FFmpeg with NVDEC hardware decoding
RUN curl -sL https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz | tar xJ -C /tmp \
    && cd /tmp/ffmpeg-8.0.1 \
    && ./configure \
        --prefix=/usr/local \
        --enable-gpl \
        --enable-nonfree \
        --enable-cuvid \
        --enable-nvdec \
        --enable-nvenc \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
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

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Copy FFmpeg with NVDEC support from builder
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

# Install deps first (cache layer)
# Build av from source against our NVDEC-enabled FFmpeg — needs build tools in a single layer
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

# MCP server configuration variables
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

CMD ["uv", "run", "server"]
