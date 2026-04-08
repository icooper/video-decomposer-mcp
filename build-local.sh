#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-cu128}"

case "$VARIANT" in
  cu128)
    BASE_BUILD=nvidia/cuda:12.8.1-devel-ubuntu22.04
    BASE_RUNTIME=nvidia/cuda:12.8.1-runtime-ubuntu22.04
    FFMPEG_CUDA=1
    TORCH_INDEX=https://download.pytorch.org/whl/cu128
    ;;
  cpu)
    BASE_BUILD=ubuntu:22.04
    BASE_RUNTIME=ubuntu:22.04
    FFMPEG_CUDA=0
    TORCH_INDEX=https://download.pytorch.org/whl/cpu
    ;;
  *)
    echo "Usage: $0 [cu128|cpu]" >&2
    exit 1
    ;;
esac

BASE_TAG="video-decomposer-base:local-${VARIANT}"
APP_TAG="video-decomposer-mcp:local-${VARIANT}"

echo "==> Building base image: ${BASE_TAG}"
docker build -f Dockerfile.base \
  --build-arg BASE_BUILD="$BASE_BUILD" \
  --build-arg BASE_RUNTIME="$BASE_RUNTIME" \
  --build-arg FFMPEG_CUDA="$FFMPEG_CUDA" \
  -t "$BASE_TAG" .

echo "==> Building app image: ${APP_TAG}"
docker build \
  --build-arg BASE_IMAGE="$BASE_TAG" \
  --build-arg TORCH_INDEX_URL="$TORCH_INDEX" \
  -t "$APP_TAG" .

echo "==> Done. Run with:"
echo "    docker run --rm -p 8000:8000 ${APP_TAG}"
