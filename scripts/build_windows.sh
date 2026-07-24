#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="tales-of-derision-windows-builder"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.7}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build the Windows executable from Linux." >&2
  exit 1
fi

cd "$REPO_ROOT"

docker build \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  -t "$IMAGE_NAME" \
  -f packaging/windows.Dockerfile \
  packaging

docker run --rm \
  -v "$REPO_ROOT:/workspace" \
  -w /workspace \
  "$IMAGE_NAME" \
  xvfb-run -a --server-args="-screen 0 1024x768x24" \
  /bin/bash packaging/build_windows_in_container.sh

echo "Built dist/TalesOfDerision.exe"
