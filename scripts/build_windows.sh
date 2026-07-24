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
  /bin/bash -lc 'wine C:\\Python312\\python.exe -m pip install --upgrade pip && wine C:\\Python312\\python.exe -m pip install esper pygame-ce pyinstaller && wine C:\\Python312\\Scripts\\pyinstaller.exe --clean --noconfirm packaging/tales_of_derision_windows.spec'

echo "Built dist/TalesOfDerision.exe"
