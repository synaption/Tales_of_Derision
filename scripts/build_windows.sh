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
  /bin/bash -lc 'xvfb-run -a wine C:\\Python312\\python.exe -m pip install --disable-pip-version-check -r requirements.txt pyinstaller && xvfb-run -a wine C:\\Python312\\python.exe -m PyInstaller --clean --noconfirm packaging/tales_of_derision_windows.spec'

echo "Built dist/TalesOfDerision.exe"
