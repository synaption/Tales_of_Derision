#!/usr/bin/env bash
set -euo pipefail

WINDOWS_PYTHON='C:\Python312\python.exe'

# Keep dependency installation and packaging on one X server. Wine keeps a
# background server for a prefix, so using separate xvfb-run calls can leave it
# connected to an X display that has already shut down.
trap 'wineserver -k >/dev/null 2>&1 || true' EXIT

wine "$WINDOWS_PYTHON" -m pip install \
  --disable-pip-version-check \
  -r requirements.txt \
  pyinstaller

wine "$WINDOWS_PYTHON" -m PyInstaller \
  --clean \
  --noconfirm \
  packaging/tales_of_derision_windows.spec

wineserver -w
