#!/usr/bin/env bash
#
# Convenience: build the web (pygbag) version with the existing build script,
# then serve it locally. Frees the port first and stops its own server cleanly
# on Ctrl+C / exit.
#
# pygbag rewrites its boot fetches to http://localhost:8000, so the port
# defaults to 8000. Override with: ./online.sh <port>

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8000}"

# Kill anything already bound to the port before building/serving.
fuser -k "${PORT}/tcp" 2>/dev/null || true

# Build using the existing build script.
bash "${ROOT_DIR}/scripts/build_pygbag.sh"

# Serve the built bundle.
python3 -m http.server "${PORT}" --directory "${ROOT_DIR}/build/web" &
SERVER_PID=$!

# On exit or Ctrl+C, stop our server and free the port.
trap 'kill "${SERVER_PID}" 2>/dev/null || true; fuser -k "${PORT}/tcp" 2>/dev/null || true' EXIT INT TERM

echo "Serving at http://localhost:${PORT}  (Ctrl+C to stop)"
wait "${SERVER_PID}"
