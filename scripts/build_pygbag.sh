#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="${ROOT_DIR}/pygbag-stage"
OUTPUT_DIR="${ROOT_DIR}/build/web"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

cp "${ROOT_DIR}/main.py" "${STAGE_DIR}/main.py"
cp -R "${ROOT_DIR}/src" "${STAGE_DIR}/src"
cp -R "${ROOT_DIR}/audio" "${STAGE_DIR}/audio"
cp -R "${ROOT_DIR}/gfx" "${STAGE_DIR}/gfx"

# Tests are not needed in the web runtime bundle.
rm -rf "${STAGE_DIR}/src/tests"

python3 -m pygbag --build --no_opt --disable-sound-format-error "${STAGE_DIR}"

rm -rf "${ROOT_DIR}/build"
mkdir -p "${ROOT_DIR}/build"
cp -R "${STAGE_DIR}/build/web" "${OUTPUT_DIR}"
rm -rf "${STAGE_DIR}"

echo "Web build ready at ${OUTPUT_DIR}"