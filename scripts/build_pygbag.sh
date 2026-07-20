#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="${ROOT_DIR}/pygbag-stage"
OUTPUT_DIR="${ROOT_DIR}/build/web"
PYGBAG_CDN="https://pygame-web.github.io/cdn/0.9.3/"
BROWSERFS_FALLBACK_URL="https://pygame-web.github.io/pygbag/0.0/browserfs.min.js"
PYGAME_WHEEL_URL="https://pygame-web.github.io/cdn/cp312/pygame_ce-2.5.7-cp312-cp312-wasm32_bi_emscripten.whl"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

cp "${ROOT_DIR}/main.py" "${STAGE_DIR}/main.py"
cp -R "${ROOT_DIR}/src" "${STAGE_DIR}/src"
cp -R "${ROOT_DIR}/audio" "${STAGE_DIR}/audio"
cp -R "${ROOT_DIR}/gfx" "${STAGE_DIR}/gfx"

# Tests are not needed in the web runtime bundle.
rm -rf "${STAGE_DIR}/src/tests"

# Trim redundant music sources from the web bundle. The web runtime prefers
# .ogg (44.1kHz, small, no pygbag resampling), so when a track has an .ogg we
# drop its larger .wav/.mp3 siblings to keep the download small. Tracks without
# an .ogg keep whatever formats exist as a fallback.
python3 - "${STAGE_DIR}/audio/music" <<'PY'
from pathlib import Path
import sys

music_dir = Path(sys.argv[1])
if music_dir.is_dir():
	redundant_suffixes = {".wav", ".mp3", ".flac", ".m4a"}
	stems_with_ogg = {path.stem for path in music_dir.glob("*.ogg")}
	for path in sorted(music_dir.iterdir()):
		if (
			path.is_file()
			and path.suffix.lower() in redundant_suffixes
			and path.stem in stems_with_ogg
		):
			path.unlink()
			print(f"web bundle: dropped redundant music source {path.name}")
PY

# Bundle esper directly into the web app so browser runtime can import it.
python3 - "${STAGE_DIR}" <<'PY'
from pathlib import Path
import shutil
import sys

stage_dir = Path(sys.argv[1])

try:
	import esper
except Exception as exc:
	raise SystemExit(
		"Missing dependency 'esper'. Install it first with: python3 -m pip install --user esper"
	) from exc

esper_pkg_dir = Path(esper.__file__).resolve().parent
target_dir = stage_dir / "esper"
if target_dir.exists():
	shutil.rmtree(target_dir)
shutil.copytree(esper_pkg_dir, target_dir)
PY

# Bundle pygame-ce wasm wheel into the staged app so browser startup does not
# depend on runtime package bootstrap.
python3 - "${STAGE_DIR}" "${PYGAME_WHEEL_URL}" <<'PY'
from pathlib import Path
import sys
import urllib.request
import zipfile

stage_dir = Path(sys.argv[1])
wheel_url = sys.argv[2]
wheel_name = wheel_url.rsplit("/", 1)[-1]
wheel_path = stage_dir / wheel_name

urllib.request.urlretrieve(wheel_url, wheel_path)
with zipfile.ZipFile(wheel_path) as zf:
	zf.extractall(stage_dir)
wheel_path.unlink(missing_ok=True)
PY

python3 -m pygbag \
	--build \
	--ume_block 0 \
	--no_opt \
	--disable-sound-format-error \
	--cdn "${PYGBAG_CDN}" \
	"${STAGE_DIR}"

rm -rf "${ROOT_DIR}/build"
mkdir -p "${ROOT_DIR}/build"
cp -R "${STAGE_DIR}/build/web" "${OUTPUT_DIR}"

# Pygbag's 0.9.3 CDN currently misses browserfs.min.js; vendor a known-good copy locally.
python3 - "${OUTPUT_DIR}" "${BROWSERFS_FALLBACK_URL}" <<'PY'
from pathlib import Path
import sys
import urllib.request

output_dir = Path(sys.argv[1])
browserfs_url = sys.argv[2]

browserfs_path = output_dir / "browserfs.min.js"
urllib.request.urlretrieve(browserfs_url, browserfs_path)

index_path = output_dir / "index.html"
html = index_path.read_text(encoding="utf-8")

for token in (
	"https://pygame-web.github.io/cdn/0.9.3//browserfs.min.js",
	"https://pygame-web.github.io/cdn/0.9.3/browserfs.min.js",
	"https://pygame-web.github.io/pygbag/0.0//browserfs.min.js",
	"https://pygame-web.github.io/pygbag/0.0/browserfs.min.js",
):
	html = html.replace(token, "./browserfs.min.js")

# The game loop is long-running and may never return from shell.source(main),
# so hide the loading overlay before entering main.
needle = "await shell.source(main, callback=ui_callback)"
replacement = "platform.window.infobox.style.display = \"none\"\n\n    await shell.source(main, callback=ui_callback)"
html = html.replace(needle, replacement)

index_path.write_text(html, encoding="utf-8")
PY

# Mirror pygame-ce wasm wheel locally so static localhost serving resolves /cdn/cp312 requests.
python3 - "${OUTPUT_DIR}" "${PYGAME_WHEEL_URL}" <<'PY'
from pathlib import Path
import sys
import urllib.request

output_dir = Path(sys.argv[1])
wheel_url = sys.argv[2]
wheel_name = wheel_url.rsplit("/", 1)[-1]

wheel_dir = output_dir / "cdn" / "cp312"
wheel_dir.mkdir(parents=True, exist_ok=True)
wheel_path = wheel_dir / wheel_name
urllib.request.urlretrieve(wheel_url, wheel_path)
PY

rm -rf "${STAGE_DIR}"

echo "Web build ready at ${OUTPUT_DIR}"
echo "Local preview: use python3 -m http.server --directory build/web 8000"
echo "Note: pygbag runtime package fetches are rewritten to http://localhost:8000/cdn/... during local preview."