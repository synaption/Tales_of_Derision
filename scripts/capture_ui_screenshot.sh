#!/usr/bin/env bash

set -euo pipefail

out_path="${1:-data/screenshots/ui_frame.png}"
save_file="${2:-src/data/saves/default_save.json}"

cd "$(dirname "$0")/.."
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
python3 src/main.py --save_file "$save_file" --screenshot "$out_path"
echo "Screenshot written to $out_path"
