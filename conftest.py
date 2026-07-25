"""Test bootstrap: put ``src`` first on ``sys.path``.

The game modules live in ``src`` (``main``, ``game_map``, ``systems`` ...), so
tests import them as top-level modules (``from main import _setup_world``).
Prepending ``src`` here makes those imports resolve during ``python3 -m pytest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
src_path = str(SRC_DIR)
if src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)
