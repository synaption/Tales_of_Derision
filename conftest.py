"""Test bootstrap: make ``src`` win over the repo root on ``sys.path``.

``python3 -m pytest`` puts the repo root (cwd) on ``sys.path`` first, where the
pygbag launcher ``main.py`` lives. That shadows ``src/main.py``, so imports like
``from main import _setup_world`` would resolve to the launcher. Prepending
``src`` here makes ``main`` resolve to the game module during tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
src_path = str(SRC_DIR)
if src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)
