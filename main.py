"""Repository-root launcher used by pygbag web builds.

Pygbag drives the game as a coroutine on the browser's single thread. It looks
for the canonical ``asyncio.run(main())`` entry pattern, so this launcher defines
an async ``main`` that prepares ``sys.path`` and awaits the real game coroutine
in ``src/main.py``. Desktop ``python3 main.py`` runs the same path.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"


async def main() -> None:
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        # Insert first so `import main` below resolves to src/main.py, not this
        # launcher (which runs as __main__, so it never shadows the module name).
        sys.path.insert(0, src_path)

    os.chdir(ROOT_DIR)

    import main as game  # src/main.py

    await game.main()


if __name__ == "__main__":
    # Pygbag runs this file as __main__, so the guard still triggers on the web
    # build while keeping the launcher import-safe for the test suite.
    asyncio.run(main())
