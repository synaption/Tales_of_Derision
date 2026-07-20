"""Repository-root launcher used by pygbag web builds."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"


def main() -> None:
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    os.chdir(ROOT_DIR)
    runpy.run_path(str(SRC_DIR / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()