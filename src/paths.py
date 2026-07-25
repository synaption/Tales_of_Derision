"""Filesystem locations for source checkouts and frozen executables."""
from __future__ import annotations

import os
from pathlib import Path
import sys


APP_NAME = "TalesOfDerision"


def resource_root() -> Path:
    """Directory containing read-only assets bundled with the application."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)
    return Path(__file__).resolve().parent.parent


def writable_data_root() -> Path:
    """Persistent save/config directory, outside PyInstaller's temporary tree."""
    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home()
        return base / APP_NAME
    return resource_root() / "src" / "data"
