"""The mod seam: load all content into the registries.

``load_all_content()`` imports the core content modules (each self-registers on
import) and then scans ``content/data/*.json`` for data-authored prefabs, registering
them through the same ``register_prefab``. That's the two authoring paths in one call:

* **drop a Python module** under ``content/`` (or a user ``mods/`` dir) that calls
  ``register_prefab`` / ``register_effect`` / ``register_item`` and add it to
  ``_CORE_MODULES`` -- the developer / new-behaviour path;
* **drop a readable JSON file** under ``content/data/`` -- the non-coder / tune-an-
  existing-feature path.

Called once at startup (``main``) and lazily by ``worldgen`` so any code path that
spawns prefabs has them registered.
"""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from content.registry import PrefabDef, register_prefab

# Core content modules, imported for their import-time registration side effect.
_CORE_MODULES = (
    "content.items",
    "content.effects",
    "content.creatures",
    "content.flora",
    "content.features",
)

_DATA_DIR = Path(__file__).resolve().parent / "data"

_loaded = False


def _as_rgb(value):
    """JSON arrays -> RGB tuples; leave ``None`` alone."""
    return tuple(value) if isinstance(value, list) else value


def _prefab_from_json(raw: dict) -> PrefabDef:
    """Build a ``PrefabDef`` from a JSON object. ``kits`` entries are ``[name, {..}]``
    or ``"name"``; component *extras* are the Python path only, so JSON prefabs
    compose from kits (which covers adding a variant of an existing creature)."""
    kits = [tuple(k) if isinstance(k, list) else k for k in raw.get("kits", [])]
    return PrefabDef(
        id=raw["id"],
        glyph=raw["glyph"],
        name=raw.get("name"),
        fg=_as_rgb(raw.get("fg")),
        bg=_as_rgb(raw.get("bg")),
        kits=kits,
        tags=tuple(raw.get("tags", ())),
    )


def _load_json_prefabs() -> int:
    if not _DATA_DIR.is_dir():
        return 0
    count = 0
    for path in sorted(_DATA_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        entries = payload if isinstance(payload, list) else payload.get("prefabs", [])
        for raw in entries:
            register_prefab(_prefab_from_json(raw))
            count += 1
    return count


def load_all_content(force: bool = False) -> None:
    """Register all core + data-authored content. Idempotent; cheap on repeat calls
    (skips the disk scan) unless ``force=True``."""
    global _loaded
    if _loaded and not force:
        return
    for module in _CORE_MODULES:
        import_module(module)
    _load_json_prefabs()
    _loaded = True
