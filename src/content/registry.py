"""Prefab registry: declarative entity templates + a ``spawn`` that builds them.

A ``PrefabDef`` reads like data -- a glyph, a name, some kits, a few extra
components -- so adding or tuning a creature/feature is editing data. The **same**
registry also accepts a factory callable for the Python path (a prefab whose
assembly needs real code). Both resolve through ``build_components``/``spawn``.

See ``wiki/Content-and-Mods.md``.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import esper

from components import Name, Position, Renderable
from content.kits import build_kit

RGB = tuple[int, int, int]

# A kit spec is either "kit_name" (default kwargs) or ("kit_name", {kwargs}).
KitSpec = "str | tuple[str, dict]"
# A factory builds an entity's full component list (including Position).
PrefabFactory = Callable[[int, int, dict], list]


@dataclass
class PrefabDef:
    """A declarative entity template. ``kits`` compose reusable component bundles;
    ``components`` are extra literal component instances (deep-copied per spawn) or
    zero-arg callables that build one. Only ``id`` and ``glyph`` are required."""
    id: str
    glyph: str
    name: str | None = None
    fg: RGB | None = None
    bg: RGB | None = None
    kits: list = field(default_factory=list)        # list[KitSpec]
    components: list = field(default_factory=list)   # extra components / factories
    tags: tuple[str, ...] = field(default_factory=tuple)


_PREFABS: dict[str, PrefabDef | PrefabFactory] = {}


def register_prefab(defn: PrefabDef | PrefabFactory, prefab_id: str | None = None) -> None:
    """Register a ``PrefabDef`` (data path) or a factory callable (Python path).
    Factories need an explicit ``prefab_id`` since they carry no ``id`` field."""
    key = prefab_id if prefab_id is not None else getattr(defn, "id", None)
    if key is None:
        raise ValueError("factory prefabs require an explicit prefab_id")
    _PREFABS[key] = defn


def has_prefab(prefab_id: str) -> bool:
    return prefab_id in _PREFABS


def get_prefab(prefab_id: str) -> PrefabDef | PrefabFactory | None:
    return _PREFABS.get(prefab_id)


def prefab_ids() -> list[str]:
    return sorted(_PREFABS)


def _instantiate(spec: Any):
    """A prefab's extra component may be given three ways, all yielding a fresh,
    unshared component so entities never alias a mutable one:
      * a component **class** (``Tree``)            -> ``Tree()``
      * a zero-arg **factory** callable             -> ``factory()``
      * a component **instance** (``Inventory([])``) -> deep-copied per spawn
    """
    if isinstance(spec, type):
        return spec()
    if callable(spec):
        return spec()
    return copy.deepcopy(spec)


def _kit_spec(spec) -> tuple[str, dict]:
    if isinstance(spec, tuple):
        name, kwargs = spec
        return name, dict(kwargs)
    return spec, {}


def build_components(prefab_id: str, x: int, y: int, **overrides) -> list:
    """Build the full component list for ``prefab_id`` at ``(x, y)`` -- pure, with no
    ECS side effects, so it is trivially unit-testable. ``overrides`` may set
    ``glyph``/``name``/``fg``/``bg`` and pass ``extra=[...]`` components."""
    defn = _PREFABS.get(prefab_id)
    if defn is None:
        raise KeyError(f"unknown prefab {prefab_id!r}")

    if not isinstance(defn, PrefabDef):
        return list(defn(x, y, overrides))  # factory (Python path)

    glyph = overrides.get("glyph", defn.glyph)
    fg = overrides.get("fg", defn.fg)
    bg = overrides.get("bg", defn.bg)
    name = overrides.get("name", defn.name)

    comps: list = [Position(x, y), Renderable(glyph, fg=fg, bg=bg)]
    if name is not None:
        comps.append(Name(name))
    for spec in defn.kits:
        kit_name, kwargs = _kit_spec(spec)
        comps.extend(build_kit(kit_name, kwargs))
    for spec in defn.components:
        comps.append(_instantiate(spec))
    comps.extend(overrides.get("extra", []))
    return comps


def spawn(prefab_id: str, x: int, y: int, **overrides) -> int:
    """Create an entity from ``prefab_id`` at ``(x, y)`` and return its id."""
    return esper.create_entity(*build_components(prefab_id, x, y, **overrides))
