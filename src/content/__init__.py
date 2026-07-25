"""Content & mod layer: the hybrid registry that all game content flows through.

Two authoring styles feed one set of registries (see ``wiki/Content-and-Mods.md``):

* **data path** -- register a declarative ``PrefabDef`` / ``ItemDef`` / ``EffectDef``
  (readable, editable, JSON-able) to add or tune a *variant* of an existing feature;
* **python path** -- register a factory callable / effect handler to add genuinely
  new behaviour, exactly how the core game adds features.

The core game's own content registers through these same registries at startup via
``content.loader.load_all_content``. This package sits at layer 2 of the import DAG:
it depends only on ``components``/``game_map`` and never imports ``systems``/``main``.
"""
