"""A small entity-component-system used by Tales of Derision.

Entities are integer ids, components are arbitrary Python objects keyed by their
type, and processors are objects with a ``process`` method.  The module-level API
is intentional: the game has one active simulation context at a time, while named
worlds keep tests and future scenes isolated without passing a World everywhere.

This module owns only storage, queries, and processor ordering.  Components remain
plain dataclasses and all game behaviour remains in the game's processors.
"""

from __future__ import annotations

from itertools import count
from typing import Any, TypeVar


Component = TypeVar("Component")


class Processor:
    """Base class for systems run by :func:`process`.

    Subclassing is optional at runtime, but provides a shared type and a priority
    attribute for the processor registry.
    """

    priority: int = 0

    def process(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError


def _fresh_world() -> tuple[Any, ...]:
    return (count(1), {}, {}, set(), [], {})


# These stores are deliberately straightforward and visible.  ``spatial.py`` uses
# their O(1) population counts to incrementally synchronize its map index.
_entity_count, _components, _entities, _dead_entities, _processors, _processors_dict = _fresh_world()
_worlds: dict[str, tuple[Any, ...]] = {}
current_world = "default"


def _save_current_world() -> None:
    _worlds[current_world] = (
        _entity_count,
        _components,
        _entities,
        _dead_entities,
        _processors,
        _processors_dict,
    )


def switch_world(name: str) -> None:
    """Activate a named, isolated ECS context, creating it when necessary."""
    global current_world, _entity_count, _components, _entities, _dead_entities
    global _processors, _processors_dict

    if name == current_world:
        return
    _save_current_world()
    state = _worlds.get(name)
    if state is None:
        state = _fresh_world()
        _worlds[name] = state
    (
        _entity_count,
        _components,
        _entities,
        _dead_entities,
        _processors,
        _processors_dict,
    ) = state
    current_world = name


def delete_world(name: str) -> None:
    """Delete an inactive named context."""
    if name == current_world:
        raise PermissionError("The active world context cannot be deleted")
    del _worlds[name]


def create_entity(*components: Component) -> int:
    """Create an entity, attach the supplied components, and return its id."""
    entity = next(_entity_count)
    entity_components: dict[type, Any] = {}
    for component in components:
        component_type = type(component)
        _components.setdefault(component_type, set()).add(entity)
        entity_components[component_type] = component
    _entities[entity] = entity_components
    return entity


def delete_entity(entity: int, immediate: bool = False) -> None:
    """Delete now, or mark an entity for deletion at the next process call."""
    if immediate:
        _delete_now(entity)
    else:
        if entity not in _entities:
            raise KeyError(entity)
        _dead_entities.add(entity)


def _delete_now(entity: int) -> None:
    entity_components = _entities.pop(entity)
    _dead_entities.discard(entity)
    for component_type in entity_components:
        entities = _components[component_type]
        entities.discard(entity)
        if not entities:
            del _components[component_type]


def clear_dead_entities() -> None:
    """Finalize all deferred entity deletions."""
    for entity in tuple(_dead_entities):
        if entity in _entities:
            _delete_now(entity)
    _dead_entities.clear()


def clear_database() -> None:
    """Remove all entities and components, retaining registered processors."""
    global _entity_count
    _entity_count = count(1)
    _components.clear()
    _entities.clear()
    _dead_entities.clear()


def entity_exists(entity: int) -> bool:
    return entity in _entities and entity not in _dead_entities


def add_component(
    entity: int,
    component: Component,
    type_alias: type[Component] | None = None,
) -> None:
    component_type = type_alias or type(component)
    _components.setdefault(component_type, set()).add(entity)
    _entities[entity][component_type] = component


def remove_component(entity: int, component_type: type[Component]) -> Component:
    entities = _components[component_type]
    entities.remove(entity)
    if not entities:
        del _components[component_type]
    return _entities[entity].pop(component_type)


def has_component(entity: int, component_type: type[Component]) -> bool:
    return component_type in _entities[entity]


def component_for_entity(entity: int, component_type: type[Component]) -> Component:
    return _entities[entity][component_type]


def get_component(component_type: type[Component]) -> list[tuple[int, Component]]:
    """Return entity/component pairs for one component type."""
    return [
        (entity, _entities[entity][component_type])
        for entity in _components.get(component_type, ())
        if entity not in _dead_entities
    ]


def get_components(*component_types: type[Any]) -> list[tuple[int, tuple[Any, ...]]]:
    """Return entities containing every requested type and their components."""
    if not component_types:
        return []
    populations = [_components.get(component_type, set()) for component_type in component_types]
    if any(not population for population in populations):
        return []
    candidates = min(populations, key=len)
    return [
        (entity, tuple(_entities[entity][component_type] for component_type in component_types))
        for entity in candidates
        if entity not in _dead_entities
        and all(entity in population for population in populations)
    ]


def add_processor(processor: Processor, priority: int = 0) -> None:
    """Register a processor; higher priorities run first, ties keep insertion order."""
    processor.priority = priority
    _processors.append(processor)
    _processors.sort(key=lambda item: item.priority, reverse=True)
    _processors_dict[type(processor)] = processor


def get_processor(processor_type: type[Processor]) -> Processor | None:
    return _processors_dict.get(processor_type)


def process(*args: Any, **kwargs: Any) -> None:
    """Finalize deferred deletions, then run every registered processor."""
    clear_dead_entities()
    for processor in tuple(_processors):
        processor.process(*args, **kwargs)
