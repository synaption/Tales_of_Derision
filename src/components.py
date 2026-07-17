"""ECS components. Pure data, no behaviour.

Components stay renderer-agnostic: a Renderable holds a glyph, not a
curses/pygame handle. Systems and renderers translate these into pixels.
"""
from dataclasses import dataclass


@dataclass
class Position:
    x: int
    y: int


@dataclass
class Renderable:
    glyph: str


@dataclass
class Player:
    """Tag component: marks the entity the input controls."""


@dataclass
class Name:
    value: str
