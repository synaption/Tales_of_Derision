"""Cramigula -- a Wing Diver, a city, and rather too many ants.

    python3 experimental/full3d/cramigula.py
    python3 experimental/full3d/cramigula.py --seed 7 --ants 24

An Earth Defence Force flight-class movement model, rendered as if it were
1998, on top of an entity-component system. The three concerns live in
separate modules and only meet here:

    worldgen.new_game()  ->  sim.Sim      the game, with no renderer at all
    ps1.Ps1Pipeline      ->  the look     320x240, snapped, dithered, foggy
    render.attach()      ->  the bridge   three processors that read the Sim

**Controls**

===============  =========================================================
Mouse            aim; the screen centre is exactly where the lance goes
W A S D          move, relative to where you are looking
Space (hold)     thrust -- drains energy continuously while held
Shift (hold)     glide -- wings out: the fall slows and the speed keeps
Left mouse       fire the lance -- costs energy too, so it costs altitude
R                restart with a fresh city
F1               toggle the vertex snapping, to see what it is doing
Escape           release the mouse; again to quit
===============  =========================================================

The whole class is one meter. Flight, gliding and shooting all drink from it,
it only refills quickly with your feet on the ground, and if you let it hit
exactly zero it locks out and refills at half speed until it is completely
full. Everything interesting about playing a Wing Diver is the arithmetic of
not letting that happen.

The glide is what makes the arithmetic work. Thrust costs 26 a second and
gliding costs 7, and a glide never gains a millimetre of height -- so the way
to cross the city is one hard burn upward followed by a long flat descent,
not a jetpack held down the whole way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from panda3d.core import GraphicsWindow, WindowProperties, loadPrcFileData

# Must be set before ShowBase is constructed.
loadPrcFileData("", "window-title Cramigula")
loadPrcFileData("", "win-size 960 720")
loadPrcFileData("", "framebuffer-multisample 0")
loadPrcFileData("", "multisamples 0")
loadPrcFileData("", "textures-power-2 none")
loadPrcFileData("", "audio-library-name null")

import esper  # noqa: E402
from direct.showbase.ShowBase import ShowBase  # noqa: E402

import ps1  # noqa: E402
import render  # noqa: E402
import worldgen  # noqa: E402
from components import Dead, Intent  # noqa: E402
from sim import clamp  # noqa: E402

MOUSE_SENSITIVITY = 0.13  # degrees per pixel of mouse travel


class Cramigula(ShowBase):
    """The application: a window, an input map, and one task.

    Deliberately thin. This class owns nothing but the keyboard state and the
    two mouse angles; the moment it has read those it writes them into the
    player's :class:`~components.Intent` and hands over to
    :meth:`sim.Sim.step`, which is the same call the headless test makes.
    """

    def __init__(self, seed: int = 1234, allies: int = 8, ants: int = 14) -> None:
        super().__init__()
        self.disableMouse()  # no default trackball camera; we drive it

        self.seed = seed
        self.allies = allies
        self.ants = ants

        self.pipeline = ps1.Ps1Pipeline(self)
        self.keys: dict[str, bool] = {}
        self.yaw = 0.0
        self.pitch = 8.0
        self.mouse_captured = False
        self._mouse_settled = False
        self.snapping = True

        self._bind_keys()
        self._start_game()
        self._capture_mouse(True)

        self.taskMgr.add(self._update, "cramigula-update")

    # -- setup -------------------------------------------------------------

    def _bind_keys(self) -> None:
        """Held keys go in a dict; one-shot keys get their own handler."""
        for key in ("w", "a", "s", "d", "space", "shift", "control", "mouse1", "mouse3"):
            self.accept(key, self._press, [key])
            self.accept(f"{key}-up", self._release, [key])
        self.accept("r", self._start_game)
        self.accept("f1", self._toggle_snapping)
        self.accept("escape", self._escape)

    def _press(self, key: str) -> None:
        self.keys[key] = True

    def _release(self, key: str) -> None:
        self.keys[key] = False

    def _start_game(self) -> None:
        """Throw the old world away and generate a new one.

        Each :class:`~sim.Sim` gets its own esper World, so the previous one's
        entities and processors simply stop being the active context -- there
        is nothing to unwind here but the scene graph it left behind.
        """
        if hasattr(self, "view"):
            self.view.root.removeNode()
            self.view.ground.removeNode()
            self.view.city.removeNode()

        self.sim = worldgen.new_game(seed=self.seed, allies=self.allies, ants=self.ants)
        self.view = render.attach(self, self.sim, self.pipeline)
        self.yaw, self.pitch = 0.0, 8.0
        self.seed += 1  # the next R gives a different city

    # -- input -------------------------------------------------------------

    def _capture_mouse(self, capture: bool) -> None:
        """Hide or restore the cursor, if there is a cursor to hide.

        Guarded on there being a real window: run under ``window-type
        offscreen`` and ``base.win`` is a ``GraphicsBuffer``, which has no
        cursor and no ``requestProperties``. That is not a mode the game ships
        in, but it is the mode the tests drive this class in.
        """
        if isinstance(self.win, GraphicsWindow):
            props = WindowProperties()
            props.setCursorHidden(capture)
            self.win.requestProperties(props)
        self.mouse_captured = capture
        self._mouse_settled = False

    def _escape(self) -> None:
        """First press hands the mouse back; a second one quits."""
        if self.mouse_captured:
            self._capture_mouse(False)
        else:
            self.userExit()

    def _toggle_snapping(self) -> None:
        """Turn the vertex grid off, so it is obvious what it was doing.

        Setting the snap resolution to something enormous makes the rounding a
        no-op without needing a second shader; the frame is still 320x240 and
        still dithered, only the geometry stops shimmering.
        """
        self.snapping = not self.snapping
        resolution = (float(ps1.RES_X), float(ps1.RES_Y)) if self.snapping else (1e5, 1e5)
        self.render.setShaderInput("jitterRes", resolution)

    def _read_mouse(self) -> None:
        """Relative mouse look, by recentring the pointer every frame.

        Panda's ``M_relative`` mouse mode is not reliable on every X11 stack
        (WSLg among them) and recentring works everywhere. The first frame
        after a capture is discarded, because the jump from wherever the
        cursor happened to be would spin the camera.
        """
        if not self.mouse_captured or not isinstance(self.win, GraphicsWindow):
            return
        if not self.win.hasPointer(0):
            return
        pointer = self.win.getPointer(0)
        if not pointer.getInWindow():
            return
        props = self.win.getProperties()
        cx, cy = props.getXSize() // 2, props.getYSize() // 2
        dx, dy = pointer.getX() - cx, pointer.getY() - cy
        moved = self.win.movePointer(0, cx, cy)
        if moved and self._mouse_settled:
            self.yaw -= dx * MOUSE_SENSITIVITY
            self.pitch = clamp(self.pitch - dy * MOUSE_SENSITIVITY, -85.0, 85.0)
        self._mouse_settled = moved

    def _write_intent(self) -> None:
        """Translate the keyboard into the player's :class:`Intent`.

        This is the entire coupling between the window and the simulation. An
        ant's brain writes the same six fields, and nothing downstream can
        tell which of the two produced them.
        """
        if self.sim.player < 0 or not esper.entity_exists(self.sim.player):
            return
        intent = esper.component_for_entity(self.sim.player, Intent)
        intent.clear()
        intent.aim_yaw = self.yaw
        intent.aim_pitch = self.pitch
        if esper.has_component(self.sim.player, Dead):
            return

        keys = self.keys
        intent.move_x = (1.0 if keys.get("d") else 0.0) - (1.0 if keys.get("a") else 0.0)
        intent.move_y = (1.0 if keys.get("w") else 0.0) - (1.0 if keys.get("s") else 0.0)
        intent.thrust = bool(keys.get("space"))
        intent.glide = bool(keys.get("shift"))
        intent.fire = bool(keys.get("mouse1")) or bool(keys.get("control"))

    # -- the loop ----------------------------------------------------------

    def _update(self, task):
        dt = self.clock.getDt()
        self._read_mouse()
        self._write_intent()
        self.sim.step(dt)
        return task.cont


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cramigula: EDF flight class, PS1 rendering.")
    parser.add_argument("--seed", type=int, default=1234, help="city and spawn seed")
    parser.add_argument("--allies", type=int, default=8, help="EDF grunts on your side")
    parser.add_argument("--ants", type=int, default=14, help="ants in the opening patrol")
    args = parser.parse_args(argv)

    Cramigula(seed=args.seed, allies=args.allies, ants=args.ants).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
