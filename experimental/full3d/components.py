"""Every component in the game, and nothing else.

Components are data. They carry no behaviour, no references to the scene
graph, and no imports beyond ``panda3d.core.Vec3`` -- which is a plain vector
type and does not open a window or touch the GPU. That is deliberate: the
whole simulation in :mod:`sim` runs against these dataclasses with no renderer
present, which is what makes ``headlesstest.py`` possible.

The split that matters most here is :class:`Intent`. Nothing in the movement
or combat code asks "is this the player?" -- it asks what the entity *wants*
to do this frame. The keyboard writes an ``Intent``; :class:`AntBrain` and
:class:`AllyBrain` write an ``Intent``. The Wing Diver flight model would fly
an ant just as happily if you gave one wings.

Units are metres, seconds, and degrees. Panda3D is Z-up, so ``pos.z`` is
altitude and a heading of 0 faces +Y.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from panda3d.core import Vec3


# --------------------------------------------------------------------------
# tags and enums
# --------------------------------------------------------------------------


class Faction(IntEnum):
    """Who shoots whom. Projectiles never damage their own faction."""

    EDF = 0
    BUGS = 1


@dataclass
class FactionTag:
    """Wrapper so a bare :class:`Faction` can live as a component."""

    side: Faction = Faction.EDF


@dataclass
class Player:
    """Marks the one entity the camera follows and the keyboard drives."""


@dataclass
class Dead:
    """Set the frame something's health hits zero.

    Death is a two-step so that the death frame is still visible to every
    other system (the score counter, the renderer's despawn sweep) before
    :class:`~sim.ReaperProcessor` removes the entity.
    """

    timer: float = 0.6


# --------------------------------------------------------------------------
# space and motion
# --------------------------------------------------------------------------


@dataclass
class Transform:
    """Where a thing is and which way it is pointing.

    ``pos`` is the point on the ground between the feet, not the centre of
    the body -- landing code and model placement both get simpler that way.
    """

    pos: Vec3 = field(default_factory=Vec3)
    heading: float = 0.0  # degrees, 0 = +Y, increasing counter-clockwise
    pitch: float = 0.0  # degrees, positive = nose up
    roll: float = 0.0  # degrees, cosmetic bank while airborne


@dataclass
class Velocity:
    """Metres per second in world space. Integrated by the physics pass."""

    vec: Vec3 = field(default_factory=Vec3)


@dataclass
class Body:
    """An upright cylinder, plus the bookkeeping collision leaves behind.

    ``grounded`` is recomputed every physics step; ``ground_z`` remembers what
    it was standing on, so a rooftop and the street are the same thing to
    everything upstream.
    """

    radius: float = 0.6
    height: float = 1.8
    gravity: float = 22.0
    grounded: bool = True
    ground_z: float = 0.0
    airborne_time: float = 0.0
    # Set by the physics pass on the frame of a landing, and consumed by the
    # renderer for the dust puff. Positive = downward speed at impact.
    landed_speed: float = 0.0


@dataclass
class Gait:
    """Walk-cycle phase, advanced from horizontal speed.

    Lives in the simulation rather than the renderer because it is derived
    state with no scene graph in it, and because a headless test can then
    assert that a walking ant actually cycles its legs.
    """

    phase: float = 0.0
    rate: float = 2.2  # cycles per metre travelled
    amplitude: float = 0.0  # 0..1, eased so stopping settles rather than snaps


# --------------------------------------------------------------------------
# the Wing Diver
# --------------------------------------------------------------------------


@dataclass
class Energy:
    """The Wing Diver's single resource: flight, gliding and shots all drink it.

    The rule that gives the class its character is the *overheat*: run the
    meter to exactly zero and ``empty`` latches, flight and gliding are locked
    out entirely, and the recharge runs at a fraction of normal speed until
    the meter is completely full again. Landing with 1% left recovers in a
    moment; landing with 0% is a punishment.

    Note how cheap the glide is next to the thrust. That ratio is the economy
    of the class: climbing is expensive and covering ground is not, so the
    efficient way anywhere is one hard burn followed by a long flat glide.
    """

    maximum: float = 100.0
    current: float = 100.0

    drain_thrust: float = 26.0  # per second of held thrust
    drain_glide: float = 7.0  # per second with the wings out
    regen_ground: float = 45.0  # per second, feet down, not thrusting
    regen_air: float = 12.0  # per second, airborne, not thrusting
    empty_penalty: float = 0.55  # regen multiplier while overheated

    empty: bool = False

    @property
    def fraction(self) -> float:
        return self.current / self.maximum if self.maximum else 0.0

    def spend(self, amount: float) -> bool:
        """Take ``amount`` if it is there. Returns whether the spend happened.

        A spend that lands exactly on zero is allowed and *causes* the
        overheat -- you are always permitted the last drop, you just pay for
        it afterwards.
        """
        if self.empty or amount > self.current:
            return False
        self.current -= amount
        if self.current <= 1e-6:
            self.current = 0.0
            self.empty = True
        return True


@dataclass
class Flight:
    """Tuning and state for jetpack movement: thrust up, or glide across.

    Thrust is an acceleration rather than a set-velocity so that momentum
    carries across a thrust tap, which is what makes EDF flight feel like
    swimming rather than like an elevator.

    The glide is the other half. With the wings out the descent is capped near
    ``glide_fall`` and the horizontal drag drops by most of an order of
    magnitude, so whatever speed you arrived with is speed you keep. It never
    adds height -- it only stops you losing it, which is what separates a
    glide from a hop.
    """

    thrust_accel: float = 34.0
    rise_max: float = 14.0
    air_accel: float = 26.0
    air_max: float = 18.0
    air_drag: float = 0.9  # horizontal damping per second, wings in
    fall_max: float = 30.0

    glide_fall: float = 4.5  # terminal descent with the wings out
    glide_bite: float = 6.0  # how fast a hard fall is eased back to that
    glide_drag: float = 0.05  # horizontal damping per second, wings out
    glide_accel: float = 15.0  # air control while gliding: committed to a line
    glide_gravity: float = 0.2  # gravity multiplier while the wings are out

    # live state
    thrusting: bool = False
    gliding: bool = False


@dataclass
class Walker:
    """Ground locomotion for anything without a jetpack."""

    accel: float = 30.0
    max_speed: float = 7.0
    friction: float = 10.0
    turn_rate: float = 360.0  # degrees per second toward the desired heading
    jump_speed: float = 0.0  # ants leap; soldiers do not


@dataclass
class Intent:
    """What an entity is trying to do this frame, whoever decided it.

    ``move`` is in the entity's *aim* frame: ``move.y`` is forward along
    ``aim_yaw`` and ``move.x`` is to its right. Length is clamped to 1 by the
    producers so that diagonals are not faster.
    """

    move_x: float = 0.0
    move_y: float = 0.0
    aim_yaw: float = 0.0
    aim_pitch: float = 0.0
    thrust: bool = False
    glide: bool = False
    fire: bool = False

    def clear(self) -> None:
        self.move_x = self.move_y = 0.0
        self.thrust = self.glide = self.fire = False


# --------------------------------------------------------------------------
# combat
# --------------------------------------------------------------------------


@dataclass
class Health:
    maximum: float = 100.0
    current: float = 100.0
    hurt_flash: float = 0.0  # seconds remaining of the white damage flash

    def damage(self, amount: float) -> None:
        self.current = max(0.0, self.current - amount)
        self.hurt_flash = 0.12


@dataclass
class Weapon:
    """A projectile launcher. Melee is a weapon with a very short range.

    ``energy_cost`` is what ties the Wing Diver's gun to her jetpack: firing
    the lance is flight you are choosing not to take.
    """

    damage: float = 30.0
    speed: float = 90.0
    cooldown: float = 0.28
    range: float = 140.0
    spread: float = 0.0  # degrees of random cone
    energy_cost: float = 0.0
    melee: bool = False
    kind: str = "lance"  # renderer picks the projectile model from this

    timer: float = 0.0  # counts down to zero, then the weapon is ready
    muzzle_flash: float = 0.0


@dataclass
class Projectile:
    """A point that travels in a straight line and hurts one thing once."""

    damage: float = 10.0
    faction: Faction = Faction.EDF
    life: float = 3.0
    radius: float = 0.5
    owner: int = -1


@dataclass
class AntBrain:
    """Ant behaviour: close the distance, then bite. Spitters shoot first.

    Ants do not path around buildings -- they walk into them and slide along,
    which is both period-correct and, in practice, how the real ones behave.
    """

    target: int = -1
    retarget: float = 0.0  # seconds until the next target search
    sight: float = 120.0
    bite_range: float = 2.6
    spitter: bool = False
    leap_cd: float = 0.0


@dataclass
class AllyBrain:
    """A grunt: hold near the rally point, shoot the nearest ant, back off.

    ``courage`` decides how close it is willing to get before it stops
    advancing, which is enough variation to make a squad look like people.
    """

    target: int = -1
    retarget: float = 0.0
    sight: float = 90.0
    standoff: float = 18.0
    courage: float = 1.0
    rally: Vec3 = field(default_factory=Vec3)


# --------------------------------------------------------------------------
# scenery and presentation
# --------------------------------------------------------------------------


@dataclass
class Building:
    """A static box the world is made of. Collision reads it as an AABB.

    Stored as half-extents around ``Transform.pos``, with the box sitting on
    the ground: it spans ``z`` from 0 to ``height``.
    """

    half_x: float = 6.0
    half_y: float = 6.0
    height: float = 20.0
    style: int = 0  # picks a texture/colour in the renderer
    rubble: bool = False


@dataclass
class Renderable:
    """A request for a model, with no idea how models work.

    The renderer keeps the entity-to-``NodePath`` map; the simulation only
    ever states a ``kind`` and a tint. Nothing here is read by any gameplay
    system, so a headless run simply never looks at it.
    """

    kind: str = "soldier"
    scale: float = 1.0
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    seed: int = 0


@dataclass
class Lifetime:
    """Seconds until the entity deletes itself. Used by effects and tracers."""

    remaining: float = 1.0
