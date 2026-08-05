//! Every component in the game, and nothing else.
//!
//! Components are data. They carry no behaviour, no handles to meshes or
//! materials, and this module imports nothing but `bevy_ecs` and `bevy_math`
//! -- neither of which can open a window. That is deliberate: the whole
//! simulation in [`crate::sim`] runs against these structs with no renderer
//! present, which is what makes `tests/headless.rs` instant.
//!
//! The split that matters most is [`Intent`]. Nothing in the movement or
//! combat code asks "is this the player?" -- it asks what the entity *wants*
//! to do this frame. The keyboard writes an `Intent`; [`AntBrain`] and
//! [`AllyBrain`] write an `Intent`. The Wing Diver flight model would fly an
//! ant just as happily if you gave one wings.
//!
//! # Coordinates
//!
//! Metres, seconds and degrees, in Bevy's right-handed Y-up space: `pos.y` is
//! altitude and the ground is the XZ plane. A heading of 0 faces -Z, which is
//! Bevy's forward, and positive heading turns counter-clockwise seen from
//! above.
//!
//! [`Pose`] rather than Bevy's own `Transform` is not an oversight. The
//! simulation wants a position and three angles in degrees; a `Transform`
//! wants a quaternion, and converting between them sixty times a second so
//! that the physics can read back a heading it wrote itself is work with no
//! product. `render::sync_transforms` does the conversion once, at the seam,
//! in the direction it is actually needed.

use bevy_ecs::prelude::*;
use bevy_math::Vec3;

// --------------------------------------------------------------------------
// tags and enums
// --------------------------------------------------------------------------

/// Who shoots whom. Projectiles never damage their own faction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Faction {
    Edf,
    Bugs,
}

impl Faction {
    /// The other one.
    pub fn opposing(self) -> Faction {
        match self {
            Faction::Edf => Faction::Bugs,
            Faction::Bugs => Faction::Edf,
        }
    }
}

/// Wrapper so a bare [`Faction`] can live as a component.
#[derive(Component, Debug, Clone, Copy)]
pub struct FactionTag(pub Faction);

/// Marks the one entity the camera follows and the keyboard drives.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Player;

/// Added the frame something's health hits zero.
///
/// Death is a two-step so that the death frame is still visible to every
/// other system (the score counter, the renderer's despawn sweep) before
/// [`crate::sim::reap`] removes the entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct Dead {
    pub timer: f32,
}

impl Default for Dead {
    fn default() -> Self {
        Dead { timer: 0.6 }
    }
}

// --------------------------------------------------------------------------
// space and motion
// --------------------------------------------------------------------------

/// Where a thing is and which way it is pointing.
///
/// `pos` is the point on the ground between the feet, not the centre of the
/// body -- landing code and model placement both get simpler that way.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Pose {
    pub pos: Vec3,
    /// Degrees. 0 faces -Z, increasing counter-clockwise seen from above.
    pub heading: f32,
    /// Degrees, positive = nose up.
    pub pitch: f32,
    /// Degrees, cosmetic bank while airborne.
    pub roll: f32,
}

impl Pose {
    pub fn at(pos: Vec3) -> Self {
        Pose {
            pos,
            ..Default::default()
        }
    }
}

/// Metres per second in world space. Integrated by the physics pass.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Velocity(pub Vec3);

/// An upright cylinder, plus the bookkeeping collision leaves behind.
///
/// `grounded` is recomputed every physics step; `ground_y` remembers what it
/// was standing on, so a rooftop and the street are the same thing to
/// everything upstream.
#[derive(Component, Debug, Clone, Copy)]
pub struct Body {
    pub radius: f32,
    pub height: f32,
    pub gravity: f32,
    pub grounded: bool,
    pub ground_y: f32,
    pub airborne_time: f32,
    /// Set by the physics pass on the frame of a landing and consumed by the
    /// renderer for the dust puff. Positive = downward speed at impact.
    pub landed_speed: f32,
}

impl Default for Body {
    fn default() -> Self {
        Body {
            radius: 0.6,
            height: 1.8,
            gravity: 22.0,
            grounded: true,
            ground_y: 0.0,
            airborne_time: 0.0,
            landed_speed: 0.0,
        }
    }
}

/// Walk-cycle phase, advanced from horizontal speed.
///
/// Lives in the simulation rather than the renderer because it is derived
/// state with no meshes in it, and because a headless test can then assert
/// that a walking ant actually cycles its legs.
#[derive(Component, Debug, Clone, Copy)]
pub struct Gait {
    pub phase: f32,
    /// Cycles per metre travelled.
    pub rate: f32,
    /// 0..1, eased so that stopping settles the cycle rather than freezing it.
    pub amplitude: f32,
}

impl Gait {
    pub fn new(rate: f32) -> Self {
        Gait {
            phase: 0.0,
            rate,
            amplitude: 0.0,
        }
    }
}

impl Default for Gait {
    fn default() -> Self {
        Gait::new(2.2)
    }
}

// --------------------------------------------------------------------------
// the Wing Diver
// --------------------------------------------------------------------------

/// The Wing Diver's single resource: flight, gliding and shots all drink it.
///
/// The rule that gives the class its character is the *overheat*: run the
/// meter to exactly zero and `empty` latches, flight and gliding are locked
/// out entirely, and the recharge runs at a fraction of normal speed until
/// the meter is completely full again. Landing with 1% left recovers in a
/// moment; landing with 0% is a punishment.
///
/// Note how cheap the glide is next to the thrust. That ratio is the economy
/// of the class: climbing is expensive and covering ground is not, so the
/// efficient way anywhere is one hard burn followed by a long flat glide.
#[derive(Component, Debug, Clone, Copy)]
pub struct Energy {
    pub maximum: f32,
    pub current: f32,

    /// Per second of held thrust.
    pub drain_thrust: f32,
    /// Per second with the wings out.
    pub drain_glide: f32,
    /// Per second, feet down, not thrusting.
    pub regen_ground: f32,
    /// Per second, airborne, not thrusting.
    pub regen_air: f32,
    /// Regen multiplier while overheated.
    pub empty_penalty: f32,

    pub empty: bool,
}

impl Default for Energy {
    fn default() -> Self {
        Energy {
            maximum: 100.0,
            current: 100.0,
            drain_thrust: 26.0,
            drain_glide: 7.0,
            regen_ground: 45.0,
            regen_air: 12.0,
            empty_penalty: 0.55,
            empty: false,
        }
    }
}

impl Energy {
    pub fn fraction(&self) -> f32 {
        if self.maximum > 0.0 {
            self.current / self.maximum
        } else {
            0.0
        }
    }

    /// Take `amount` if it is there. Returns whether the spend happened.
    ///
    /// A spend that lands exactly on zero is allowed and *causes* the
    /// overheat -- you are always permitted the last drop, you just pay for
    /// it afterwards.
    pub fn spend(&mut self, amount: f32) -> bool {
        if self.empty || amount > self.current {
            return false;
        }
        self.current -= amount;
        if self.current <= 1e-6 {
            self.current = 0.0;
            self.empty = true;
        }
        true
    }
}

/// Tuning and state for jetpack movement: thrust up, or glide across.
///
/// Thrust is an acceleration rather than a set-velocity so that momentum
/// carries across a thrust tap, which is what makes EDF flight feel like
/// swimming rather than like an elevator.
///
/// The glide is the other half. With the wings out the descent is capped near
/// `glide_fall` and the horizontal drag drops by most of an order of
/// magnitude, so whatever speed you arrived with is speed you keep. It never
/// adds height -- it only stops you losing it, which is what separates a
/// glide from a hop.
#[derive(Component, Debug, Clone, Copy)]
pub struct Flight {
    pub thrust_accel: f32,
    pub rise_max: f32,
    pub air_accel: f32,
    pub air_max: f32,
    /// Horizontal damping per second, wings in.
    pub air_drag: f32,
    pub fall_max: f32,

    /// Terminal descent with the wings out.
    pub glide_fall: f32,
    /// How fast a hard fall is eased back to that.
    pub glide_bite: f32,
    /// Horizontal damping per second, wings out.
    pub glide_drag: f32,
    /// Air control while gliding: committed to a line.
    pub glide_accel: f32,
    /// Gravity multiplier while the wings are out and already descending.
    pub glide_gravity: f32,

    // live state
    pub thrusting: bool,
    pub gliding: bool,
}

impl Default for Flight {
    fn default() -> Self {
        Flight {
            thrust_accel: 34.0,
            rise_max: 14.0,
            air_accel: 26.0,
            air_max: 18.0,
            air_drag: 0.9,
            fall_max: 30.0,
            glide_fall: 4.5,
            glide_bite: 6.0,
            glide_drag: 0.05,
            glide_accel: 15.0,
            glide_gravity: 0.2,
            thrusting: false,
            gliding: false,
        }
    }
}

/// Ground locomotion for anything without a jetpack.
#[derive(Component, Debug, Clone, Copy)]
pub struct Walker {
    pub accel: f32,
    pub max_speed: f32,
    pub friction: f32,
    /// Degrees per second toward the desired heading.
    pub turn_rate: f32,
    /// Ants leap; soldiers do not.
    pub jump_speed: f32,
}

impl Default for Walker {
    fn default() -> Self {
        Walker {
            accel: 30.0,
            max_speed: 7.0,
            friction: 10.0,
            turn_rate: 360.0,
            jump_speed: 0.0,
        }
    }
}

/// What an entity is trying to do this frame, whoever decided it.
///
/// `move_x` / `move_y` are in the entity's *aim* frame: `move_y` is forward
/// along `aim_yaw` and `move_x` is to its right. Length is clamped to 1 by
/// the movement systems so that diagonals are not faster.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Intent {
    pub move_x: f32,
    pub move_y: f32,
    pub aim_yaw: f32,
    pub aim_pitch: f32,
    pub thrust: bool,
    pub glide: bool,
    pub fire: bool,
}

impl Intent {
    /// Wipe everything except the aim.
    ///
    /// The aim survives on purpose: it is where the entity is *looking*,
    /// which does not stop being true because it took a frame off. A brain
    /// that clears its aim as well spins to face north whenever it loses
    /// its target.
    pub fn clear(&mut self) {
        self.move_x = 0.0;
        self.move_y = 0.0;
        self.thrust = false;
        self.glide = false;
        self.fire = false;
    }
}

// --------------------------------------------------------------------------
// combat
// --------------------------------------------------------------------------

#[derive(Component, Debug, Clone, Copy)]
pub struct Health {
    pub maximum: f32,
    pub current: f32,
    /// Seconds remaining of the white damage flash.
    pub hurt_flash: f32,
}

impl Health {
    pub fn new(maximum: f32) -> Self {
        Health {
            maximum,
            current: maximum,
            hurt_flash: 0.0,
        }
    }

    pub fn damage(&mut self, amount: f32) {
        self.current = (self.current - amount).max(0.0);
        self.hurt_flash = 0.12;
    }
}

/// What a projectile looks like when the renderer gets to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoltKind {
    Lance,
    Tracer,
    Acid,
    /// Melee. Never spawns a projectile entity.
    Bite,
}

/// A projectile launcher. Melee is a weapon with a very short range.
///
/// `energy_cost` is what ties the Wing Diver's gun to her jetpack: firing the
/// lance is flight you are choosing not to take.
#[derive(Component, Debug, Clone, Copy)]
pub struct Weapon {
    pub damage: f32,
    pub speed: f32,
    pub cooldown: f32,
    pub range: f32,
    /// Degrees of random cone.
    pub spread: f32,
    pub energy_cost: f32,
    pub melee: bool,
    pub kind: BoltKind,

    /// Counts down to zero, then the weapon is ready.
    pub timer: f32,
    pub muzzle_flash: f32,
}

impl Default for Weapon {
    fn default() -> Self {
        Weapon {
            damage: 30.0,
            speed: 90.0,
            cooldown: 0.28,
            range: 140.0,
            spread: 0.0,
            energy_cost: 0.0,
            melee: false,
            kind: BoltKind::Lance,
            timer: 0.0,
            muzzle_flash: 0.0,
        }
    }
}

/// A point that travels in a straight line and hurts one thing once.
#[derive(Component, Debug, Clone, Copy)]
pub struct Projectile {
    pub damage: f32,
    pub faction: Faction,
    pub life: f32,
    pub radius: f32,
    pub owner: Entity,
}

/// Ant behaviour: close the distance, then bite. Spitters shoot first.
///
/// Ants do not path around buildings -- they walk into them and slide along,
/// which is both period-correct and, in practice, how the real ones behave.
#[derive(Component, Debug, Clone, Copy)]
pub struct AntBrain {
    pub target: Option<Entity>,
    /// Seconds until the next target search.
    pub retarget: f32,
    pub sight: f32,
    pub bite_range: f32,
    pub spitter: bool,
    pub leap_cd: f32,
}

impl Default for AntBrain {
    fn default() -> Self {
        AntBrain {
            target: None,
            retarget: 0.0,
            sight: 120.0,
            bite_range: 2.6,
            spitter: false,
            leap_cd: 0.0,
        }
    }
}

/// A grunt: hold near the rally point, shoot the nearest ant, back off.
///
/// `courage` decides how close it is willing to get before it stops
/// advancing, which is enough variation to make a squad look like people.
#[derive(Component, Debug, Clone, Copy)]
pub struct AllyBrain {
    pub target: Option<Entity>,
    pub retarget: f32,
    pub sight: f32,
    pub standoff: f32,
    pub courage: f32,
    pub rally: Vec3,
    /// Fixed at spawn, so a squad's strafe directions do not all agree.
    pub strafe: f32,
}

impl Default for AllyBrain {
    fn default() -> Self {
        AllyBrain {
            target: None,
            retarget: 0.0,
            sight: 90.0,
            standoff: 18.0,
            courage: 1.0,
            rally: Vec3::ZERO,
            strafe: 0.35,
        }
    }
}

// --------------------------------------------------------------------------
// scenery and presentation
// --------------------------------------------------------------------------

/// A static box the world is made of. Collision reads it as an AABB.
///
/// Stored as half-extents around [`Pose::pos`], with the box sitting on the
/// ground: it spans `y` from 0 to `height`.
#[derive(Component, Debug, Clone, Copy)]
pub struct Building {
    pub half_x: f32,
    pub half_z: f32,
    pub height: f32,
    /// Picks a texture in the renderer.
    pub style: usize,
}

/// What model the renderer should give this entity.
///
/// The simulation only ever states a `kind` and a tint; it has no idea how
/// meshes work. Nothing here is read by any gameplay system, so a headless
/// run simply never looks at it.
#[derive(Component, Debug, Clone, Copy)]
pub struct Renderable {
    pub kind: ModelKind,
    pub scale: f32,
    pub tint: [f32; 3],
}

impl Renderable {
    pub fn new(kind: ModelKind, tint: [f32; 3]) -> Self {
        Renderable {
            kind,
            scale: 1.0,
            tint,
        }
    }
}

/// Every model the renderer knows how to build.
///
/// An enum rather than the Python original's string, because the compiler
/// will then tell you about the arm you forgot when you add one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    Ant,
    Soldier,
    WingDiver,
    Bolt(BoltKind),
    Spark,
    Blood,
    Dust,
}

/// Seconds until the entity deletes itself. Used by effects and tracers.
#[derive(Component, Debug, Clone, Copy)]
pub struct Lifetime {
    pub remaining: f32,
    /// What it started at, so the renderer can shrink the puff over its life.
    pub total: f32,
}

impl Lifetime {
    pub fn new(seconds: f32) -> Self {
        Lifetime {
            remaining: seconds,
            total: seconds,
        }
    }
}
