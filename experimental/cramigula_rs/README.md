# Cramigula, in Bevy

A port of [`experimental/full3d`](../full3d) from Panda3D + `esper` to Rust +
Bevy 0.18. Same game, same numbers, same look; a different engine and a
different ECS.

```
cargo run --release                            # play it
cargo run --release -- --seed 7 --ants 24
cargo run --release --bin contactsheet         # six frames to a PNG, no window
                                               # (plain `cargo run` gives the game)
cargo test                     # 66 tests, no window -- 19 unit, 40 sim, 7 renderer
cargo test --no-default-features   # 51 of those, with no renderer in the build at all
```

Needs a Rust toolchain newer than the one Debian ships — Bevy 0.18 wants
1.88+. `~/.cargo/bin/cargo` is the rustup one; `/usr/bin/cargo` is not.

---

## What is different, and what is not

Every tuning constant is unchanged. Thrust still drains 26 a second, the
glide still costs 7, the overheat still latches at exactly zero and still
refuses to release below a full bar. The port is not an excuse to rebalance
anything.

Four things did change, and each of them is the engine's opinion rather than
mine.

**Coordinates.** Panda3D is Z-up; Bevy is Y-up. The map `(x, y, z) ->
(x, z, -y)` is a proper rotation, so every angle survives it: a heading of 0
still faces forward, positive yaw still turns counter-clockwise from above,
and `aim_vector(yaw, pitch)` is still the forward vector of the camera's
rotation. Not one number in the movement table had to move.

**Ordering.** `esper` sorts processors by an integer priority; Bevy has a
dependency graph. The eleven processors became one `.chain()`, which is
strictly better than the numbers were: Bevy inserts a synchronisation point
between two chained systems when the earlier one queues commands the later
one reads, so a lance spawned by `weapons` really is in the world when
`projectiles` runs, in the same frame. Under `esper` that worked because
spawning was immediate; here it works because it was declared.

**Death.** The Python version counted kills wherever the damage happened and
had to guard every site against counting the same death twice. In Bevy that
became one system with the query "alive things that are not alive any more",
which cannot double-count by construction. It is the shape the ECS wanted and
the Python was working around.

**Visuals are the entity.** Panda kept a side table from entity id to
`NodePath`. Bevy lets the ant *be* the node: `spawn_visuals` inserts a
`Transform`, a `Mesh3d` and a few children onto the same entity the
simulation is already stepping. Nothing needs keeping in sync and nothing
needs cleaning up — despawning an ant despawns its legs.

## The movement system

The Wing Diver is one meter with three straws in it. Flight, gliding and the
lance all drink from the same 100 points, and the whole class is the
arithmetic of not running out.

| Rule | Number | Why it is the number |
| --- | --- | --- |
| Thrust drain | 26/s | ~3.8s of continuous flight from full |
| Glide drain | 7/s | Altitude is expensive; distance is not |
| Lance | 12/shot | A shot from the air is altitude you chose not to buy |
| Ground recharge | 45/s | Landing is always the right answer, eventually |
| Air recharge | 12/s | Coasting recovers something, but not enough |
| Overheat penalty | x0.55 | And it does not release until the bar is **full** |

Thrust is an *acceleration*, not an assignment to `vel.y`. That single choice
is most of the feel: a tap gives a hop, a hold gives a climb, and a diver who
has been falling for a second has to pay that momentum back before she rises.

The glide is the other half, and the two prices are what make the class play.
With the wings out the descent settles near 4.5 m/s and the horizontal drag
drops by a factor of eighteen, so the speed you arrived with is speed you
keep. At 7 a second against the thrust's 26, the way across the city is one
hard burn upward and then a long flat descent — not a jetpack held down the
whole way.

**A glide never gains height.** Not from a fall, and not from a climb either:
gravity is only softened while already descending, so holding glide on the
way up reaches exactly the same apex as not holding it. Softening gravity
during an ascent would stretch the arc and let you float higher than you had
momentum for, which is a hop wearing a glide's name. Two tests guard that
(`a_glide_never_gains_height`, `a_glide_does_not_extend_a_climb`).

The overheat is the interesting rule. Touching exactly zero latches
`Energy::empty`: no flight, no glide, and a recharge at 55% that does not
release until the meter is completely full. Landing with 1% left costs you a
moment. Landing with 0% costs you the fight — you are a slow soldier with no
gun until it fills.

All of it is `sim::flight`, about a hundred lines, and `tests/headless.rs`
asserts every row of that table.

## The camera

Third-person, on a spring, behind and to the right. The one thing that
matters: **it looks along the aim, never at the player.**

The obvious chase camera positions itself from the aim and then looks at the
character — and the moment you pitch, screen centre is her head rather than
the direction the lance travels. The crosshair then lies about where the shot
goes, by more the harder you are aiming, and the gun feels broken in a way
that is very hard to blame on the camera.

So orientation comes straight from the aim angles and nothing else. The
forward vector of `Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0)` is exactly
`sim::aim_vector`, so screen centre *is* the fire direction by construction.
Position is then fully decoupled: the boom can be yanked out of a wall,
lifted off the tarmac or snapped across the map on a restart without the
reticle ever drifting off the shot. The player is kept from behind the
crosshair by offsetting the boom right and up, not by aiming somewhere she is
not.

The boom retracts out of walls fast (26 m/s) and extends back out slow (7
m/s), because a camera that springs back at the rate it went in lurches every
time you skim a building. `tests/rendered.rs` checks the camera's forward
vector against `aim_vector` at five aim angles including two past 74 degrees
of pitch, flies a lap of the city checking the boom against the collision
grid, and measures the two boom rates against each other.

## The PlayStation look

Not a post-process filter. The console did not have an aesthetic, it had five
limitations, and the interesting part is how they interact — the wobble only
reads as wobble because the framebuffer is 320x240.

1. **No subpixel precision.** The vertex shader snaps clip-space XY to the
   low-res pixel grid, so geometry shimmers and shared edges crack open.
2. **No perspective correction.** UVs are declared `@interpolate(linear)`,
   WGSL's spelling of GLSL's `noperspective`, so textures swim exactly as
   they did. This is why large surfaces are chopped into 5-8m cells: the warp
   scales with how much depth a single polygon spans, and every game of the
   era subdivided for the same reason.
3. **15-bit colour.** Five bits a channel with a 4x4 Bayer dither, applied at
   320x240 so the crosshatch is coarse and visible.
4. **Vertex lighting**, Gouraud, quantised to four bands. No smooth ramps.
5. **Hard distance fog**, doing the job the draw distance could not.

The one thing deliberately *not* reproduced is the missing Z-buffer. Real
hardware sorted per polygon and got it wrong at the seams; a real ordering
table would make the city unreadable, and games shipped with that artifact
rather than because of it.

Three of Bevy's defaults had to be switched off, and all three for the same
reason — they are corrections for problems this renderer is trying to have.
`Msaa::Off`, because multisampling smooths exactly the stair-stepped edges
the snapping exists to produce. `Tonemapping::None`, because a filmic curve
applied after a five-bit quantise un-quantises it. `DebandDither::Disabled`,
because Bevy's own dither fighting the Bayer matrix gives a mush that is
neither.

The quantisation happens in *display* space, not in the linear space the rest
of the pipeline works in. The console's framebuffer held five bits of an
already gamma-encoded signal, so its 32 levels were evenly spaced to the eye;
quantising the linear value instead spends nearly all of them on the darks
and leaves every bright gradient visibly stepped.

The HUD is `bevy_ui` bound to the scene camera, so it lays out against the
320x240 target and is as chunky as the world instead of floating above it at
native resolution looking like a different decade. Press **F1** in game to
switch the vertex snapping off and see what it was doing.

There are no art assets. Every texture is painted procedurally in `ps1.rs`
and every model is boxes from `models.rs` — which is convenient for a
repository and, as it happens, roughly what a PS1 character was anyway. The
shader is embedded in the binary rather than loaded from an `assets/`
directory, so there is nothing to ship alongside it and no working directory
to get wrong.

## Layout

| File | What is in it |
| --- | --- |
| [components.rs](src/components.rs) | Every component. Data only |
| [rng.rs](src/rng.rs) | PCG32, so a seed means the same thing on every machine |
| [spatial.rs](src/spatial.rs) | Rebuilt-per-frame point hash, and a build-once box grid |
| [spawn.rs](src/spawn.rs) | Factories — the only place that knows which components go together |
| [sim.rs](src/sim.rs) | Twelve systems and the plugin that orders them |
| [worldgen.rs](src/worldgen.rs) | The city, the squad, the opening patrol |
| [models.rs](src/models.rs) | Procedural geometry |
| [ps1.rs](src/ps1.rs) / [ps1.wgsl](src/ps1.wgsl) | The material, the shaders, the low-res target |
| [render.rs](src/render.rs) | Systems that read the simulation and never write it |
| [hud.rs](src/hud.rs) | Five numbers and a crosshair |
| [main.rs](src/main.rs) | Window, input, one system |

### The seam is a cargo feature

`sim`, `components`, `spatial`, `spawn`, `worldgen` and `rng` depend on
`bevy_ecs`, `bevy_app`, `bevy_math` and `bevy_platform`. The renderer depends
on the full `bevy` facade, which is an **optional** dependency behind the
default-on `render` feature.

So `cargo test --no-default-features` runs the entire game — flight,
collision, ants, gunfire, waves, world generation — with `bevy_render` not
merely unused but *absent from the build graph*. It cannot open a window
however hard it tries, and that is checked by the compiler on every run
rather than by a convention somebody has to remember.

### Composition, not inheritance

An ant is *a thing with a `Walker`, an `AntBrain` and a bite*. A Wing Diver is
*a thing with a `Flight` and an `Energy`*. Neither is a subtype of anything.
The load-bearing piece is `Intent`: the keyboard writes six fields, and so do
`ant_brains` and `ally_brains`. Nothing downstream can tell which produced
them, which is why giving a grunt a `Flight` component would simply make him
fly.

## Controls

| | |
| --- | --- |
| Mouse | Aim. Screen centre is exactly where the lance goes |
| W A S D | Move, relative to where you are looking |
| Space (hold) | Thrust |
| Shift (hold) | Glide — wings out: the fall slows and the speed keeps |
| Left mouse / Ctrl | Fire the lance |
| R | Restart with a fresh city |
| F1 | Toggle the vertex snapping |
| Escape | Release the mouse; again to quit |

## Testing

Nothing here opens a window, including the tests that use the GPU.

`tests/headless.rs` is the simulation: a bare `App`, `SimPlugin`, and a
timestep written straight into `SimClock` so a test that wants nine hundred
exact sixtieths of a second gets exactly that.

`tests/rendered.rs` is in two halves. The camera tests need no GPU at all —
`chase_camera` is an ordinary system over `Pose`, `Intent` and the collision
grid. The pipeline tests do build a real wgpu device, offscreen, with
`WinitPlugin` disabled and no primary window, and read the 320x240 buffer
back through `Screenshot::image`.

Those last two earn their keep. The reason the PS1 shader is visibly a shader
at all is that the equivalent Python test — "is the output quantised to 32
levels a channel" — caught a stray call silently discarding the shader set on
the line above it, leaving a scene that rendered through fixed-function and
looked *nearly* right.

`cargo run --bin contactsheet` writes `cramigula_contactsheet.png`, six
panels — street level, a thrust climb, a glide, the lance, the overheat
warning, a swarm — so the art can be eyeballed without launching the game.

---

## Where this goes next

Everything the [Panda3D README](../full3d/README.md#where-this-goes-next)
lists as future work still applies, and Bevy changes what the work costs:

- **Enemies in compact arrays** is already how `bevy_ecs` stores components,
  and `ant_brains` is a single pass over one query.
- **Rendering LODs** would key off the same `Gait` the animation already
  reads. Bevy's `VisibilityRange` does the distance banding for free, and
  distant swarms want one instanced mesh rather than one entity apiece.
- **Staggered AI** already exists as `AntBrain::retarget`, on a randomised
  0.35–0.75s timer. Bevy's answer to the rest is a fixed-timestep schedule
  with the far band on a slower one.
- **Physics** is upright cylinders against a static AABB grid, which is the
  "custom ground-following logic" end of the plan. `avian` or `bevy_rapier`
  would be the drop-in for the handful of bodies that want real dynamics.
- **Destruction** means splitting the damaged tower out of the baked batch;
  `rebuild_city` already rebuilds the whole thing from a query, so the
  machinery is there.
- **Explosions** are cubes with a `Lifetime` today. GPU particles in Bevy
  mean a compute shader and an instanced draw, which is the same shape the
  Panda3D plan described.
