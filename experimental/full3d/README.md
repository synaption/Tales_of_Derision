# Cramigula

> use panda 3d to replicate the movement system from the earth defence force jet
> class.  give it a retro ps1 asthetic.  include some rudimentary NPC allys, ant
> enemies, and buildings.  use an ECS framework.

```
python3 experimental/full3d/cramigula.py
python3 experimental/full3d/headlesstest.py                # 46 tests, no window
python3 experimental/full3d/headlesstest.py --no-render    # sim only, no GL
```

Needs `panda3d` and `esper`.

There is a Rust + Bevy port of this in [`../cramigula_rs`](../cramigula_rs),
with the same numbers and the same look. Its README covers what the change of
engine changed and what it did not.

---

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

Thrust is an *acceleration*, not an assignment to `vel.z`. That single choice
is most of the feel: a tap gives a hop, a hold gives a climb, and a diver who
has been falling for a second has to pay that momentum back before she rises.

The glide is the other half, and the two prices are what make the class play.
With the wings out the descent settles near 4.5 m/s and the horizontal drag
drops by a factor of eighteen, so the speed you arrived with is speed you
keep. At 7 a second against the thrust's 26, the way across the city is one
hard burn upward and then a long flat descent — not a jetpack held down the
whole way.

**A glide never gains height.** Not from a fall, and not from a climb either:
gravity is only softened while already descending, so holding glide on the way
up reaches exactly the same apex as not holding it. Softening gravity during an
ascent would stretch the arc and let you float higher than you had momentum
for, which is a hop wearing a glide's name. Two tests guard that
(`test_glide_never_gains_height`, `test_glide_does_not_extend_a_climb`).

The overheat is the interesting rule. Touching exactly zero latches
`Energy.empty`: no flight, no glide, and a recharge at 55% that does not
release until the meter is completely full. Landing with 1% left costs you a
moment. Landing with 0% costs you the fight — you are a slow soldier with no
gun until it fills. Everything else about playing the class is downstream of
learning to always leave yourself the glide home.

All of it is `FlightProcessor` in [sim.py](sim.py), about a hundred lines, and
`headlesstest.py` asserts every row of that table.

## The camera

Third-person, on a spring, behind and to the right. The one thing that matters:
**it looks along the aim, never at the player.**

The obvious chase camera positions itself from the aim and then `lookAt`s the
character — and the moment you pitch, screen centre is her head rather than the
direction the lance travels. The crosshair then lies about where the shot goes,
by more the harder you are aiming, and the gun feels broken in a way that is
very hard to blame on the camera.

So orientation comes straight from the aim angles and nothing else. Panda's HPR
forward vector for `(yaw, pitch, 0)` is exactly `sim.aim_vector`, so screen
centre *is* the fire direction by construction. Position is then fully
decoupled: the boom can be yanked out of a wall, lifted off the tarmac or
snapped across the map on a restart without the reticle ever drifting off the
shot. The player is kept from behind the crosshair by offsetting the boom right
and up, not by aiming somewhere she is not.

The boom retracts out of walls fast (26 m/s) and extends back out slow (7 m/s),
because a camera that springs back at the rate it went in lurches every time
you skim a building. The "crosshair points where the lance goes" test checks
the camera's forward vector against `aim_vector` at five aim angles, including
two past 70 degrees of pitch, and the "stays out of the ground and walls" test
flies a lap of the city checking the boom against the collision grid.

## The PlayStation look

Not a post-process filter. The console did not have an aesthetic, it had five
limitations, and the interesting part is how they interact -- the wobble only
reads as wobble because the framebuffer is 320x240.

1. **No subpixel precision.** The vertex shader snaps clip-space XY to the
   low-res pixel grid, so geometry shimmers and shared edges crack open.
2. **No perspective correction.** UVs are declared `noperspective`, so textures
   swim exactly as they did. This is why large surfaces are chopped into 5-8m
   cells: the warp scales with how much depth a single polygon spans, and every
   game of the era subdivided for the same reason.
3. **15-bit colour.** Five bits a channel with a 4x4 Bayer dither, applied at
   320x240 so the crosshatch is coarse and visible.
4. **Vertex lighting**, Gouraud, quantised to four bands. No smooth ramps.
5. **Hard distance fog**, doing the job the draw distance could not.

The one thing deliberately *not* reproduced is the missing Z-buffer. Real
hardware sorted per polygon and got it wrong at the seams; a real ordering
table would make the city unreadable, and games shipped with that artifact
rather than because of it.

The HUD is drawn into the same 320x240 buffer on a second display region, so it
is as chunky as the world instead of floating above it at native resolution
looking like a different decade. Press **F1** in game to switch the vertex
snapping off and see what it was doing.

There are no art assets. Every texture is painted procedurally in
[ps1.py](ps1.py) and every model is boxes from [models.py](models.py) -- which
is convenient for a repository and, as it happens, roughly what a PS1 character
was anyway.

## Layout

| File | What is in it |
| --- | --- |
| [components.py](components.py) | Every component. Data only; imports nothing but `Vec3` |
| [entities.py](entities.py) | Factories -- the only place that knows which components go together |
| [spatialhash.py](spatialhash.py) | Rebuilt-per-frame point hash, and a build-once box grid |
| [sim.py](sim.py) | Eleven processors and the `Sim` that owns them |
| [worldgen.py](worldgen.py) | The city, the squad, the opening patrol |
| [ps1.py](ps1.py) | Shaders, procedural textures, the low-res pipeline |
| [models.py](models.py) | Procedural geometry |
| [render.py](render.py) | Three processors that read the sim and never write it |
| [cramigula.py](cramigula.py) | Window, input, one task |
| [headlesstest.py](headlesstest.py) | 46 tests and a contact sheet |

The seam that matters: `sim.py` imports `esper`, `math` and
`panda3d.core.Vec3`, and nothing else. It never touches a `NodePath` and cannot
open a window. Delete every processor in `render.py` and the game still runs --
which is exactly what `--no-render` does, and why the flight model can be
tested to six decimal places with no GPU in the room.

Both halves are the same ECS. `render.attach()` adds its processors to the same
`esper` world at low priority, so one `sim.step(dt)` runs the ants, the physics
and the camera.

### Composition, not inheritance

An ant is *a thing with a `Walker`, an `AntBrain` and a bite*. A Wing Diver is
*a thing with a `Flight` and an `Energy`*. Neither is a subclass of anything.
The load-bearing piece is `Intent`: the keyboard writes six fields, and so do
`AntProcessor` and `AllyProcessor`. Nothing downstream can tell which produced
them, which is why giving a grunt a `Flight` component would simply make him
fly.

## Controls

| | |
| --- | --- |
| Mouse | Aim. Screen centre is exactly where the lance goes |
| W A S D | Move, relative to where you are looking |
| Space (hold) | Thrust |
| Shift (hold) | Glide -- wings out: the fall slows and the speed keeps |
| Left mouse | Fire the lance |
| R | Restart with a fresh city |
| F1 | Toggle the vertex snapping |
| Escape | Release the mouse; again to quit |

## Testing

`headlesstest.py` opens no window. The simulation half runs with no GL context
at all; the renderer half uses `window-type offscreen`, so nothing appears on
the desktop even while the shaders are being exercised. It writes
`cramigula_headless.png`, a six-panel contact sheet -- street level, climbing,
gliding, firing, overheated, a swarm -- so the art can be eyeballed without
launching the game.

The renderer tests earn their keep. The reason the PS1 shader is visibly a
shader at all is that a test measuring "is the output quantised to 32 levels a
channel" caught `setShaderAuto(False)` silently discarding the shader set on the
line above it, leaving a scene that rendered through fixed-function and looked
*nearly* right.

---

## Where this goes next

Current scale is ~50 ants, ~230 buildings and one flat arena, at about 3.7ms a
frame for sim and render together (0.26ms of that is the simulation). None of
the following is implemented; it is the plan for when the ant count goes up by
an order of magnitude.

**Enemies:** Use one centralized swarm manager rather than separate Python
objects doing independent work. Store positions, health and states in compact
arrays. *(The `esper` component store is already array-of-structs per type, and
`AntProcessor` is a single pass over one query, so the change is the storage
layout rather than the control flow.)*

**Rendering:** Use three levels:

- Nearby enemies: complete skeletal animation, accurate collisions and full AI.
- Mid-distance enemies: reduced animation and simplified collision.
- Distant swarms: hardware-instanced meshes with shader-driven animation.

Panda3D supports geometry instancing, and its scene graph can separately cull
instances or use lower-level hardware instancing where appropriate. *(Today
`RenderProcessor` clones a prototype per entity and animates two leg racks; the
LOD split would key off the same `Gait` component the animation already reads.)*

**AI:** Update enemies at staggered rates. Nearby enemies might think every
frame, while distant enemies update five or ten times per second. Use a spatial
grid and shared flow-field movement rather than running an expensive path search
for every insect. *(`AntBrain.retarget` already staggers target selection on a
randomised 0.35-0.75s timer, and `spatialhash.SpatialHash` is the grid. There is
no path search to replace -- ants walk straight and slide off walls.)*

**Physics:** Reserve Bullet bodies for the player, vehicles, major enemies and
nearby interactive objects. Ordinary insects can use spheres, capsules, ray
tests and custom ground-following logic. Panda3D includes Bullet integration,
but giving every visible object full rigid-body simulation would be wasteful.
*(Nothing uses Bullet yet -- `PhysicsProcessor` is upright cylinders against a
static AABB grid, which is the "custom ground-following logic" end of this.)*

**Destruction:** Swap intact buildings for damaged or collapsed versions and
spawn a controlled quantity of pooled debris. That produces EDF-style
destruction without simulating every brick. *(`Building.rubble` is reserved for
this and currently unread; the city is baked into one geom per facade style, so
a swap means splitting the damaged tower out of the baked batch.)*

**Explosions and projectiles:** Pool projectiles, combine rapid-fire weapons
into ray tests when possible, and run large particle systems on the GPU.
Panda3D's compute-shader interface explicitly supports GPU-based particle
algorithms. *(Projectiles are entities with a substepped segment test today, and
effects are billboarded quads with a `Lifetime`.)*
