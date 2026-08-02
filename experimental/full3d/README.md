# Cramigula

> use panda 3d to replicate the movement system from the earth defence force jet
> class.  give it a retro ps1 asthetic.  include some rudimentary NPC allys, ant
> enemies, and buildings.  use an ECS framework.

```
python3 experimental/full3d/cramigula.py
python3 experimental/full3d/headlesstest.py                # 40 tests, no window
python3 experimental/full3d/headlesstest.py --no-render    # sim only, no GL
```

Needs `panda3d` and `esper`.

---

## The movement system

The Wing Diver is one meter with three straws in it. Flight, dashing and the
lance all drink from the same 100 points, and the whole class is the
arithmetic of not running out.

| Rule | Number | Why it is the number |
| --- | --- | --- |
| Thrust drain | 26/s | ~3.8s of continuous flight from full |
| Dash | 18 flat | Five dashes, or one dash and two seconds of air |
| Lance | 12/shot | A shot from the air is altitude you chose not to buy |
| Ground recharge | 45/s | Landing is always the right answer, eventually |
| Air recharge | 12/s | Gliding recovers something, but not enough |
| Overheat penalty | x0.55 | And it does not release until the bar is **full** |

Thrust is an *acceleration*, not an assignment to `vel.z`. That single choice
is most of the feel: a tap gives a hop, a hold gives a climb, and a diver who
has been falling for a second has to pay that momentum back before she rises.
The dash is an impulse rather than a state, so the speed it leaves behind is
yours to keep -- dash, then thrust, and you hold more speed than air control
alone could give you, because the soft cap only stops you *accelerating* past
it.

The overheat is the interesting rule. Touching exactly zero latches
`Energy.empty`: no flight, no dash, and a recharge at 55% that does not release
until the meter is completely full. Landing with 1% left costs you a moment.
Landing with 0% costs you the fight. Everything else about playing the class is
downstream of learning to always leave yourself one dash.

All of it is `FlightProcessor` in [sim.py](sim.py), about eighty lines, and
`headlesstest.py` asserts every row of that table.

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
| [headlesstest.py](headlesstest.py) | 40 tests and a contact sheet |

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
| Mouse | Aim. The camera follows the aim, not the velocity |
| W A S D | Move, relative to where you are looking |
| Space (hold) | Thrust |
| Shift | Boost dash |
| Left mouse | Fire the lance |
| R | Restart with a fresh city |
| F1 | Toggle the vertex snapping |
| Escape | Release the mouse; again to quit |

## Testing

`headlesstest.py` opens no window. The simulation half runs with no GL context
at all; the renderer half uses `window-type offscreen`, so nothing appears on
the desktop even while the shaders are being exercised. It writes
`cramigula_headless.png`, a six-panel contact sheet -- street level, in flight,
firing, overheated, a rooftop, a swarm -- so the art can be eyeballed without
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
