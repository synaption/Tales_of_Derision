# juice workbench

A bench for looking at *game feel* one switch at a time, in the exact context
this repo cares about: a tile-based roguelike, where the sim is a spreadsheet
that occasionally changes a cell and every drop of physicality has to be
manufactured on top of it.

```
python3 rogue_juice.py                      # the bench, on the CPU
python3 rogue_juice_gl.py                   # the same bench, on the card
python3 rogue_juice.py --headless a.png     # one scripted swing, no window
python3 rogue_juice_gl.py --headless b.png  # the same, through the shaders
python3 juicetest.py                        # 157 headless tests
python3 gltest.py                           # 81 more, on a windowless context
python3 -m pytest juicetest.py gltest.py -q # all of them
```

Walk into something to hit it. `Tab` flips the whole thing between "raw sim"
and "juiced" mid-swing, which is the comparison the bench exists for -- reading
about hit-stop is nothing like turning it off and hitting a dummy.

## The files

| file | what is in it | pygame? | GL? |
|---|---|---|---|
| `juicefx.py` | every effect, as maths | **no** | **no** |
| `audiofx.py` | synthesis, pitch ladders, buses | only in `SoundBank` | no |
| `tiles.py` | slicing and tinting the sheet | yes | no |
| `rogue_juice.py` | the sim, the panel, software compositing | yes | no |
| `glfx.py` | shaders, batching, post chain, normals | only to upload | yes |
| `juicesettings.py` | the settings file, and the merge rules | **no** | **no** |
| `terrain.py` | levels, cliffs and stairs, as data | **no** | **no** |
| `rogue_juice_gl.py` | the same bench on the card, plus physics, menu, options | yes | yes |

That split is the point rather than tidiness. An effect you cannot test without
opening a window is an effect you will never tune, so the maths runs, and is
tested, with no display anywhere; the audio's synthesis and pitch machinery are
plain numpy and are tested with no sound card. `SoundBank` keeps a log of what
it was asked to play, so "the swing fires on the wind-up and the impact on
contact" is an assertion rather than something you listen for.

It also means the GL version is not a fork. `rogue_juice_gl.py` imports `World`,
`Juice`, `Entity` and every toggle and slider from `rogue_juice.py` and replaces
only the drawing; `juicefx.py` does not know it exists. Its two non-drawing
additions follow the same rule from the other side. `RagdollField` reads the
wall grid, the entity list and the shockwave pool through the sim's own public
surface and writes back onto the `Corpse` objects, so it is a layer on top of
`rogue_juice.py` rather than an edit to it. `terrain.py` could not quite manage
that -- whether a step is legal is a question only the sim can be asked -- so it
is composed *into* `World` through one empty slot and one method that returns
`True` while the slot is empty, which is the smallest seam that could work and
leaves the software bench exactly as flat as it was.

## The governing idea

**The sim is instant, the picture is late.** When the player walks north the
logical position changes on that frame, finally and completely; the sim is free
to path, spot and swing on the new tile immediately. What the eye sees is a body
still standing on the old tile and hurrying to catch up, and nearly everything
here is a variation on how it hurries.

That gives the arrangement its shape:

    Body      where a thing is drawn: offsets, scale, spin, fade, whiten, and
              a per-band shear. Reset to identity every frame.
    Motion    something that scribbles on a Body for a while and reports that
              it is finished.
    Animator  holds the Motions running on one Body and lets every one of them
              contribute.

Because the Body is wiped and then *accumulated* into -- offsets add, scales
multiply, whiteness takes the max -- Motions compose without knowing about each
other. There is no `HoppingFlashingBreathingSprite`; there are three small
Motions in a list. `Sequence` and `Parallel` are Motions too, so an attack is a
plain nested structure with a `Callback` in the middle that fires on the frame
the fist *arrives*.

## The panel

Seven groups, one scrolling column. Click a row to toggle it; drag a slider;
right-click a slider to put it back. The wheel scrolls.

The sliders matter more than the toggles. A toggle answers "does this effect
help?"; a slider answers "how much of it?", and almost every effect here is
good at some amplitude and ridiculous at twice it. The only way to find the
line is to drag past it.

The frame cost in milliseconds is in the header, in red once it exceeds a 60Hz
slot. Several of these effects are genuinely expensive, and a bench that hid
that would be teaching the wrong lesson. Toggle lighting and watch it move.

## What is here, and why

**movement** — hop arcs, squash and stretch, slime drag, banking, idle
breathing and fidgets, turn-to-face, afterimages, landing dust, blob shadows,
tails and capes, wall bumps.

The slime drag is ten springs across the sprite with a *stiffness gradient*, so
the leading band nearly keeps up and the trailing one is still leaving as the
front arrives; the sprite is then drawn band by band. A single spring can only
drag a sprite about as a rigid block. Tails are the other half of
follow-through -- a part that is visibly late -- and are ropes, not springs:
without a hard length limit a fast step stretches the chain to twice its size
and it draws as a spike sticking out sideways.

**attack** — anticipation, lunge, hit-stop, hit flash, knockback, weight,
crits, shiver, weapon arc, slash cut, chip damage, deaths, corpses.

Weight divides every reaction the target receives, which turns one knockback
into nine from one number per monster. Crits are *tiers*: the same hit with
more of everything, on every channel at once -- shake, freeze, particles,
pitch, the size of the number -- because an escalation of something familiar
reads better than a substitution.

**camera** — trauma shake, directional kick, zoom punch, roll tilt, follow lag,
look-ahead, gamepad rumble.

Everything contributes *trauma*, one number in 0..1 that decays on its own, and
displacement is `trauma ** 2`. Effects that each set their own shake fight each
other and stack into a seizure. Rumble is driven from that same number, so the
motors and the picture can never disagree about how hard something hit.

**world** — particles, damage numbers, shockwaves, floor ripple, blood decals,
ambient sway, and how many enemies a reset puts on the floor.

Decals are the only thing here that is *evidence* rather than an event. The
ambient sway is given to tufts and torches rather than to the floor tiles,
because the arena is baked into one surface and wobbling it would undo that.

**screen** — screen flash, vignette, RGB split, bloom, lighting with cast
shadows, lit sprites, outlines, heat haze, pixel snapping.

Pygame has no shaders, so this group is the software cousins. Bloom is a
downscale, a threshold, an upscale and an add -- the downscale *is* the blur.
Lighting is a half-resolution light map, multiplied *and* added: a multiply can
only take brightness away, so on a floor this dark a torch would light nothing.
Heat haze is rows re-blitted at sine offsets, which is the same trick the slime
uses, pointed at the background.

The light map is not flat. Four sliders shape it, and they are the same four
numbers the GL bench uses, doing the same arithmetic:

| slider | what it decides |
|---|---|
| `light height` | how far the lights float above the floor. Low is a hot spot a tile wide; high is a lamp on the ceiling. Shape, not reach. |
| `light / dark contrast` | ambient down and direct light up together. Ten is a torch-only dungeon; below one the shadows fade out with everything else. |
| `wall shadow amount` | masonry as a real occluder. |
| `NPC shadow amount` | the soft shadows bodies throw. |

What the card answers per fragment -- march towards the light, see what you
crossed -- is answered here per *occluder* instead. A wall's shadow is the quad
between its two silhouette corners and those corners projected away from the
light, which is one polygon fill for a shape a ray march pays four hundred
thousand samples for; a body's is three nested trapezoids, because a smudge two
tiles long reads as soft at three steps and a real blur would cost more than
every light in the room.

The bodies go into a layer of their own and are *multiplied* into the walls'
rather than drawn on top of them. This is the one place the software version
had to learn something the shader gets for nothing: `draw.polygon` replaces the
pixels it covers, so a monster standing between you and a pillar used to stamp
its 55%-dark trapezoid across the pillar's fully-dark wedge and cut a lighter,
body-shaped hole through it. Visibility terms multiply -- the shader says
`wall_shadow * npc_shadow` -- and within the body layer the three penumbra
steps are drawn widest-first across every caster before the next step starts,
so one body's soft edge can never land on another's core.

Both layers are drawn in the light's *own* space, so the
torches -- which never move -- are built once and cached for the session and
only the light the player carries is rebuilt as they walk.

Wall tiles are painted back in afterwards: a wall the light can see is lit
whatever is behind it, and without that rule the far side of a pillar goes to a
flat silhouette and reads as a hole in the room.

**lit sprites** is the software reading of the GL bench's normal-mapped
creatures. There are no per-texel normals here, but what those normals actually
produce at 32 pixels is a warm band down the side facing the light and a cool
one opposite, so that is what this draws: two blits a body, both masked by the
sprite's own alpha, both *after* the light map has been multiplied through --
in the other order the highlight would only be dimmed again.

Pixel mode has three answers and no free one:

| mode | camera | sprites | cost |
|---|---|---|---|
| `rounded` | fractional | whole pixels | one blit |
| `snapped` | whole pixels | whole pixels | one blit, slow motion stair-steps |
| `subpixel` | fractional | blended across two pixels | four blits, slightly soft |

**feel** — input buffering and turn pacing. No pixels in either, and they are
the two that matter most. A press arriving 40ms before the current step ends is
the player telling you exactly what they want; dropping it makes the game feel
like it is ignoring them, and no animation polish covers for that. Turn pacing
shortens the turn while the player keeps walking and hands the weight straight
back when they stop -- corridors get walked at a sprint and the fight at the end
of one is at full weight, without anybody asking for either.

**sound** — layered impacts, pre-rendered pitch ladders, ducking, footsteps.

`pygame.mixer.Sound` has a volume and a left/right balance and nothing else:
there is no way to ask it to play a buffer faster. Five hits through one
unvaried sample sound like a stapler. So the cost is paid at load time instead
-- every voice is rendered once and resampled into a ladder of ordinary Sounds,
and picking a rung at play time is free. A hit is then a stack (impact +
material + tier), not a sample, which gives a combinatorial spread from a
handful of clips and lets a skeleton sound like bone.

The mixer buffer is set to 512 samples (~12ms). Pygame's default of 4096 is
93ms -- nearly six frames -- and is enough on its own to make a perfectly timed
hit feel mushy.

Two clips come from `audio/sfx`; the rest are synthesised, so the whole bench is
one checkout with nothing to download and every sound is a number away from
being different.

## Keys

```
wasd / arrows / hjkl / numpad   move, and attack by walking into something
space  swing at air      x  take a hit      shift-K  kill the nearest
r      reset (keeps your position)
Tab    A/B the whole lot        F1 / F2  all on / all off
F3     sliders back to defaults
[ ]    move easing              - =  master intensity
p      pixel mode (software)    m    music
b      throw a bomb (GL only)
esc    quit (software) / menu, options and save (GL)
```

## Notes

* Every duration is in seconds and every distance on a Body is in *tiles*, so
  the effects survive a change of tile size. Anything genuinely in pixels is
  written as a multiple of `PX`.
* Nothing uses `random`. Every random-looking thing is a seeded hash, so two
  runs with the same input draw the same picture and record the same mix.
* Nearest-neighbour transforms only -- `scale` and `rotate`, never
  `smoothscale` or `rotozoom` on a sprite. Squash and stretch resamples the
  sprite every frame, and a smooth path would leave a monster blurry for the
  whole length of every hop.
* The arena is baked into one surface. Only the tiles a ripple is passing
  through are redrawn.

## The same bench on the graphics card

`rogue_juice_gl.py` is the identical bench with the presentation moved to
ModernGL. Everything above still applies -- same sim, same panel, same sliders,
same sounds -- and a handful of things stop being compromises.

**The slime deforms continuously.** `Jelly` is drawn on the CPU as ten bands
with a surface-tension clamp holding them together, and that entire design
exists because `Surface.blit` can move rectangles and nothing else. In the
shader the deformation is a function: every fragment asks where the material it
is showing came from. No band count, no `link` constraint, nothing to tear. The
same springs drive it.

**Creatures are lit rather than tinted.** Each sprite carries a normal map
generated from its own silhouette -- a blurred alpha mask's gradient *is* the
normal of an inflated shape -- so a torch rakes across a body and a hit flash
shades the room. No new art, and it is per texel: the light crosses a body
continuously and picks out every bump in the silhouette on the way. The
software bench answers the same question with two gradient-masked blits a body
and a light map that has to be added back as well as multiplied, because a
multiply on a floor this dark can only ever take brightness away.

**Pixel art is crisp *and* moves smoothly.** The three-way `pixel mode` choice
is a `blit` artefact, not a property of pixel art. A textured quad sampled flat
inside a texel and blended across the seam over one screen pixel is exactly as
sharp as nearest at rest and smooth at any sub-pixel speed. The GL build drops
`pixel mode` and gains a `filter sharpness` slider you can drag to zero to watch
what the filter is actually buying.

**One draw call.** Sprites, particles, shadows, decals, corpses, tails,
afterimages, damage numbers, shockwave rings, weapon arcs and slash cuts are all
instanced quads in one buffer, with the fragment shader branching on a shape id
to draw a ring or an arc procedurally. Particle counts stop being a budget.

The floor is a texture, so the ripple is a displacement of the coordinate that
samples it rather than a repaint of the tiles it passes -- and the ambient sway
can move the whole floor instead of a few dozen props, because it costs the
same either way.

**The dead come off the grid.** The one addition here that is not a rendering
argument. Everything alive is on a tile because the *sim* needs it there: it
takes a turn, it occupies a square, you have to be able to walk into it. A
corpse has none of that -- no turn, no collision the rules care about -- so the
grid is the last thing still holding it, and `RagdollField` takes it away. From
the moment `leave_corpse` drops one, a body is a circle with a velocity:

* it **bounces off walls**, circle against the tile rectangle rather than the
  tile centre, so it slides along a wall on a clean tangent instead of catching
  on the seam between two tiles of the same wall;
* it **is shoved aside in real time** by anything that walks through it, at the
  speed of the thing pushing it, mid-step, with no turn taken and no tile
  changing hands. On the grid there are exactly two things walking into a body
  can mean -- blocked, or nothing there -- and both are wrong;
* it **goes end over end when something explodes near it**. The field watches
  `world.fx.shockwaves` rather than the code that raises them, so every
  explosion in the bench throws corpses without knowing corpses can be thrown,
  and the ring on screen and the force in the physics are the same event by
  construction. **B** throws one;
* bodies **collide with each other** and settle into a heap, and a body that
  has stopped rolls the last few degrees flat -- a corpse frozen at twenty
  degrees reads as one still falling.

Seven sliders (`corpse radius`, `bounce`, `floor drag`, `shove strength`,
`blast force`, `death launch`, `tumble`) and one toggle, in the **attack** group
next to `corpses remain`. It runs at a fixed step with the frame chopped into as
many sub-steps as the fastest body needs, because a body leaving a blast at
3000px/s covers two tiles in a 60Hz frame and a single step that long walks
straight through a wall without ever overlapping it. Nine bodies cost 0.02ms.

`RagdollField` imports no GL, needs no frame and is stepped by the loop rather
than by `render`, so `gltest.py` drives all of it -- adoption, walls, shoving,
blasts, settling, determinism -- on a machine that cannot open a window.

**B throws a bomb.** It leaves the hand on an arc with a shadow under it,
bounces off walls through the corpses' own wall pass -- `_walls` never asks what
shape it is holding, so there is exactly one place a circle can be wrong about
masonry -- rolls to a stop, and goes off on a fuse wherever it ended up. Thrown
rather than dropped because *where* an explosion happens is the interesting
decision, and a bomb at your feet takes that decision away. Two sliders, `bomb
throw speed` and `bomb fuse`; a short enough fuse goes off in the air, which is
a different weapon.

**The floor has hills in it.** The other addition that is not purely a
rendering argument, and the shape of it is the interesting part: almost all of
verticality turns out to be a picture, and the small remainder that is not has
to go in the sim.

The model, in `terrain.py`, is the smallest one that reads as terrain. Every
tile has an integer **level**; tiles at the same level are one continuous floor;
the boundary between two is a **cliff**; and a **stair** is a single tile that
ramps from its low neighbour to its high one and is the only legal way between
levels. Passability is decided at the *edge* rather than by comparing levels,
which is one rule doing three jobs -- a cliff is impassable, a staircase is
climbable end to end, and the *side* of a staircase is a wall, because its side
edges sit at a half level and half levels never match anything.

That rule is the whole of what the sim knows. `World` grew a `terrain` slot and
`World.step_allowed`, which returns `True` when the slot is empty, and three
call sites ask it: the player's step, the enemies' step, and the flood fill that
tells the enemies where the player is. Walking into a cliff bumps and costs no
turn, exactly as walking into masonry does, because the animation is what tells
you the key was heard. Monsters route round to the stairs, because the goal map
has the heights in it. A *swing* is never stopped by an edge, at either end -- a
monster on a ledge that could not be reached and could not reach back would be
scenery.

Everything else is drawn:

* a level is **a number of pixels a thing is drawn further up the screen** --
  applied in the vertex shader in world space, so the camera's roll takes the
  hill with it, and applied to the *drawn* position rather than the tile, so
  walking up a stair rises smoothly across the step tween the sim already had;
* **the shadow is not lifted with the body**, which is the entire height cue,
  and is the same trick the hop and the thrown bomb already use;
* the floor shader **works out which terrace is visible** at each fragment.
  Raised ground is drawn shifted up, so a pixel can be showing the floor that
  is really there, the top of a tile some rows nearer the camera, or the cliff
  face hanging under that tile's near edge -- resolved front to back over three
  rows, in closed form rather than by marching down the column;
* lights stay on the flat plane, so a torch three tiles away lights a body on a
  plateau from three tiles away -- but raised ground is genuinely closer to
  them, so a plateau catches a torch the floor beside it misses;
* **corpses go over the edge.** Gravity along the ground is one force, and the
  moment the floor under a body is lower than it was, the difference becomes
  *height* and is spent falling. Written that way, a body that slides, is
  shoved, or is blown over a ledge falls off it without any of those three
  knowing ledges exist -- and a bomb rolls off a hill before its fuse runs out.

One toggle and three sliders (`level height`, `cliff shade`, `downhill slide`)
in the **world** group. The arena's own hills are hand-drawn as a 40x20 picture
in `terrain.ARENA` and checked at load: a stair that climbs two levels, a
plateau on top of a pillar or a corner of the room nobody can reach is an error
rather than a bad afternoon. The one thing no test can assert about a shader is
checked by measurement instead -- the ground height is computed twice, in Python
for the sprites and in GLSL for the floor, so `gltest.py` renders the same hill
with the terrain on and off and asserts the picture moved by exactly the number
of pixels the model claims.

**A hill hides what is behind it**, and the way that is done is the one place
the terrain pushed back on the renderer's design. The obvious answer is a depth
buffer -- and it is the wrong one here, because this batch is a single draw call
in painter's order and that is exactly what lets four thousand additive sparks
stack into a glow instead of into paint. Depth *writes* from translucent quads
would take that away, and depth without writes is not something this GL binding
exposes.

So the question is asked from the other end. Each fragment of each sprite walks
the terrain itself, finds which piece of ground is visible where it is about to
draw -- the same `resolve_surface` the floor runs, injected into both programs
from one source so the two cannot drift -- and stands down if that ground is
further forward than the thing it belongs to. Sorting solves the general
problem; this solves the one the room has. It costs three iterations of one
texture fetch over the pixels sprites actually cover, it is skipped outright on
a flat arena, and the whole terrain feature measures at about 0.05ms a frame.

What each quad is compared against is *where it stands*, which is not where it
is drawn: the foot of the quad by default, which is right for feet and right for
a shadow lying flat, and an explicit value for the two things whose picture and
position genuinely differ -- a bomb in the air, drawn well above the spot it is
over, and a damage number, which is a reading rather than an object and is put
in front of every hill in the room.

**A menu, on escape.** Resume, options, *save current settings as defaults*, and
quit. It pauses the sim -- a settings screen with a fight going on behind it is
one you cannot use -- and swallows every key while it is up, because a modal
that lets movement through is how you walk into a wall while picking a
resolution. It is drawn by the same software renderer that draws the panel,
into the same surface, and so costs nothing extra and shares its fonts.

**Display options, and a settings file.** Resolution, fullscreen and a frame
cap, all live; vsync, which is a property of the window and so honestly labelled
"on restart". Resizing goes through SDL rather than `pygame.display.set_mode`,
which would build a new window and take the GL context with it -- SDL simply
resizes the one that exists, so only the size-dependent buffers are rebuilt and
every shader, texture and atlas survives. The window is also resizable by drag.

The file is `juice_settings.json`, beside the checkout and gitignored, and the
rule it is built on is that **the file and the build never have to agree**:

* a key the file has and the build does not know is kept and written back, so
  opening an old build and saving does not delete a newer one's settings;
* a key the build has and the file does not keeps its code default, so adding a
  slider tomorrow needs no migration and no version bump;
* a value that is present but wrong -- a string where a number goes, a slider
  past its own maximum, a choice whose options have changed -- is ignored on its
  own, and everything around it still loads. A stray comma must not be a bench
  that will not start.

`juicesettings.py` holds all of that and imports neither pygame nor moderngl, so
the merge rules are tested as what they are: a dictionary and some arithmetic.
Saving also makes the saved values what F3 returns to, since "save as my
defaults" that F3 then undoes is not a save.

**An enemy count slider** (`world` group). Up to nine it is the hand-placed
roster, unchanged and in order; past that the cast repeats onto free tiles
picked by the same seeded hash everything else here uses, so forty bodies in the
room is one drag and still the same forty every run. It takes effect on the next
reset, because respawning the room under a fight in progress is not a thing a
slider should do.

The panel is the deliberate exception: it is drawn by the *software* renderer
into an offscreen surface and uploaded as a texture. Text layout is the one
thing pygame does better than a weekend of shader work, and it means every
slider, blurb, scroll and drag behaves identically in both builds.

### Notes on the GL build

* Under WSL, Mesa silently rasterises on the processor unless it is pointed at
  the card. `glfx.py` carries the same check and override as
  `experimental/fantacy_maps/inkfx.py`, and `run()` relaunches itself once if it
  finds it started on llvmpipe.
* `gltest.py` runs against a standalone EGL context with no window, per the
  repo rule about ModernGL tests not opening windows on the desktop.
* Shaders have no assertions and no stack traces -- a wrong uniform does not
  raise, it just draws something slightly wrong for ever. So `gltest.py` mostly
  renders and reads pixels back. Every real bug found writing it was of that
  kind: an atlas uploaded flipped, an array uniform the driver spelled
  `u_lights[0]`, a displacement that compressed where it should have stretched,
  and a "sharp" filter that snapped to texel seams and came out blurrier than
  bilinear. None of them raised anything.
* The floor palette is brightened before it becomes a texture. A colour chosen
  to look right *unlit* is far too dark to be an albedo: at 28/255 a torch can
  multiply it by two and it is still black. Switching lighting off scales it
  back so the toggle still compares like with like.

## Cost

On this machine, at 800x600, with a fight going on. Software is one core;
GL is a D3D12/WSL context on an RTX 4080.

| | software median | software p95 | GL median | GL p95 |
|---|---|---|---|---|
| defaults | 6.8ms | 14.1ms | **0.98ms** | 1.6ms |
| every effect on | 10.9ms | 18.4ms | **0.99ms** | 1.6ms |

The second row is the interesting one. On the CPU, switching everything on
costs 60% more frame; on the card it costs nothing measurable, because the
entire `screen` group is one fragment shader either way. Software lighting is
~3.7ms and the RGB split is most of a frame on its own; in GL neither is worth
a toggle except to see what it does.

The shadows are the cheap part of that 3.7ms, not the expensive one: wall
shadows, body shadows and lit sprites together add under half a millisecond,
because the occluder count is a few dozen polygons and the torches' masks are
cached across the whole session. What
costs is what always cost -- the two full-frame blends and the upscale of the
map itself, which is why it is built at half resolution.

What is left in the GL frame is mostly *not* graphics: the sim is 0.25ms and a
panel redraw is about 1.2ms of pygame text layout, which is why the panel is
throttled to 30Hz and any input forces one immediately.
