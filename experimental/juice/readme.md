# juice workbench

A bench for looking at *game feel* one switch at a time, in the exact context
this repo cares about: a tile-based roguelike, where the sim is a spreadsheet
that occasionally changes a cell and every drop of physicality has to be
manufactured on top of it.

```
python3 rogue_juice.py                   # the bench
python3 rogue_juice.py --headless a.png  # one scripted swing, no window
python3 juicetest.py                     # 142 headless tests
python3 -m pytest juicetest.py -q        # the same, under pytest
```

Walk into something to hit it. `Tab` flips the whole thing between "raw sim"
and "juiced" mid-swing, which is the comparison the bench exists for -- reading
about hit-stop is nothing like turning it off and hitting a dummy.

## The four files

| file | what is in it | pygame? |
|---|---|---|
| `juicefx.py` | every effect, as maths | **no** |
| `audiofx.py` | synthesis, pitch ladders, buses | only inside `SoundBank` |
| `tiles.py` | slicing and tinting the sheet | yes |
| `rogue_juice.py` | the sim, the panel, the compositing | yes |

That split is the point rather than tidiness. An effect you cannot test without
opening a window is an effect you will never tune, so the maths runs, and is
tested, with no display anywhere; the audio's synthesis and pitch machinery are
plain numpy and are tested with no sound card. `SoundBank` keeps a log of what
it was asked to play, so "the swing fires on the wind-up and the impact on
contact" is an assertion rather than something you listen for.

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
ambient sway.

Decals are the only thing here that is *evidence* rather than an event. The
ambient sway is given to tufts and torches rather than to the floor tiles,
because the arena is baked into one surface and wobbling it would undo that.

**screen** — screen flash, vignette, RGB split, bloom, lighting, sprite
outlines, heat haze, pixel snapping.

Pygame has no shaders, so this group is the software cousins. Bloom is a
downscale, a threshold, an upscale and an add -- the downscale *is* the blur.
Lighting is a half-resolution light map, multiplied *and* added: a multiply can
only take brightness away, so on a floor this dark a torch would light nothing.
Heat haze is rows re-blitted at sine offsets, which is the same trick the slime
uses, pointed at the background.

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
p      pixel mode               m    music
esc    quit
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

## Cost

On this machine, at 800x600, with a fight going on:

| | median | p95 |
|---|---|---|
| defaults | 6.8ms | 14.1ms |
| every effect on | 10.9ms | 18.4ms |

Lighting is ~3.8ms and the RGB split is most of a frame on its own. Both are
toggles, and both are supposed to be.
