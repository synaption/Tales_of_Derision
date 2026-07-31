"""Tests for the roguelike workbench, with nothing on screen.

Two halves, matching the two files. `roguesim` is checked as plain data --
it never touches a graphics driver at all -- and `rogue` is checked through a
real headless renderer: an actual OpenGL context, an actual page, and frames
read back out of it. SDL is pointed at a driver that cannot display anything
and `set_mode` is never called, so anything that quietly wanted a window fails
here rather than on someone's build machine.

    python3 roguetest.py
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy
import pygame

import roguesim
from roguesim import (
    Attack,
    Blocker,
    Brain,
    Dungeon,
    Game,
    Glyph,
    Health,
    Name,
    Position,
    World,
    approach_map,
    field_of_view,
    generate_dungeon,
)

failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def section(title):
    print(f"\n--- {title} ---")


# =========================================================================
section("the registry")
# =========================================================================

world = World()
rat = world.spawn(Position(1, 2), Health(3, 3), Name("rat"))
rock = world.spawn(Position(4, 4))

check("a component comes back", world.get(rat, Position).cell == (1, 2))
check("a missing one is None", world.get(rock, Health) is None)
check("has() spans several types", world.has(rat, Position, Health, Name))
check("and fails on a missing one", not world.has(rock, Health))

matched = [entity for entity, _, _ in world.query(Position, Health)]
check("query intersects the stores", matched == [rat], f"{matched}")
check(
    "query over an absent type finds nothing",
    list(world.query(Position, Brain)) == [],
)
check("query with no types is empty", list(world.query()) == [])

# The corpse case: an entity is what it still holds, not what it was made as.
world.remove(rat, Health)
check("a removed component is gone", list(world.query(Position, Health)) == [])
check("but the entity survives", world.get(rat, Position) is not None)

# Killing while iterating is what a fight does, so it has to be safe.
survivors = World()
victims = [survivors.spawn(Position(i, 0), Health(1, 1)) for i in range(5)]
for entity, _, _ in survivors.query(Position, Health):
    survivors.kill(entity)
check(
    "killing during a query does not explode",
    list(survivors.query(Position)) == [],
    f"{len(victims)} entities removed mid-iteration",
)


# =========================================================================
section("the map")
# =========================================================================

import random

rng = random.Random(11)
dungeon = generate_dungeon(48, 30, rng)
check("rooms were placed", len(dungeon.rooms) >= 4, f"{len(dungeon.rooms)} rooms")
check(
    "the border is solid rock",
    not dungeon.tiles[0].any()
    and not dungeon.tiles[-1].any()
    and not dungeon.tiles[:, 0].any()
    and not dungeon.tiles[:, -1].any(),
)

# Every floor cell has to be reachable, or the player can be walled into a
# room with the exit on the wrong side of the rock.
floor = int((dungeon.tiles == roguesim.FLOOR).sum())
reached = approach_map(dungeon, [dungeon.rooms[0].center])
check(
    "every floor cell is reachable from the first room",
    int((reached >= 0).sum()) == floor,
    f"{int((reached >= 0).sum())} of {floor}",
)

# A tiny map still has to produce something playable.
cramped = generate_dungeon(9, 7, random.Random(2))
check("a cramped map still has a room", len(cramped.rooms) >= 1)
check("and somewhere to stand", int((cramped.tiles == roguesim.FLOOR).sum()) > 0)


# =========================================================================
section("the flood")
# =========================================================================

open_field = Dungeon(9, 9)
open_field.tiles[:] = roguesim.FLOOR
distance = approach_map(open_field, [(4, 4)])
check("the goal is zero steps away", distance[4, 4] == 0)
check("a diagonal neighbour is one step", distance[3, 3] == 1)
check(
    "and a corner is four, moving eight ways",
    distance[0, 0] == 4,
    f"{distance[0, 0]}",
)

walled = Dungeon(5, 5)
walled.tiles[:] = roguesim.FLOOR
walled.tiles[2, :] = roguesim.WALL
cut = approach_map(walled, [(0, 2)])
check("a wall leaves the far side unreachable", cut[4, 2] == -1, f"{cut[4, 2]}")
check("and the near side reachable", cut[1, 2] == 1)


# =========================================================================
section("sight")
# =========================================================================

room = Dungeon(21, 21)
room.tiles[1:20, 1:20] = roguesim.FLOOR
seen = field_of_view(room, (10, 10), 6)
check("you can see your own square", bool(seen[10, 10]))
check("and out to the radius", bool(seen[10, 4]))
check(
    "but not past it",
    not seen[10, 3],
    "radius 6 does not reach 7 cells away",
)
check(
    "the lit area is round, not square",
    not seen[10 + 5, 10 + 5],
    "the diagonal corner of the box is outside the circle",
)

# A pillar has to cast a shadow, and the shadow has to be behind it.
pillar = Dungeon(21, 21)
pillar.tiles[1:20, 1:20] = roguesim.FLOOR
pillar.tiles[10, 8] = roguesim.WALL
shadowed = field_of_view(pillar, (10, 10), 8)
check("the wall itself is seen", bool(shadowed[10, 8]))
check("what is behind it is not", not shadowed[10, 6])
check("what is beside it still is", bool(shadowed[12, 6]))

# Two walls meeting at a corner. Each shadows what is behind it, and the
# diagonal between them stays lit -- see the note in `field_of_view` about
# corners. Asserted so that a change in that behaviour is a decision rather
# than a surprise.
gap = Dungeon(11, 11)
gap.tiles[1:10, 1:10] = roguesim.FLOOR
gap.tiles[5, 4] = roguesim.WALL
gap.tiles[4, 5] = roguesim.WALL
diagonal = field_of_view(gap, (5, 5), 6)
check("each of the two walls shadows behind itself",
      not diagonal[5, 2] and not diagonal[2, 5])
check(
    "and the corner between them stays lit, as this algorithm does",
    bool(diagonal[4, 4]),
    "octant-based shadowcasting is permissive on a shared corner",
)

into = numpy.ones((21, 21), dtype=bool)
again = field_of_view(room, (10, 10), 6, into)
check(
    "reusing a buffer gives the same answer",
    again is into and numpy.array_equal(into, seen),
)


# =========================================================================
section("taking a turn")
# =========================================================================

game = Game((30, 18), seed=5)
start = game.player_position
check("the player starts in the first room", start == game.dungeon.rooms[0].center)
check("and can see where they are", bool(game.visible[start]))
check("the opening view is already explored", int(game.explored.sum()) > 0)
check(
    "explored is exactly what has been visible",
    int(game.explored.sum()) == int(game.visible.sum()),
    "nothing has moved yet",
)

dirty, fresh = game.take_dirty()
check("the opening view arrives as dirty cells", len(dirty) > 0, f"{len(dirty)}")
check("all of it is fresh ink", fresh <= dirty and len(fresh) == len(dirty))
check("and asking twice gives nothing", game.take_dirty() == (set(), set()))

# Walking into rock is not a turn.
walls = [
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if (dx or dy)
    and not game.dungeon.walkable(start[0] + dx, start[1] + dy)
]
if walls:
    before = game.turn
    moved = game.act(*walls[0])
    check(
        "walking into rock is not a turn",
        not moved and game.turn == before and game.player_position == start,
    )
else:
    check("walking into rock is not a turn", True, "no wall adjacent to test")

# Walking into floor is.
floors = [
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if (dx or dy) and game.dungeon.walkable(start[0] + dx, start[1] + dy)
]
before = game.turn
check("stepping onto floor is a turn", game.act(*floors[0]) and game.turn == before + 1)
check("the player actually moved", game.player_position != start)
moved_dirty, _ = game.take_dirty()
check(
    "both the cell left and the cell entered are dirty",
    start in moved_dirty and game.player_position in moved_dirty,
)

check("waiting is a turn too", game.act(0, 0))


# =========================================================================
section("fighting")
# =========================================================================

duel = Game((30, 18), seed=5, monsters=0)
here = duel.player_position
beside = None
for dx, dy in roguesim.NEIGHBOURS:
    if duel.dungeon.walkable(here[0] + dx, here[1] + dy):
        beside = (here[0] + dx, here[1] + dy)
        break

target = duel.world.spawn(
    Position(*beside),
    Glyph(ord("r"), layer=2),
    Blocker(),
    Health(2, 2),
    Attack(1),
    Brain(),
    Name("rat"),
)
check("a monster blocks its cell", duel.blocker_at(*beside) == target)

duel.act(beside[0] - here[0], beside[1] - here[1])
check("the player stayed put to attack", duel.player_position == here)
check("the monster is dead", not duel.world.get(target, Health).alive)
check("the corpse stopped acting", duel.world.get(target, Brain) is None)
check("and stopped blocking", duel.blocker_at(*beside) is None)
check(
    "and is drawn as a corpse",
    duel.world.get(target, Glyph).code == roguesim.CORPSE_GLYPH,
)
check("none are left", duel.monsters_left == 0)
check("something was said about it", any("dies" in m for m in duel.messages))

# The monsters have to actually be able to reach the player, or the whole
# hunt is decoration.
hunt = Game((30, 18), seed=5, monsters=0)
where = hunt.player_position
open_cells = [
    (x, y)
    for x, y in zip(*numpy.where(hunt.dungeon.tiles == roguesim.FLOOR))
    if 3 <= abs(int(x) - where[0]) + abs(int(y) - where[1]) <= 6
]
if open_cells:
    spot = (int(open_cells[0][0]), int(open_cells[0][1]))
    hunter = hunt.world.spawn(
        Position(*spot),
        Glyph(ord("k"), layer=2),
        Blocker(),
        Health(40, 40),
        Attack(1),
        Brain(sight=30),
        Name("kobold"),
    )
    opening = approach_map(hunt.dungeon, [where])[spot]
    for _ in range(12):
        hunt.act(0, 0)
    closed = approach_map(hunt.dungeon, [hunt.player_position])[
        hunt.world.get(hunter, Position).cell
    ]
    check(
        "a monster closes the distance",
        closed < opening,
        f"{opening} steps away, then {closed}",
    )
    check(
        "and gets its hits in",
        hunt.health.current < hunt.health.maximum,
        f"health {hunt.health.current}/{hunt.health.maximum}",
    )
else:
    check("a monster closes the distance", True, "nowhere to put one")


# =========================================================================
section("the page, headless")
# =========================================================================

check("no display was ever opened", pygame.display.get_surface() is None)

pygame.init()
pygame.font.init()

import rogue
from inkfx import InkSettings, View

SIZE = (640, 480)
settings = InkSettings()
page = rogue.RoguePage(SIZE, settings, seed=4, supersample=2, headless=True)

check(
    "a context exists without a window",
    page.renderer.headless and page.renderer.target is not page.renderer.context.screen,
    page.renderer.context.info["GL_RENDERER"],
)
check("the tileset loaded", page.tileset is not None,
      "" if page.tileset is None else page.tileset.path.name)
check(
    "the grid fits on the page",
    page.layout.origin[1] + page.layout.rows * page.layout.cell <= SIZE[1]
    and page.layout.columns * page.layout.cell <= SIZE[0],
    f"{page.layout.columns}x{page.layout.rows} cells of {page.layout.cell}px",
)
check(
    "the dungeon is the size of the grid",
    page.game.size == page.layout.grid_size,
)

if page.tileset is not None:
    glyph = page.tileset.glyph(ord("@"), 32)
    check("a glyph comes back at the size asked for", glyph.get_size() == (32, 32))
    check("it is ink: white with an alpha shape", (
        pygame.surfarray.array_alpha(glyph).max() > 200
        and pygame.surfarray.array_alpha(glyph).min() == 0
    ))
    check("and it is cached", page.tileset.glyph(ord("@"), 32) is glyph)
    faint = page.tileset.glyph(ord("@"), 32, rogue.INK_REMEMBERED)
    check(
        "a remembered glyph is the same shape, fainter",
        pygame.surfarray.array_alpha(faint).max()
        < pygame.surfarray.array_alpha(glyph).max(),
    )

view = View(SIZE)
page.draw((SIZE[0] // 2, SIZE[1] // 2), view)
frame = numpy.frombuffer(page.renderer.capture(), numpy.uint8).reshape(
    SIZE[1], SIZE[0], 3
)
check(
    "it is parchment, not an empty buffer",
    frame.mean() > 40 and frame.std() > 4,
    f"mean {frame.mean():.1f}, spread {frame.std():.1f}",
)
check(
    "and there is ink on it",
    pygame.surfarray.array_alpha(page.canvas.base).max() == 255,
)

# The invariant the incremental repaint rests on.
check(
    "the composite matches the base artwork",
    numpy.array_equal(
        pygame.surfarray.array_alpha(page.canvas.base),
        pygame.surfarray.array_alpha(page.canvas.surface),
    ),
)

# Unexplored parchment must be bare: fog of war is the whole look.
blank = 0
for x in range(page.layout.columns):
    for y in range(page.layout.rows):
        if page.game.explored[x, y]:
            continue
        rect = page.canvas.mask_rect(page.layout.cell_rect(x, y))
        if pygame.surfarray.pixels_alpha(page.canvas.base)[
            rect.left:rect.right, rect.top:rect.bottom
        ].any():
            blank += 1
check("unexplored cells are bare parchment", blank == 0, f"{blank} cells with ink")

# Fresh ink: the opening view has to be wet, and drying has to end.
check("the opening view went on wet", page.canvas.wet_bounds is not None)
for _ in range(400):
    if page.canvas.dry(0.1, 1.0) is None and page.canvas.wet_bounds is None:
        break
check("and dries off completely", page.canvas.wet_bounds is None)

# Play, then force a full repaint. If the incremental path is right, redrawing
# every cell from scratch cannot change a single texel -- which is the only
# way to be sure that painting a few dozen cells a turn is not slowly drifting
# away from what the game actually says is there.
walked = 0
for step in range(40):
    if not page.game.alive:
        break
    if page.game.act(*[(0, -1), (1, 0), (0, 1), (-1, 0), (1, 1)][step % 5]):
        page.catch_up()
        page.refresh_status()
        walked += 1

incremental = pygame.surfarray.array_alpha(page.canvas.base).copy()
page.game.dirty = {
    (x, y)
    for x in range(page.layout.columns)
    for y in range(page.layout.rows)
}
page.catch_up()
complete = pygame.surfarray.array_alpha(page.canvas.base)
differing = int((incremental != complete).sum())
check(
    "painting cell by cell equals painting the lot",
    differing == 0,
    f"{walked} turns, {differing} texels differ",
)
check(
    "and the composite still matches the base",
    numpy.array_equal(
        complete, pygame.surfarray.array_alpha(page.canvas.surface)
    ),
)

# A new dungeon has to leave nothing of the old one behind.
page.new_dungeon(page.seed + 1)
stale = 0
for x in range(page.layout.columns):
    for y in range(page.layout.rows):
        if page.game.explored[x, y]:
            continue
        rect = page.canvas.mask_rect(page.layout.cell_rect(x, y))
        if pygame.surfarray.pixels_alpha(page.canvas.base)[
            rect.left:rect.right, rect.top:rect.bottom
        ].any():
            stale += 1
check("a new dungeon starts on a clean sheet", stale == 0, f"{stale} cells left over")

# The debug views, and the help card, both work with nothing on screen.
looks = {}
for mode, name in enumerate(("final", "normals", "roughness", "ink", "wetness")):
    settings.debug_mode = mode
    page.draw((SIZE[0] // 2, SIZE[1] // 2), view)
    looks[name] = numpy.frombuffer(page.renderer.capture(), numpy.uint8).reshape(
        SIZE[1], SIZE[0], 3
    )
    check(f"  debug view {mode} ({name}) renders", looks[name].std() > 1.0,
          f"spread {looks[name].std():.1f}")
check(
    "  the ink view shows the dungeon",
    not numpy.array_equal(looks["ink"], looks["final"]),
)
settings.debug_mode = 0

page.help.toggle()
page.draw((SIZE[0] // 2, SIZE[1] // 2), view)
carded = numpy.frombuffer(page.renderer.capture(), numpy.uint8).reshape(
    SIZE[1], SIZE[0], 3
)
page.help.toggle()
page.draw((SIZE[0] // 2, SIZE[1] // 2), view)
bare = numpy.frombuffer(page.renderer.capture(), numpy.uint8).reshape(
    SIZE[1], SIZE[0], 3
)
check(
    "the help card draws over the page",
    float(numpy.abs(carded.astype(int) - bare.astype(int)).mean()) > 3.0,
)

page.release()


# =========================================================================
section("playing it with nobody watching")
# =========================================================================

here = os.path.dirname(os.path.abspath(__file__))
shot = rogue.play_headless(
    os.path.join(here, "rogue_headless.png"), size=(560, 400), turns=20, seed=7
)
check("a game plays and saves itself", os.path.getsize(shot) > 1000,
      f"{os.path.getsize(shot)} bytes at {shot}")

print()
print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
