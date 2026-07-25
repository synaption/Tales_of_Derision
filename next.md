reproduction and gender.  
names and family onymancer 

I want full simulation of all map tiles eventually.  I want simulation to happen in the background more than 1.5ms per turn.  I want to do as much as I can without it becoming disruptive.  It shouldn't be per turn.  It should especially do more simulation when not much is going on.  I don't want to prioritize the stailest tiles.  I want to prioritize the closes tiles.  when I enter a new map tile I want to bring that whole 120x60 region fully up to date.  when I sleep, I want to bring the whole world up to date.

the game should move everything into memory at start, i.e. sound files, tiles, state data.  memory > disk wherever aplicable.  

change name to Seeds of Derision.

make sure people don't get stuck.  

better pathfinding, better fov (not tcod)

Everything takes a certain amount of time.  

Lets give this a try.  There are a few things I think it's important to preserve.  There needs to be seed based determinism for time traveling purposes.  Only the players actions should change the course of events.  I want to be able to go back in time and change outcomes.  


ctrl-z like undo for time travel.  
- Treat each player action plus all resulting simulation as one undoable turn.
- Keep all gameplay data in one authoritative GameState; exclude particles, sound, and UI animation.
- Start by saving a full snapshot before each turn in a limited history buffer.
- Save and restore the random-number-generator state so undo cannot reroll outcomes.
- Let background threads calculate results, but only apply them at safe turn boundaries.
- Cancel or reject old async results after undo using a timeline/version number.
- Optimize later with change logs or periodic checkpoints only when snapshots become too expensive.

smooth scrolling/ zoom in zoom out?  

esper.process(action) needs to differentiate between actions happening in active map tiles, and actions happening on inactive map tiles.  simulation of inactive map tiles should only happen in simulate_idle(), _catch_up_entered_region_cooperatively, and during rest.  eliminate global entity scanns.  All entity scans should be regional, map tile by map tile, and inactive map tiles, and active map tiles should be stricktly partitioned.  

DONE, except where noted below.  src/spatial.py holds a regional entity index (every
positioned entity bucketed by region and kind, built once at worldgen, maintained
incrementally).  Every per-turn scan is regional now: movement collision, needs,
housing (residents/beds/sites/houses), build-site search, obstruction clearing,
blueprint ghosts, bed ownership, the NPC and fish AI goal buckets, and the AI's
blocker map.  Births are region-scoped too, with the rest of the world's delivered by
the idle pump and by sleep.  The partition is held by tests
(src/tests/test_active_region_partition.py); the index's correctness is held by
src/tests/test_spatial_index.py.  At 9 islands: 10.8 -> 8.2 ms/turn, and everything
except the AI is now flat in world size.

ALSO DONE (2026-07-25): needs and status effects are now steps on the region
scheduler, so hunger accrues wherever a region's turn runs -- a caught-up region is
charged one turn of hunger for every turn it owed, instead of waking with the
appetites it had when the player left.  The player still ticks against the world
clock, so a slow action still costs them proportionally more.  The camp lookup
(_camp_at, the only global Camp scan -- there is no camp-expiry pass; I mislabelled
it) is regional too.

STILL OPEN:
- Region debt is still unbounded: every region accrues a turn of debt per turn and the
  pump can only pay a trickle, so at 100 islands entering a stale region replays
  thousands of turns in the input path.  Needs a debt policy (fast-forward or clamp),
  not more scan removal.
- One shared world scheduler, instead of NpcAi's and FishAi's separate ones plus
  TreeGrowth's and Reproduction's own per-region day tracking, would make the
  partition structural rather than a rule each system has to follow.