reproduction and gender.  
names and family onymancer 

I want full simulation of all map tiles eventually.  I want simulation to happen in the background more than 1.5ms per turn.  I want to do as much as I can without it becoming disruptive.  It shouldn't be per turn.  It should especially do more simulation when not much is going on.  I don't want to prioritize the stailest tiles.  I want to prioritize the closes tiles.  when I enter a new map tile I want to bring that whole 120x60 region fully up to date.  when I sleep, I want to bring the whole world up to date.

the game should move everything into memory at start, i.e. sound files, tiles, state data.  memory > disk wherever aplicable.  

change name to Seeds of Derision.

make sure people don't get stuck.  

better pathfinding, better fov (not tcod)

Everything takes a certain amount of time.  

Use seeds to make things at least close to deterministic for time traveling purposes.  Players actions should change the course of events.  I want to be able to go back in time and change outcomes.  


ctrl-z like undo for time travel?  
- Treat each player action plus all resulting simulation as one undoable turn.
- Keep all gameplay data in one authoritative GameState; exclude particles, sound, and UI animation.
- Start by saving a full snapshot before each turn in a limited history buffer.
- Save and restore the random-number-generator state so undo cannot reroll outcomes.
- Let background threads calculate results, but only apply them at safe turn boundaries.
- Cancel or reject old async results after undo using a timeline/version number.
- Optimize later with change logs or periodic checkpoints only when snapshots become too expensive.

smooth scrolling/ zoom in zoom out?  

STILL OPEN:
- Region debt is still unbounded: every region accrues a turn of debt per turn and the
  pump can only pay a trickle, so at 100 islands entering a stale region replays
  thousands of turns in the input path.  Needs a debt policy (fast-forward or clamp),
  not more scan removal.
- One shared world scheduler, instead of NpcAi's and FishAi's separate ones plus
  TreeGrowth's and Reproduction's own per-region day tracking, would make the
  partition structural rather than a rule each system has to follow.

simulate_idleshould prioritize tiles based on how close they are to the player.  

A* needs to handle different types of ground have different walk speed bonuses?


I've given up on pure determinism.  What I want to keep deterministic is the family trees.  Pairs of parents might not always get together in different timelines, but if they do the children they have and the order they have them in will be the same.  

Here is how I want the simulation to work.  I want to come up with procedurally generated cities, that can be regenerated while in intermediate states, as needs change.  Then I want to calculate the cost in time/energy of the variouse steps of realizing the cities.  As needs change, and capabilities advance, I want to recalculate the procedural generation.  

Switch to the paradigm of generating personal records, resource and population simulations