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

smooth scrolling.  