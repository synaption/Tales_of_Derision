"""Start Tales of Derision, and run its turn loop."""

import time
from game import game_session, read_command_line

# Wall-clock floor for one turn the *player* spent, so the world can't run away
# from them when they hold a key down.
#
# This paces player actions and nothing else. Facing, key releases and menu work
# are free -- throttling those would make input feel several presses behind --
# and so is every turn the world takes on its own: the regions the background
# pump advances while the player thinks, the backlog an entered region replays,
# and the night a sleep passes. Those are not the player's turns to wait for, and
# a sleep is one action however many turns of world time it covers.
#
# World work done *during* a player's action still counts toward this budget
# rather than being added on top of it, because the floor is measured from the
# end of the previous turn: a turn that took 40 ms to simulate waits 60 ms, not
# 100. So a slow turn is never slower than a fast one, which is the whole point
# -- movement should look even.
SECONDS_PER_TURN = 0.1


def main() -> None:
    """Read the player's choices, then play the game one turn at a time.

    A turn is: wait for a command, work out what it means, and let the world
    move only if it cost the player a turn. Time in this game passes when the
    player spends it -- opening a menu, turning to face a tile or chopping a
    tree are free, so the world holds still for them.
    """
    choices = read_command_line()
    with game_session(choices) as game:
        if game is None:
            return  # startup was cancelled, or --screenshot captured its frame

        last_turn = time.monotonic()
        while True:

            # 1. Wait for a command. None means the player didn't act, and idle
            #    time is when we simulate the regions no one is watching.            
            action = game.next_action()
            if action is None:
                game.simulate_idle()
                continue

            # 2. Work out what the command means. Menus, dialogue and world
            #    interactions run here and report back what the world owes.
            intent = game.interpret(action)
            if intent.quit:
                break

            # 3. Move the world -- but only for an action that cost a turn.
            #    The frame is held back until step 4, so every turn appears on
            #    the beat instead of as soon as its simulation finished.
            if intent.world_action is not None:
                game.take_turn(intent.world_action, draw=False)
                # 4. Hold the turn to its minimum length -- once, for the one
                #    action the player just spent, however many turns of world
                #    time it covered. Measured from the end of the previous turn,
                #    so the world's own work counts toward the budget instead of
                #    being added on top of it.
                now = time.monotonic()
                short_by = SECONDS_PER_TURN - (now - last_turn)
                if short_by > 0:
                    time.sleep(short_by)
                    now = time.monotonic()
                last_turn = now
                game.redraw()  # this turn's frame, on the beat
            elif intent.redraw:
                game.redraw()

if __name__ == "__main__":
    main()
