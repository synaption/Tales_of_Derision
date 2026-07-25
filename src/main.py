"""Start Tales of Derision, and run its turn loop."""

from datetime import datetime
import time
from game import game_session, read_command_line

# Wall-clock floor for one *turn*, so the world can't run away from the player
# when they hold a key down. Only turns are paced: facing, key releases and menu
# work are free, and throttling those would make input feel like it's lagging
# several presses behind.
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

        last_turn = datetime.now()
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
                # 4. Hold the turn to its minimum length. Measured from the end
                #    of the previous turn, so the world's own work counts toward
                #    the budget instead of being added on top of it.
                now = datetime.now()
                short_by = SECONDS_PER_TURN - (now - last_turn).total_seconds()
                if short_by > 0:
                    time.sleep(short_by)
                    now = datetime.now()
                print(f"Time since last turn: {now - last_turn}")
                last_turn = now
                game.redraw()  # this turn's frame, on the beat
            elif intent.redraw:
                game.redraw()

if __name__ == "__main__":
    main()
