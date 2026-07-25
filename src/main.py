"""Start Tales of Derision, and run its turn loop."""

from datetime import datetime
from game import game_session, read_command_line


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
            if intent.world_action is not None:
                game.take_turn(intent.world_action)
            elif intent.redraw:
                game.redraw()


if __name__ == "__main__":
    main()
