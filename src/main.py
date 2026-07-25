"""Start Tales of Derision."""

from game import play_game, read_command_line


def main() -> None:
    """Read the player's choices, then play the game."""
    choices = read_command_line()
    play_game(choices)


if __name__ == "__main__":
    main()
