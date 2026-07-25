"""Start Tales of Derision."""

from game import play_game, read_command_line


def main() -> None:
    """Read the player's choices and use them to play the game."""
    choices = read_command_line()
    play_game(
        save_file=choices.save_file,
        screenshot=choices.screenshot,
        rat_flood=choices.rat_flood,
        seed=choices.seed,
    )


if __name__ == "__main__":
    main()
