"""curses-based terminal renderer + input.

Translates raw key codes into abstract actions so the rest of the game never
imports curses.
"""
import curses

from .base import Renderer

# Raw key -> abstract action. Supports arrows, vi keys (hjkl), and wasd.
_KEY_TO_ACTION = {
    curses.KEY_UP: "move_up",
    curses.KEY_DOWN: "move_down",
    curses.KEY_LEFT: "move_left",
    curses.KEY_RIGHT: "move_right",
    ord("k"): "move_up",
    ord("j"): "move_down",
    ord("h"): "move_left",
    ord("l"): "move_right",
    ord("w"): "move_up",
    ord("s"): "move_down",
    ord("a"): "move_left",
    ord("d"): "move_right",
    curses.KEY_ENTER: "menu_select",
    10: "menu_select",  # Enter (LF)
    13: "menu_select",  # Enter (CR)
    27: "open_pause_menu",  # ESC
}


class TerminalRenderer(Renderer):
    def setup(self) -> None:
        curses.set_escdelay(25)
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.stdscr.keypad(True)

    def teardown(self) -> None:
        curses.curs_set(1)
        self.stdscr.keypad(False)
        curses.nocbreak()
        curses.echo()
        curses.endwin()

    def clear(self) -> None:
        self.stdscr.erase()

    def draw_glyph(self, x: int, y: int, glyph: str) -> None:
        # addstr raises at the bottom-right cell; swallow it.
        try:
            self.stdscr.addstr(y, x, glyph)
        except curses.error:
            pass

    def draw_text(self, x: int, y: int, text: str) -> None:
        try:
            self.stdscr.addstr(y, x, text)
        except curses.error:
            pass

    def present(self) -> None:
        self.stdscr.refresh()

    def poll_action(self) -> str | None:
        return _KEY_TO_ACTION.get(self.stdscr.getch())
