"""Which SDL video driver the game asks for on WSL.

WSLg offers both Wayland and X11. On Wayland SDL backs the window with EGL/OpenGL,
which this Mesa stack cannot create -- and it fails *silently*: `set_mode` returns a
good surface, frames are drawn and flipped, and nothing ever appears while the game
runs on invisibly. X11 gets a plain software surface and works. Since SDL's choice
depends on what the launching shell exported, the same build showed a window from one
terminal and not from another; these tests pin the choice down.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from renderer.pygame_renderer import PygameRenderer

pytestmark = pytest.mark.unrendered

WSL_PROC_VERSION = "Linux version 6.6.87.2-microsoft-standard-WSL2 (gcc ...)"
STOCK_PROC_VERSION = "Linux version 6.8.0-45-generic (buildd@lcy02) ..."


@pytest.fixture
def env(monkeypatch):
    """A clean environment with a fake /proc/version, so these never depend on the
    machine actually running the suite."""
    for name in ("SDL_VIDEODRIVER", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)

    def set_kernel(text: str) -> None:
        real_read_text = Path.read_text

        def fake_read_text(self, *args, **kwargs):
            if str(self) == "/proc/version":
                return text
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fake_read_text)

    monkeypatch.setattr("sys.platform", "linux")
    return set_kernel


def test_wsl_gets_pinned_to_x11(env, monkeypatch) -> None:
    env(WSL_PROC_VERSION)

    PygameRenderer._prefer_x11_on_wsl()

    import os
    assert os.environ["SDL_VIDEODRIVER"] == "x11"
    # WSLg always publishes :0; without it SDL has no X display to pick.
    assert os.environ["DISPLAY"] == ":0"


def test_an_explicit_driver_is_never_overridden(env, monkeypatch) -> None:
    """Screenshot capture and the headless test runner both set this; the game
    must not fight them."""
    env(WSL_PROC_VERSION)
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    PygameRenderer._prefer_x11_on_wsl()

    import os
    assert os.environ["SDL_VIDEODRIVER"] == "dummy"
    assert "DISPLAY" not in os.environ


def test_an_existing_display_is_kept(env, monkeypatch) -> None:
    env(WSL_PROC_VERSION)
    monkeypatch.setenv("DISPLAY", ":1")

    PygameRenderer._prefer_x11_on_wsl()

    import os
    assert os.environ["SDL_VIDEODRIVER"] == "x11"
    assert os.environ["DISPLAY"] == ":1"


def test_ordinary_linux_is_left_to_sdl(env) -> None:
    """A real desktop's Wayland session is fine -- this is a WSL workaround, not a
    blanket "never use Wayland"."""
    env(STOCK_PROC_VERSION)

    PygameRenderer._prefer_x11_on_wsl()

    import os
    assert "SDL_VIDEODRIVER" not in os.environ
    assert "DISPLAY" not in os.environ


def test_non_linux_is_left_alone(env, monkeypatch) -> None:
    env(WSL_PROC_VERSION)
    monkeypatch.setattr("sys.platform", "win32")

    PygameRenderer._prefer_x11_on_wsl()

    import os
    assert "SDL_VIDEODRIVER" not in os.environ


def test_an_unreadable_proc_version_is_not_fatal(env, monkeypatch) -> None:
    def boom(self, *args, **kwargs):
        raise OSError("no /proc here")

    monkeypatch.setattr(Path, "read_text", boom)

    PygameRenderer._prefer_x11_on_wsl()  # must not raise

    import os
    assert "SDL_VIDEODRIVER" not in os.environ
