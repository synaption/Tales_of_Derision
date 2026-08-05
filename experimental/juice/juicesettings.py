"""What the bench remembers between runs.

One JSON file, and one rule that shapes everything in here: **the file and the
build never have to agree**. The bench grows a slider or two most times anyone
opens it, and a settings file that has to be complete would break on every one
of them -- either by refusing to load, or by resetting a whole session's tuning
because one new key was missing.

So the contract is a merge in both directions:

* a key the file has and this build does not know is **kept, not dropped**. It
  is written back out untouched, so opening an old build and saving does not
  silently delete the newer build's settings;
* a key this build has and the file does not is **left at its code default**.
  Adding a slider tomorrow needs no migration and no version bump;
* a key that is present but wrong -- a string where a number goes, a number
  outside the slider's range, a choice whose options have since changed -- is
  ignored on its own, and everything around it still loads. One bad value must
  never cost the whole file.

Nothing in here imports pygame or moderngl. It is a dictionary, a file, and the
rules above, so it can be tested in full without a display -- which is also why
the display *settings* live here while the code that acts on them does not.

    settings = Settings()
    settings.load()
    settings.apply_juice(world.juice)
    ...
    settings.capture_juice(world.juice)
    settings.save()
"""

from __future__ import annotations

import json
import os
import tempfile

#: Bumped only if the *shape* of the file changes -- a section renamed, a value
#: given a different meaning. Adding keys is not a schema change; that is the
#: whole point of the merge rules above.
SCHEMA = 1

#: Beside the bench rather than in a config directory. This is a workbench: the
#: settings belong to the checkout you are tuning, and being able to delete
#: them by deleting an obvious file next to the script is worth more than being
#: well-behaved about `$XDG_CONFIG_HOME`.
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "juice_settings.json")

#: The sections a build understands. Anything else in the file is carried
#: through untouched.
SECTIONS = ("display", "toggles", "params", "choices")


class Settings:
    """A settings file, and the merge between it and a live `Juice`."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self.data: dict = {"schema": SCHEMA}
        #: Set by `load` and `save` to a short line worth putting in the log --
        #: including the failures, which are otherwise invisible.
        self.note = ""
        self.loaded = False

    # -- the file -----------------------------------------------------------
    def load(self) -> bool:
        """Read the file if there is one. Never raises, never leaves a mess.

        A missing file is the normal first run and is not an error. A corrupt
        one is reported and then ignored: the alternative is a bench that will
        not start because of a stray comma, and the worst case of ignoring it
        is that you set your options again.
        """
        if not os.path.exists(self.path):
            self.note = "no saved settings -- using the built-in defaults"
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("the file is not an object")
        except Exception as exc:                     # noqa: BLE001
            self.note = f"settings unreadable ({type(exc).__name__}) -- ignored"
            return False
        self.data = data
        self.data.setdefault("schema", SCHEMA)
        self.loaded = True
        self.note = f"loaded settings from {os.path.basename(self.path)}"
        return True

    def save(self) -> bool:
        """Write the file atomically, so a crash mid-write costs nothing.

        Temp file in the same directory and then `os.replace`, which is atomic
        on both platforms this ever runs on. Writing in place would give a
        half-written settings file the next launch has to cope with, and
        "coping with a half-written file" is a much worse problem than "the
        save did not happen".
        """
        self.data["schema"] = SCHEMA
        folder = os.path.dirname(self.path) or "."
        try:
            os.makedirs(folder, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".juice_settings.", dir=folder)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self.data, fh, indent=2, sort_keys=True)
                    fh.write("\n")
                os.replace(tmp, self.path)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
        except Exception as exc:                     # noqa: BLE001
            self.note = f"could not save settings: {exc}"
            return False
        self.note = f"saved {os.path.basename(self.path)}"
        self.loaded = True
        return True

    def section(self, name: str) -> dict:
        """One section of the file, created empty if it is not there yet."""
        got = self.data.get(name)
        if not isinstance(got, dict):
            got = {}
            self.data[name] = got
        return got

    # -- display ------------------------------------------------------------
    def get(self, section: str, key: str, fallback):
        """One value, with the fallback used for absent *and* wrong-typed.

        The type of `fallback` is the type check: a settings file that says the
        window is 900 pixels wide and `"yes"` tall should give you a 900-wide
        window, not a stack trace.
        """
        value = self.section(section).get(key, None)
        if value is None:
            return fallback
        if isinstance(fallback, bool):
            return value if isinstance(value, bool) else fallback
        if isinstance(fallback, int) and isinstance(value, (int, float)):
            return int(value)
        if isinstance(fallback, float) and isinstance(value, (int, float)):
            return float(value)
        if isinstance(fallback, str) and isinstance(value, str):
            return value
        return fallback

    def set(self, section: str, key: str, value) -> None:
        self.section(section)[key] = value

    # -- the juice registry ---------------------------------------------------
    def apply_juice(self, juice) -> int:
        """Push the saved values onto a live registry. Returns how many landed.

        Driven from the *file*, not from the registry, so a key the build has
        never heard of costs one dictionary lookup and is skipped. Every value
        is validated on its own terms -- a slider is clamped into its range, a
        choice has to still be one of the options -- because a settings file is
        an input like any other and half of it being sensible is no reason to
        throw the other half away.
        """
        applied = 0
        for key, value in self.section("toggles").items():
            toggle = juice.toggles.get(key)
            if toggle is not None and isinstance(value, bool):
                toggle.on = value
                applied += 1
        for key, value in self.section("params").items():
            if key not in juice.params or not isinstance(value, (int, float)):
                continue
            if isinstance(value, bool):              # bool is an int; not a slider
                continue
            p = juice.params[key]
            p.value = min(max(float(value), p.lo), p.hi)
            # The saved value becomes what "back to defaults" means, which is
            # the only reading of "save these as my defaults" that survives
            # someone then pressing F3.
            p.default = p.value
            applied += 1
        for key, value in self.section("choices").items():
            choice = juice.choices.get(key)
            if choice is None or not isinstance(value, str):
                continue
            if value in choice.options:
                choice.index = choice.options.index(value)
                applied += 1
        return applied

    def capture_juice(self, juice) -> None:
        """Snapshot a live registry into the file's sections.

        Updated in place rather than replaced, so sections and keys this build
        does not know about survive the round trip.
        """
        toggles = self.section("toggles")
        for key, toggle in juice.toggles.items():
            toggles[key] = bool(toggle.on)
        params = self.section("params")
        for key, p in juice.params.params.items():
            params[key] = round(float(p.value), 6)
            p.default = p.value
        choices = self.section("choices")
        for key, choice in juice.choices.items():
            choices[key] = choice.value

    def restore_defaults(self, juice) -> None:
        """What F3 means once a file exists: back to *your* defaults.

        The built-in defaults first, so a slider added since the file was
        written still has one, then the file on top of it.
        """
        juice.defaults()
        self.apply_juice(juice)
