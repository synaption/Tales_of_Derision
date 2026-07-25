"""The onymancer: a procedural name-magician.

Assembles people's names from syllables rather than drawing from fixed lists, so
the village never runs out of fresh names as it grows (and, later, breeds). Given
a seeded ``random.Random`` the output is fully deterministic, matching the game's
fixed-seed world layout -- the same seed always conjures the same villagers.

Names are built from an onset (opening consonant cluster), a vowel nucleus, and
an optional coda (closing consonant), repeated for one to three syllables. Gender
is flavour, not hard rule: female names lean toward soft vowel endings, male names
toward harder consonant endings. Passing ``gender=None`` merges both pools, which
is how a future *monogender* race would draw its names.
"""
from __future__ import annotations

import random

# Syllable building blocks. Kept deliberately small and pronounceable; edit these
# to reshape the whole naming aesthetic without touching the assembly logic.
_ONSETS = [
    "b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n",
    "r", "s", "sh", "st", "t", "th", "tr", "v", "w",
]
# A syllable may also start bare (vowel-initial), so onsets include "".
_ONSETS_WITH_BARE = _ONSETS + [""]

_VOWELS = ["a", "e", "i", "o", "u", "ae", "ia", "ei", "ou"]

# Codas that close a syllable. Soft codas suit vowel-flavoured (female-leaning)
# names; hard codas suit consonant-flavoured (male-leaning) names.
_SOFT_CODAS = ["", "", "n", "l", "r", "s", "th"]
_HARD_CODAS = ["n", "r", "s", "d", "k", "l", "th", "rn", "ld", "st"]

# Surname flavour: earthy, place-and-trade-like roots joined into one word.
_SURNAME_HEADS = [
    "Ash", "Black", "Brook", "Fair", "Green", "Grey", "Hollow", "Iron",
    "Marsh", "Oak", "Red", "Stone", "Thorn", "Vale", "White", "Wind",
]
_SURNAME_TAILS = [
    "ford", "brook", "wood", "field", "vale", "ridge", "moor", "well",
    "stone", "hill", "worth", "bourne", "combe", "mere",
]


def _coda(rng: random.Random, gender: str | None) -> str:
    """The closing consonant of the whole name, biased by gender: female names
    lean soft/vowel endings, male names lean harder consonant endings; an
    unspecified (monogender) gender draws from both."""
    if gender == "female":
        return rng.choice(_SOFT_CODAS)
    if gender == "male":
        return rng.choice(_HARD_CODAS)
    return rng.choice(_SOFT_CODAS + _HARD_CODAS)


def _open_syllable(rng: random.Random, *, first: bool) -> str:
    """An onset + vowel with no coda. Interior syllables stay open so they don't
    collide with the next onset into an unpronounceable consonant pile-up; only
    the first syllable may open bare on a vowel."""
    onset = rng.choice(_ONSETS_WITH_BARE if first else _ONSETS)
    return onset + rng.choice(_VOWELS)


class Onymancer:
    """Conjures names from a seeded RNG. One instance per world; call it repeatedly
    to name each new person. Stateless apart from the RNG it advances."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def given_name(self, gender: str | None = None) -> str:
        """A personal (first) name. ``gender`` is ``"male"``/``"female"`` for its
        ending flavour, or ``None`` to draw from the merged pool (monogender)."""
        syllables = self.rng.randint(1, 3)
        # Interior syllables are open (onset + vowel); the single gendered coda
        # closes the whole name, so the flavour lands where it's heard and the
        # name stays pronounceable.
        word = "".join(
            _open_syllable(self.rng, first=(i == 0)) for i in range(syllables)
        )
        return (word + _coda(self.rng, gender)).capitalize()

    def surname(self) -> str:
        """A family name, e.g. ``"Ashford"`` or ``"Thornvale"``."""
        return self.rng.choice(_SURNAME_HEADS) + self.rng.choice(_SURNAME_TAILS)

    def full_name(
        self, gender: str | None = None, surname: str | None = None
    ) -> tuple[str, str, str]:
        """Return ``(given, surname, full)``. Generates a surname when none is
        supplied (a lone person or the founder of a family)."""
        given = self.given_name(gender)
        family = surname if surname is not None else self.surname()
        return given, family, f"{given} {family}"


def make_onymancer(seed: int) -> Onymancer:
    """A deterministic onymancer seeded from ``seed``."""
    return Onymancer(random.Random(seed))
