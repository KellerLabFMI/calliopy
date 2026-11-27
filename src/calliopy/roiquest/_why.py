"""Random answers to "But... why?" (calliope ``08_RoiQuest/+utils/whyKeller.m``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["DEFAULT_WHY_SUBJECTS", "why_keller"]

#: Stand-ins used when no names are given. MATLAB hard-codes the first names of lab
#: members; calliopy is public, so they are not shipped (see :func:`why_keller`).
DEFAULT_WHY_SUBJECTS: tuple[str, ...] = (
    "the PI",
    "a postdoc",
    "a PhD student",
    "the lab manager",
    "the reviewer",
)

_SPECIAL = (
    "Because someone is still eating meat.",
    "It's trivial.",
    "Splendid...",
    "Because we dont have evidence that this is not the case.",
    "Go and figure it out.",
)
_ARTICLE = ("the", "some", "a")
_NOUN = (
    "mathemagical unicorn",
    "code whisperer",
    "cybernetic overlord",
    "quantum time traveler",
    "hamster emperor",
    "pixel sorcerer supreme",
    "crypto wizard",
    "data ninja",
    "space cowboy",
    "robotic samurai",
)
_NOMINATIVE = ("I", "you", "he", "she", "they")
_ACCUSATIVE = ("me", "all", "her", "him")
_ADVERB = ("very", "not very", "not excessively", "absolutely", "utterly", "completely")
_ADJECTIVE = (
    "towering",
    "smelly",
    "ageless",
    "genius-level",
    "bald",
    "furry",
    "splendiferous",
    "bionic",
    "loquacious",
    "supercalifragilisticexpialidocious",
)
_PREPOSITION = ("of", "from")
_PRESENT_VERB = ("fool", "please", "satisfy")
# MATLAB: randi(6) with case 5 and 6 both "obeyed" (otherwise branch).
_TRANSITIVE = ("threatened", "told", "asked", "helped", "obeyed", "obeyed")
_INTRANSITIVE = (
    "insisted on it",
    "suggested it",
    "told me to",
    "wanted it",
    "knew it was a good idea",
    "wanted it that way",
)
_NOUNED_VERB = ("love", "approval")


class _Grammar:
    def __init__(self, rng: np.random.Generator, names: Sequence[str]) -> None:
        self.rng = rng
        self.names = tuple(names)

    def randi(self, n: int) -> int:
        """MATLAB ``randi(n)``: 1..n."""
        return int(self.rng.integers(1, n + 1))

    def pick(self, words: Sequence[str]) -> str:
        return words[self.randi(len(words)) - 1]

    def lab_member(self) -> str:
        return self.pick(self.names)

    def special_case(self) -> str:
        # MATLAB ``switch 1``: always the first case.
        return _SPECIAL[0]

    def phrase(self) -> str:
        k = self.randi(3)
        if k == 1:
            return f"for the {self.pick(_NOUNED_VERB)} {self.prepositional_phrase()}."
        if k == 2:
            return f"to {self.pick(_PRESENT_VERB)} {self.object()}."
        return "because " + self.sentence()

    def sentence(self) -> str:
        return f"{self.subject()} {self.predicate()}."

    def subject(self) -> str:
        k = self.randi(4)
        if k in (1, 2):
            return self.lab_member()
        if k == 3:
            return self.pick(_NOMINATIVE)
        return f"{self.pick(_ARTICLE)} {self.noun_phrase()}"

    def object(self) -> str:
        k = self.randi(5)
        if k == 1:
            return self.pick(_ACCUSATIVE)
        if k == 2:
            return self.lab_member()
        return f"{self.pick(_ARTICLE)} {self.noun_phrase()}"

    def prepositional_phrase(self) -> str:
        k = self.randi(4)
        if k == 1:
            return f"{self.pick(_PREPOSITION)} {self.pick(_ARTICLE)} {self.noun_phrase()}"
        if k in (2, 3):
            return f"{self.pick(_PREPOSITION)} {self.lab_member()}"
        return f"{self.pick(_PREPOSITION)} {self.pick(_ACCUSATIVE)}"

    def noun_phrase(self) -> str:
        k = self.randi(4)
        if k == 1:
            return self.pick(_NOUN)
        if k == 2:
            return f"{self.adjective_phrase()} {self.noun_phrase()}"
        return f"{self.adjective_phrase()} {self.pick(_NOUN)}"

    def adjective_phrase(self) -> str:
        k = self.randi(6)
        if k in (1, 2, 3):
            return self.pick(_ADJECTIVE)
        if k in (4, 5):
            return f"{self.adjective_phrase()} and {self.adjective_phrase()}"
        return f"{self.pick(_ADVERB)} {self.pick(_ADJECTIVE)}"

    def predicate(self) -> str:
        if self.randi(3) == 1:
            return f"{self.pick(_TRANSITIVE)} {self.object()}"
        return self.pick(_INTRANSITIVE)


def why_keller(
    names: Sequence[str] | None = None, *, rng: np.random.Generator | None = None
) -> str:
    """Return a random answer to "But... why?".

    Port of calliope ``08_RoiQuest/+utils/whyKeller.m``: a small random grammar that
    builds a special answer (1 in 7), a phrase (3 in 7) or a sentence (3 in 7), with the
    first letter capitalised. The RoiQuest window shows it under *Help > But... why?*.

    Parameters
    ----------
    names : sequence of str, optional
        People the grammar may name as subject or object. Default:
        :data:`DEFAULT_WHY_SUBJECTS`. Pass your own list to get MATLAB's lab-member jokes.
    rng : numpy.random.Generator, optional
        Random source (MATLAB uses the global ``randi``). Default: a new
        ``numpy.random.default_rng()``.

    Returns
    -------
    str

    Differences from MATLAB
    -----------------------
    - MATLAB hard-codes the first names of lab members. They are not shipped in this
      public package: the names are an argument, with neutral stand-ins by default.
    - Random numbers come from a ``numpy.random.Generator``, so sequences differ from
      MATLAB's.
    - Kept as in MATLAB: the special answer is always "Because someone is still eating
      meat." (``switch 1`` selects the first of its five cases).
    """
    names = DEFAULT_WHY_SUBJECTS if names is None else tuple(names)
    if not names:
        raise ValueError("names must not be empty")
    g = _Grammar(np.random.default_rng() if rng is None else rng, names)
    k = g.randi(7)
    if k == 1:
        text = g.special_case()
    elif k in (2, 3, 4):
        text = g.phrase()
    else:
        text = g.sentence()
    return text[:1].upper() + text[1:]
