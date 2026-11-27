"""A random remark for usage errors (calliope ``06_Library/wtf.m``)."""

from __future__ import annotations

import numpy as np

__all__ = ["WTF_ANSWERS", "wtf"]

#: The 17 remarks of ``wtf.m``, in MATLAB's order (MATLAB spelling kept).
WTF_ANSWERS: tuple[str, ...] = (
    "Really - you can do better",
    "That's it???",
    "You cannot be serious?",
    "Never mind",
    "A three-year-old could do this",
    "Moving swiftly along",
    "Let us pretend that never happened",
    "Is that why they call it a PhD?",
    "Whoever said you cant do simple tasks?",
    "Having troule choosing?",
    "Well this is a mess",
    "Ok - let's take a breath",
    "No need to panic",
    "Now you can panic",
    "Penguins are real?",
    "How many quarterbacks in a homerun?",
    "I can do statistics by eye",
)


def wtf(*, rng: np.random.Generator | None = None) -> str:
    """Return one of 17 remarks, for when you don't know what to say.

    Port of calliope ``06_Library/wtf.m`` (``out{ceil(17*rand)}``).

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random source. Default: a new ``numpy.random.default_rng()``.

    Returns
    -------
    str
        One element of :data:`WTF_ANSWERS`, each with probability 1/17.

    Differences from MATLAB
    -----------------------
    - Random numbers come from a ``numpy.random.Generator``, so sequences differ from
      MATLAB's ``rand``.
    """
    g = np.random.default_rng() if rng is None else rng
    return WTF_ANSWERS[int(g.integers(len(WTF_ANSWERS)))]
