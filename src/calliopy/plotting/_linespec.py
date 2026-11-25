"""MATLAB LineSpec strings (``'-k'``, ``'--or'``) split into matplotlib keyword arguments."""

from __future__ import annotations

__all__ = ["linespec_kwargs"]

_STYLES = ("--", "-.", "-", ":")
_COLORS = set("rgbcmykw")
_MARKERS = set("o+*.xsd^v<>ph")


def linespec_kwargs(spec: str | None) -> dict:
    """Translate a MATLAB LineSpec into ``linestyle``/``color``/``marker`` keyword arguments.

    A LineSpec with a marker but no line style draws markers only, as in MATLAB. ``None`` or
    ``''`` gives an empty dict. Unknown characters raise ``ValueError``.
    """
    if not spec:
        return {}
    s = str(spec)
    out: dict = {}
    i = 0
    while i < len(s):
        for st in _STYLES:
            if s.startswith(st, i):
                if "linestyle" in out:
                    raise ValueError(f"two line styles in LineSpec {spec!r}")
                out["linestyle"] = st
                i += len(st)
                break
        else:
            c = s[i]
            if c in _COLORS and "color" not in out:
                out["color"] = c
            elif c in _MARKERS and "marker" not in out:
                out["marker"] = c
            else:
                raise ValueError(f"cannot read LineSpec {spec!r} at {c!r}")
            i += 1
    if "marker" in out and "linestyle" not in out:
        out["linestyle"] = "none"
    return out
