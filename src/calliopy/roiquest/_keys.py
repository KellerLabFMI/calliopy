"""Key handling of RoiQuest: MATLAB ``clbk.keyParser`` and the menu accelerators.

Port of calliope ``08_RoiQuest/+clbk/keyParser.m`` and the ``Accelerator`` keys of
``08_RoiQuest/+gfx/createMainfigure.m`` (Leonardo Lupori, 2024). A window forwards every key
press to :func:`key_press` and redraws what the result names; requests that need a dialog or
another window are returned, not executed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ._types import KeyResult, Redraw, RoiQuestRefused

if TYPE_CHECKING:  # pragma: no cover
    from ._session import RoiQuestSession

__all__ = ["KEY_BINDINGS", "key_press", "normalize_key"]

log = logging.getLogger("calliopy.roiquest")

KEY_BINDINGS: tuple[tuple[str, str, str], ...] = (
    ("rightbracket", "", "next_image"),
    ("leftbracket", "", "previous_image"),
    ("uparrow", "shift: x5", "threshold_or_move_up"),
    ("downarrow", "shift: x5", "threshold_or_move_down"),
    ("leftarrow", "shift: x10", "move_left"),
    ("rightarrow", "shift: x10", "move_right"),
    ("escape", "", "escape"),
    ("q", "", "escape"),
    ("a", "", "accept_putative"),
    ("g", "", "god_mode"),
    ("delete", "", "delete"),
    ("d", "", "delete"),
    ("space", "", "toggle_roi_visible"),
    ("1-9", "", "tag"),
    ("i", "", "previous_layer"),
    ("j", "", "next_layer"),
    ("o", "", "toggle_original"),
    ("c", "shift: local only, control: global only", "auto_roi_matching"),
    ("h", "", "histogram"),
    ("s", "control (menu File > Save to AData)", "save"),
    ("r", "control (menu File > Revert to last saved)", "revert"),
    ("t", "control (menu ROIs > Autodetect)", "cellpose_detect"),
    ("v", "control (menu ROIs > View activity)", "view_activity"),
    ("a", "control (menu ROIs > Select all)", "select_all"),
    ("n", "control (menu ROIs > Select none)", "select_none"),
    ("p", "control (menu ROIs > Expand selected)", "expand"),
    ("l", "control (menu ROIs > Shrink selected)", "shrink"),
)
"""``(key, modifiers, action)`` of RoiQuest, with MATLAB's key names (``event.Key``)."""

_ALIASES = {
    "up": "uparrow", "down": "downarrow", "left": "leftarrow", "right": "rightarrow",
    "]": "rightbracket", "[": "leftbracket", "esc": "escape", " ": "space",
    "del": "delete", "backspace": "backspace",
}

_ACCELERATORS = {"s": "save", "r": "revert", "t": "cellpose_detect", "v": "view_activity",
                 "a": "select_all", "n": "select_none", "p": "expand", "l": "shrink"}
_REQUESTS = {"save", "revert", "cellpose_detect", "view_activity"}
_MENU_ENABLE = {"cellpose_detect": "cellposeDetect", "expand": "roiExpand",
                "shrink": "roiShrink"}


def normalize_key(key: str, modifiers: Sequence[str] = ()) -> tuple[str, frozenset[str]]:
    """MATLAB key name and modifier set from a key event.

    Accepts MATLAB names (``'uparrow'``, ``'rightbracket'``), matplotlib/Tk-style names
    (``'up'``, ``']'``, ``'escape'``, ``' '``) and combined strings such as ``'ctrl+a'`` or
    ``'shift+up'``. Modifiers are lower-cased; ``'ctrl'`` becomes ``'control'``.
    """
    mods = {str(m).lower() for m in (modifiers or ())}
    k = str(key)
    if len(k) > 1 and "+" in k:
        parts = k.split("+")
        k = parts[-1] if parts[-1] else "+"
        mods.update(p.lower() for p in parts[:-1] if p)
    mods = {"control" if m in ("ctrl", "control") else m for m in mods}
    if k != " ":
        k = k.strip()
    if len(k) == 1 and k.isalpha() and k.isupper():  # matplotlib reports Shift + c as 'C'
        mods.add("shift")
    k = k.lower()
    return _ALIASES.get(k, k), frozenset(mods)


def key_press(s: RoiQuestSession, key: str, modifiers: Sequence[str] = ()) -> KeyResult:
    """Handle a key press, MATLAB ``clbk.keyParser`` plus the menu accelerators.

    Parameters
    ----------
    s
        The session.
    key
        Key name (see :func:`normalize_key`).
    modifiers
        Modifier names (``'shift'``, ``'control'``, ``'alt'``).

    Returns
    -------
    KeyResult
        The action taken and what to redraw. With ``executed=False`` the window must finish
        the request: ``'save'`` and ``'revert'`` (confirmation dialogs), ``'cellpose_detect'``
        (model choice), ``'view_activity'`` (ActiQuest window), ``'histogram'`` (histogram
        window) and ``'confirm_delete'`` (chronically tracked ROIs selected: MATLAB's two
        deletion dialogs, then :meth:`RoiQuestSession.delete_selected` with
        ``include_tracked``).

    Notes
    -----
    Control + key runs the menu item with that accelerator (MATLAB menus take these keys
    before ``keyParser``); a disabled menu item does nothing. Refusals
    (:class:`RoiQuestRefused`) are logged and reported as executed with nothing to redraw.
    """
    k, mods = normalize_key(key, modifiers)
    try:
        if "control" in mods and k in _ACCELERATORS:
            action = _ACCELERATORS[k]
            menu = _MENU_ENABLE.get(action)
            if menu is not None and not s.enabled.get(menu, True):
                return KeyResult(action, Redraw.NONE, True)
            if action in _REQUESTS:
                return KeyResult(action, Redraw.NONE, False)
            fn = {"select_all": s.select_all, "select_none": s.select_none,
                  "expand": s.expand_selected, "shrink": s.shrink_selected}[action]
            return KeyResult(action, fn(), True)
        return _parse(s, k, mods)
    except RoiQuestRefused as err:
        log.info("%s", err)
        return KeyResult(_name_of(k), Redraw.NONE, True)


def _name_of(k: str) -> str:
    for key, _, action in KEY_BINDINGS:
        if key == k:
            return action
    return "tag" if k in "123456789" and len(k) == 1 else ""


def _parse(s: RoiQuestSession, k: str, mods: frozenset[str]) -> KeyResult:
    full = Redraw.DISPLAY | Redraw.OVERLAY | Redraw.HISTOGRAM
    if k == "rightbracket":
        return KeyResult("next_image", s.cycle_image("+"), True)
    if k == "leftbracket":
        return KeyResult("previous_image", s.cycle_image("-"), True)
    if k in ("uparrow", "downarrow"):
        d = "up" if k == "uparrow" else "down"
        return KeyResult(f"threshold_or_move_{d}", s.threshold_step(d, mods), True)
    if k in ("leftarrow", "rightarrow"):
        d = "left" if k == "leftarrow" else "right"
        s.move_selected(d, mods)
        return KeyResult(f"move_{d}", Redraw.OVERLAY, True)
    if k in ("escape", "q"):
        return KeyResult("escape", s.escape(), True)
    if k == "a":
        s.accept_putative()
        s.reset_view()
        return KeyResult("accept_putative", Redraw.VIEW | Redraw.OVERLAY, True)
    if k == "g":
        return KeyResult("god_mode", s.god_mode_key(), True)
    if k in ("delete", "d"):
        kind = s.deletion_kind()
        if kind in ("mixed", "tracked"):
            return KeyResult("confirm_delete", Redraw.NONE, False)
        s.delete_selected()
        return KeyResult("delete", Redraw.OVERLAY, True)
    if k == "space":
        return KeyResult("toggle_roi_visible", s.toggle_roi_visible(), True)
    if len(k) == 1 and k in "123456789":
        return KeyResult("tag", s.tag_selected(int(k)), True)
    if k == "i":
        s.change_layer("-")
        return KeyResult("previous_layer", full, True)
    if k == "j":
        s.change_layer("+")
        return KeyResult("next_layer", full, True)
    if k == "o":
        return KeyResult("toggle_original", s.toggle_original(), True)
    if k == "c":
        s.auto_roi_matching(mods)
        return KeyResult("auto_roi_matching", Redraw.OVERLAY, True)
    if k == "h":
        return KeyResult("histogram", Redraw.NONE, False)
    return KeyResult("", Redraw.NONE, True)
