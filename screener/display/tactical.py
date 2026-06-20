"""Tactical-readout display: support/resistance frame, buy-zone band + caption.

Tidies a :class:`screener.levels.LevelSet` into a display frame and formats a
:class:`screener.levels.BuyZone` into a ``"$low – $high"`` band plus its
always-disclaimed caption (the guardrail for the relaxed buy-zone decision). All
fail-soft. Pandas/numpy/stdlib only — never streamlit. Re-exported by
:mod:`screener.display`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import _MISSING, _is_missing
from .formatting import format_signed_pct


_LEVELS_FRAME_COLUMNS = ["Level", "Kind", "Price", "Touches", "Strength", "Distance"]


def levels_to_frame(level_set) -> pd.DataFrame:
    """Tidy a :class:`screener.levels.LevelSet` into a display frame.

    Columns ``Level`` (humanised kind + ordinal, e.g. ``"Resistance 1"`` /
    ``"Support 1"``), ``Kind`` (``"Support"`` / ``"Resistance"``), ``Price`` (float),
    ``Touches`` (int), ``Strength`` (0..1 float — left numeric for a progress
    column), and ``Distance`` (signed-percent STRING via :func:`format_signed_pct`).

    Rows are resistances (nearest-above first) ABOVE supports (nearest-below first),
    matching how a chart stacks them around the last close. Fail-soft: ``None`` / an
    empty :class:`LevelSet` / anything without the expected ``supports`` /
    ``resistances`` tuples yields an empty frame with the right columns (never
    raises). ``Strength`` is coerced to a finite ``[0, 1]`` float so the progress
    column never breaks; a non-finite ``distance_pct`` renders ``"—"``.
    """
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in _LEVELS_FRAME_COLUMNS})
    if level_set is None:
        return empty

    resistances = list(getattr(level_set, "resistances", ()) or ())
    supports = list(getattr(level_set, "supports", ()) or ())
    if not resistances and not supports:
        return empty

    rows: "list[dict]" = []
    # Resistances first (top of the stack), then supports — each already ordered
    # nearest-first by levels.support_resistance.
    for kind_label, levels_seq in (("Resistance", resistances), ("Support", supports)):
        for i, lvl in enumerate(levels_seq, start=1):
            try:
                price = float(getattr(lvl, "price", float("nan")))
            except (TypeError, ValueError):
                price = float("nan")
            try:
                touches = int(getattr(lvl, "touches", 0))
            except (TypeError, ValueError):
                touches = 0
            try:
                strength = float(getattr(lvl, "strength", float("nan")))
            except (TypeError, ValueError):
                strength = float("nan")
            if not np.isfinite(strength):
                strength = 0.0
            strength = max(0.0, min(1.0, strength))
            rows.append(
                {
                    "Level": f"{kind_label} {i}",
                    "Kind": kind_label,
                    "Price": price,
                    "Touches": touches,
                    "Strength": strength,
                    "Distance": format_signed_pct(getattr(lvl, "distance_pct", None)),
                }
            )
    return pd.DataFrame(rows, columns=_LEVELS_FRAME_COLUMNS)


def format_buy_zone(zone) -> str:
    """A ``"$low – $high"`` band string for a :class:`screener.levels.BuyZone`.

    e.g. ``"$145.20 – $148.50"``. Returns ``"—"`` for a ``None`` zone, or when
    either edge is missing / non-finite (fail-soft — never raises). Uses an en-dash
    with surrounding spaces to read as a range.
    """
    if zone is None:
        return _MISSING
    low = getattr(zone, "low", None)
    high = getattr(zone, "high", None)
    if _is_missing(low) or _is_missing(high):
        return _MISSING
    try:
        lo = float(low)
        hi = float(high)
    except (TypeError, ValueError):
        return _MISSING
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return _MISSING
    return f"${lo:.2f} – ${hi:.2f}"


def buy_zone_caption(zone) -> str:
    """Caption for a buy zone: its basis plus the not-advice disclaimer.

    e.g. ``"Basis: nearest support · 3 touches. Educational entry zone, not
    financial advice."``. For a ``None`` zone returns a plain ``"No buy zone below
    the current price. Educational only, not financial advice."``. The disclaimer is
    ALWAYS present (the guardrail for the relaxed buy-zone decision), so the caption
    can never read as a recommendation.
    """
    disclaimer = "Educational entry zone, not financial advice."
    if zone is None:
        return f"No buy zone below the current price. {disclaimer}"
    basis = getattr(zone, "basis", None)
    basis_str = "" if _is_missing(basis) else str(basis).strip()
    if basis_str:
        return f"Basis: {basis_str}. {disclaimer}"
    return disclaimer
