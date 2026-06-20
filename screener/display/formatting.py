"""Type-aware, fail-soft value formatters (numbers, prices, percents, badges).

Pure string formatting for the dashboard's scalar cells: the type-aware
``format_value`` dispatcher, the price / market-cap / signed-percent formatters,
and the extension-state badge / colour helpers. Every one tolerates the engine's
fail-soft ``NaN``/``None`` and collapses to ``"—"`` so callers can render blindly.
Pandas/numpy/stdlib only — never streamlit. Re-exported by :mod:`screener.display`.
"""

from __future__ import annotations

import numpy as np

from ._base import (
    _BOOL_FEATURES,
    _DERIVED_01_FEATURES,
    _MISSING,
    _PERCENT2_FEATURES,
    _PERCENT_FEATURES,
    _PE_FEATURES,
    _SIGNED_PERCENT_FEATURES,
    _is_missing,
)


# --- value formatting ----------------------------------------------------
def format_value(feature: str, value) -> str:
    """Type-aware string for one feature value (fail-soft on ``NaN``/``None``).

    - percent-style fractions (momentum_*, revenue/earnings_growth,
      dist_52w_high) -> ``f"{v*100:.1f}%"``;
    - ``forward_pe`` / ``trailing_pe`` -> ``f"{v:.1f}"``;
    - ``rel_volume_20`` -> ``f"{v:.2f}×"``;
    - ``rsi_14`` -> ``f"{v:.0f}"``; ``macd_hist`` -> ``f"{v:.3f}"``;
    - derived [0, 1] scores (ema_5_9_cross_score, pullback_quality, rsi_health,
      sector_strength_score) -> ``f"{v:.2f}"``;
    - ``extension_score`` -> ``f"{v*100:.2f}%"`` (a 0..1 fraction as a percent);
    - ``dist_to_buy_zone_pct`` -> :func:`format_signed_pct` ("+3.2%"/"-1.0%");
    - ``extension_state`` -> the badge text via :func:`extension_badge`;
    - bool ``sma_stacked_20_50_150`` / ``in_buy_zone`` -> ``"Yes"`` / ``"No"``;
    - missing -> ``"—"``.
    """
    if feature in _BOOL_FEATURES:
        if _is_missing(value):
            return _MISSING
        return "Yes" if bool(value) else "No"

    # Categorical extension state -> the coloured badge text (never a float).
    if feature == "extension_state":
        if _is_missing(value):
            return _MISSING
        return extension_badge(value)

    if _is_missing(value):
        return _MISSING

    # Signed-percent fractions delegate to the shared signed formatter.
    if feature in _SIGNED_PERCENT_FEATURES:
        return format_signed_pct(value)

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if feature in _PERCENT_FEATURES:
        return f"{v * 100:.1f}%"
    if feature in _PERCENT2_FEATURES:
        return f"{v * 100:.2f}%"
    if feature in _PE_FEATURES:
        return f"{v:.1f}"
    if feature == "rel_volume_20":
        return f"{v:.2f}×"
    if feature == "rsi_14":
        return f"{v:.0f}"
    if feature == "macd_hist":
        return f"{v:.3f}"
    if feature in _DERIVED_01_FEATURES:
        return f"{v:.2f}"
    # Generic numeric fallback.
    return f"{v:.2f}"


# --- tactical-readout formatters (S/R, overextension, buy zone) ----------
# All pure + fail-soft: a NaN/None/degenerate input collapses to "—" (or an
# empty frame), exactly like the engine's fail-soft cells, so app.py can render
# them blindly. Mirrors the levels.Level / indicators.ExtensionState / levels.
# BuyZone contracts (signed distance_pct, 0..1 strength, categorical state).
def format_signed_pct(x) -> str:
    """A signed percent string for a fraction: ``0.032 -> "+3.2%"``, ``-0.01 ->
    "-1.0%"``, missing -> ``"—"``.

    Always carries an explicit ``+``/``-`` (a plain ``0.0`` reads ``"+0.0%"``) so a
    gap above vs below a band is unambiguous. The input is a FRACTION (``0.032`` ==
    ``3.2%``), matching ``Level.distance_pct`` / ``BuyZone.distance_pct``.
    """
    if _is_missing(x):
        return _MISSING
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _MISSING
    if not np.isfinite(v):
        return _MISSING
    return f"{v * 100:+.1f}%"


def format_price(x) -> str:
    """A ``$``-prefixed price, two decimals + thousands separators (fail-soft).

    ``195.0 -> "$195.00"``, ``1234.5 -> "$1,234.50"``; missing / non-finite -> ``"—"``.
    """
    if _is_missing(x):
        return _MISSING
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _MISSING
    if not np.isfinite(v):
        return _MISSING
    return f"${v:,.2f}"


# Human-readable market-cap units, largest first.
_CAP_UNITS = (("T", 1e12), ("B", 1e9), ("M", 1e6))


def format_market_cap(x) -> str:
    """Market cap in human units (fail-soft): ``1.23e12 -> "$1.2T"``,
    ``3.45e11 -> "$345.0B"``, ``1.2e7 -> "$12.0M"``.

    Below $1M falls back to a plain ``"$12,345"``. Missing / non-finite / ``<= 0``
    (a degenerate or unknown cap) -> ``"—"``.
    """
    if _is_missing(x):
        return _MISSING
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _MISSING
    if not np.isfinite(v) or v <= 0:
        return _MISSING
    for suffix, scale in _CAP_UNITS:
        if v >= scale:
            return f"${v / scale:.1f}{suffix}"
    return f"${v:,.0f}"


_EXTENSION_BADGES: "dict[str, str]" = {
    "normal": "🟢 Normal",
    "extended": "🟠 Extended",
    "parabolic": "🔴 Parabolic",
}
_EXTENSION_COLORS: "dict[str, str]" = {
    "normal": "gray",
    "extended": "orange",
    "parabolic": "red",
}


def extension_badge(state) -> str:
    """Emoji-prefixed badge text for an extension ``state`` string.

    ``"normal" -> "🟢 Normal"``, ``"extended" -> "🟠 Extended"``, ``"parabolic" ->
    "🔴 Parabolic"``. An unknown / missing state falls back to ``"🟢 Normal"`` (the
    safe baseline — ``extension_state`` is never ``None`` in the engine, defaulting
    to ``"normal"``), so this never raises.
    """
    key = "" if _is_missing(state) else str(state).strip().lower()
    return _EXTENSION_BADGES.get(key, _EXTENSION_BADGES["normal"])


def extension_state_color(state) -> str:
    """Semantic colour name for an extension ``state`` (for ``st.badge`` etc.).

    ``"normal" -> "gray"``, ``"extended" -> "orange"``, ``"parabolic" -> "red"``;
    an unknown / missing state falls back to ``"gray"`` (the neutral baseline).
    """
    key = "" if _is_missing(state) else str(state).strip().lower()
    return _EXTENSION_COLORS.get(key, _EXTENSION_COLORS["normal"])
