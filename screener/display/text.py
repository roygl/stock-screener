"""Module-level copy, external links, captions, earnings strings, and hints.

The dashboard's non-tabular text: the always-on disclaimers, the column / score
``?`` tooltip copy, the tactical-readout help captions, the per-ticker external
link URLs, the earnings badge / summary strings, the universe-size ETA hint, the
scan-context and empty-state messages, and the hard-filter selectivity hint.
Pandas/numpy/stdlib only — never streamlit. Re-exported by :mod:`screener.display`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from urllib.parse import quote

from ..profiles import Profile
from ._base import _is_missing
from .features import feature_label


# --- module-level copy ---------------------------------------------------
# Short disclaimer rendered in EVERY app state (the sidebar): an always-on
# educational, not-advice notice (now including the buy-zone caveat) shown
# regardless of scan state. The fuller text lives in app.py's disclaimer expander.
DISCLAIMER_TEXT = (
    "This tool describes and ranks US large-cap equities from end-of-day data, "
    "and surfaces an educational 'buy zone' (entry band). It is not financial "
    "advice and makes no sell/exit calls."
)

DISCLAIMER_DETAIL = (
    "This screener is an educational research aid, not investment advice. It "
    "ranks and describes stocks from delayed, end-of-day data using mechanical "
    "rules — it does not know your goals, risk tolerance, or circumstances. It "
    "surfaces an educational 'buy zone' (a descriptive entry band) for context, "
    "but makes no sell, exit, or position-sizing recommendation and cannot tell "
    "you whether to act. Signals can be wrong or stale, and past behaviour does "
    "not predict future returns. Do your own research and consult a licensed "
    "professional before investing."
)

# Column / score "?" tooltip copy. Kept here (pure, unit-testable) so app.py can
# pass them straight to st.* ``help=`` arguments. PERCENTILE_HELP explains the
# lower-is-better inversion so a high bar on a cheap P/E reads correctly.
SCORE_HELP = (
    "Weighted blend of this profile's signal percentiles (0–1). Higher = a better "
    "match to the style. Not a price target or a buy signal."
)
PERCENTILE_HELP = (
    "Rank versus every stock in this scan (0 = bottom, 1 = top). For 'lower is "
    "better' signals like P/E the rank is flipped, so cheaper still scores high."
)
CONTRIBUTION_HELP = (
    "This signal's share of the Score = its normalized weight × its percentile. "
    "The rows sum to the Score."
)
WHAT_HELP = "What each signal measures, and which direction ranks better."
WHY_HELP = (
    "Plain-English read of where this name is strongest and weakest versus the "
    "scan (its top signals by percentile). Descriptive only — not advice."
)

# --- tactical-readout help copy (support/resistance, overextension, buy zone) -
# Three NON-advisory captions for the per-ticker detail panel. LEVELS_HELP and
# EXTENSION_HELP describe what the readout means; BUY_ZONE_HELP MUST carry an
# explicit not-advice disclaimer (the guardrail for the relaxed "explicit buy
# zone" decision — pinned by a test).
LEVELS_HELP = (
    "Support and resistance are price levels the stock has revisited repeatedly. "
    "Support sits at or below the last close (buyers have stepped in there before); "
    "resistance sits above it (sellers have). Strength (0–1) blends how many times "
    "the level was touched, how recently, and how tightly the touches cluster. "
    "Descriptive only — levels can break, and a touch is not a forecast."
)
EXTENSION_HELP = (
    "Overextension gauges how stretched the stock is above its trend — % above the "
    "20- and 50-day EMAs, RSI, the run of consecutive up-days, and volatility (ATR). "
    "'Normal' is unremarkable, 'Extended' is stretched, 'Parabolic' is a steep, "
    "potentially unsustainable run. Describes the current state — it is not a sell "
    "signal or a price target."
)
BUY_ZONE_HELP = (
    "Descriptive entry band from historical support — educational, not financial "
    "advice. The buy zone marks a price band the stock has found support in before "
    "(or a shallow pullback to a rising 20-day EMA); it is NOT a buy recommendation, "
    "a price target, or a guarantee, and it carries no exit/sell call. The level can "
    "break. Do your own research and consult a licensed professional before investing."
)


def universe_size_hint(n: int) -> str:
    """Plain-English cold/warm ETA bucket for scanning ``n`` names.

    A cold scan hits Yahoo ~3× per name (prices + fundamentals + earnings), so
    the cold estimate scales with ``n`` while a warm (cached) re-run is near
    instant. Buckets are deliberately coarse — they are a guardrail, not a
    promise.
    """
    n = int(n)
    if n <= 25:
        cold = "~1-2 min cold"
    elif n <= 50:
        cold = "~2-4 min cold"
    elif n <= 100:
        cold = "~4-8 min cold"
    else:
        cold = "many minutes cold"
    return f"~{n} names: a few seconds warm, {cold} (first run of the day)."


# --- per-ticker external links (pure, derived from `symbol`) -------------
def tradingview_url(symbol: str) -> str:
    """Interactive TradingView chart URL for ``symbol`` (``-`` -> ``.``)."""
    sym = str(symbol).strip().upper().replace("-", ".")
    return f"https://www.tradingview.com/chart/?symbol={quote(sym)}"


def yahoo_url(symbol: str) -> str:
    """Yahoo Finance quote URL for ``symbol`` (bare symbol; Yahoo keeps ``-``)."""
    return f"https://finance.yahoo.com/quote/{quote(str(symbol).strip().upper())}"


# --- earnings badge / summary -------------------------------------------
def earnings_badge(in_window, days_to_earnings) -> str:
    """Scalar badge string for one row, or ``""`` when not applicable.

    ``"⚠ Earnings in {d}d"`` only when ``in_window`` is truthy AND
    ``days_to_earnings`` is a finite number (guarded so it NEVER calls
    ``int(NaN)`` — ``days_to_earnings`` is float64 with ``NaN`` in the engine).
    Anything else -> ``""``.
    """
    if _is_missing(in_window) or not bool(in_window):
        return ""
    if _is_missing(days_to_earnings):
        return ""
    try:
        days = float(days_to_earnings)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(days):
        return ""
    return f"⚠ Earnings in {int(days)}d"


def earnings_badge_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized :func:`earnings_badge` over a frame (empty strings if absent).

    Returns an all-empty-string Series (indexed like ``df``) when the earnings
    columns are missing — e.g. a non-swing result or the wholly-empty frame.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype="object")
    if "earnings_in_window" not in df.columns or "days_to_earnings" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return pd.Series(
        [
            earnings_badge(w, d)
            for w, d in zip(df["earnings_in_window"], df["days_to_earnings"])
        ],
        index=df.index,
        dtype="object",
    )


def earnings_summary(df: pd.DataFrame) -> "str | None":
    """Banner text counting in-window names, or ``None`` when 0 / column absent.

    e.g. ``"⚠ 1 of 4 names report earnings within 7 days — elevated event
    risk."``. Returns ``None`` (no banner) when the column is missing or no row
    is flagged.
    """
    if df is None or len(df) == 0 or "earnings_in_window" not in df.columns:
        return None
    flagged = df["earnings_in_window"].fillna(False).astype(bool)
    k = int(flagged.sum())
    if k == 0:
        return None
    n = len(df)
    return f"⚠ {k} of {n} names report earnings within 7 days — elevated event risk."


# --- captions / context --------------------------------------------------
def scan_context_line(
    profile_label: str, n_names: int, cache_day: str, n_results: int
) -> str:
    """``"<label> · <n> names · as of <day> · <n_results> matches"``."""
    return f"{profile_label} · {int(n_names)} names · as of {cache_day} · {int(n_results)} matches"


# --- hard-filter selectivity hint ----------------------------------------
def _join_clauses(items: "list[str]") -> str:
    """Join clauses with commas and a trailing "and" (Oxford-style for 3+)."""
    items = [s for s in items if s]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _filter_phrase(filt) -> str:
    """Plain-English clause for one hard :class:`~screener.profiles.Filter`.

    Bespoke phrasing for the features the shipped profiles filter on (so a tuned
    threshold stays in sync — e.g. a changed ``rel_volume_20`` cutoff re-renders),
    with a generic ``"<label> <op> <threshold>"`` fallback so a brand-new filter
    can never raise. Phrases avoid inner parentheses so they nest cleanly inside
    :func:`selectivity_hint`'s own parenthetical.
    """
    feat = filt.feature
    if feat == "rel_volume_20" and filt.op in (">", ">="):
        return f"relative volume {filt.op} {filt.threshold:g}×"
    if feat == "in_leading_sector":
        return "in a top-3 sector by 3-mo return"
    if feat == "price_above_sma_150":
        return "price above its 150-day average"
    if feat == "price_above_sma_50":
        return "price above its 50-day average"
    if feat == "forward_pe" and filt.op == ">":
        return "a positive forward P/E"
    # Generic fallback: "<label> <op> <threshold>" (e.g. "RSI (14) > 50").
    label = feature_label(feat)
    if filt.op == "is_true":
        return label
    thr = filt.threshold
    thr_str = f"{thr:g}" if isinstance(thr, (int, float)) and not isinstance(thr, bool) else str(thr)
    return f"{label} {filt.op} {thr_str}"


def hard_filter_phrases(profile: Profile) -> "list[str]":
    """Plain-English clause for each of ``profile``'s hard filters, in order."""
    if profile is None or not getattr(profile, "filters", None):
        return []
    return [_filter_phrase(f) for f in profile.filters]


def selectivity_hint(profile: Profile, n_results: int, n_scanned: int) -> str:
    """Caption explaining how many scanned names cleared the hard filters.

    e.g. ``"35 of 500 scanned cleared the Swing hard filters (relative volume > 2×
    and in a top-3 sector by 3-mo return) — hard filters narrow the universe, so a
    small match count is the profile being selective, not a data error."``.

    This is the antidote to "it only found 35 of 500" confusion: it names the exact
    cutoffs doing the narrowing so a small result reads as intended selectivity, not
    a failed fetch. Returns ``""`` for a profile with no hard filters (nothing is
    screened out, so there is nothing to explain).
    """
    phrases = hard_filter_phrases(profile)
    if not phrases:
        return ""
    label = getattr(profile, "label", "") or getattr(profile, "name", "") or "this profile"
    return (
        f"{int(n_results)} of {int(n_scanned)} scanned cleared the {label} hard "
        f"filters ({_join_clauses(phrases)}) — hard filters narrow the universe, so "
        f"a small match count is the profile being selective, not a data error."
    )
