"""Feature labels/descriptions, profile blurbs, and the headline fit number.

The single source of truth for the human label and the plain-English definition
of every scored signal and surfaced raw feature, plus the one-line profile blurbs
and the ``fit_score`` headline metric. Pure (pandas/numpy/stdlib only — no
streamlit). Re-exported by :mod:`screener.display`.
"""

from __future__ import annotations

import numpy as np

from ._base import _is_missing


# Human labels for every feature that can surface in the table or the reasons
# panel. Covers all scored signals across the three profiles plus a few raw
# features the reason breakdown can show; .get(...) with a title-case fallback
# means an unknown key can never KeyError.
FEATURE_LABELS: "dict[str, str]" = {
    # momentum
    "momentum_1m": "1M Momentum",
    "momentum_3m": "3M Momentum",
    "momentum_6m": "6M Momentum",
    "momentum_12m": "12M Momentum",
    # trend / structure
    "sma_stacked_20_50_150": "Trend Stacked (20>50>150)",
    "dist_52w_high": "Distance From 52W High",
    # valuation / growth
    "forward_pe": "Forward P/E",
    "trailing_pe": "Trailing P/E",
    "revenue_growth": "Revenue Growth",
    "earnings_growth": "Earnings Growth",
    # swing derived / flow
    "ema_5_9_cross_score": "5/9 EMA Cross",
    "rel_volume_20": "Relative Volume (20d)",
    "macd_hist": "MACD Histogram",
    "rsi_14": "RSI (14)",
    "rsi_health": "RSI Health",
    "pullback_quality": "Pullback Quality",
    "sector_strength_score": "Sector Strength",
    # tactical readouts (overextension + buy zone)
    "extension_state": "Extension",
    "extension_score": "Extension Score",
    "in_buy_zone": "In Buy Zone",
    "dist_to_buy_zone_pct": "Distance To Buy Zone",
    # headline price + company (compact table + the Stage-3 detail card)
    "price": "Price",
    "change_pct": "Daily Change",
    "atr": "ATR (14)",
    "atr_pct": "ATR %",
    "market_cap": "Market Cap",
    "industry": "Industry",
}

# Plain-English, cell-sized definition for every feature in FEATURE_LABELS. Used
# BOTH by the inline "What it measures" column and the "How to read this" glossary
# (one source of truth — no per-profile duplication). Direction hints are baked in
# where polarity isn't obvious (e.g. forward_pe — lower is better). Parallel to
# FEATURE_LABELS: every label key has a description (a test asserts the parity).
FEATURE_DESCRIPTIONS: "dict[str, str]" = {
    # momentum
    "momentum_1m": "Price return over the last ~1 month.",
    "momentum_3m": "Price return over the last ~3 months.",
    "momentum_6m": "Price return over the last ~6 months.",
    "momentum_12m": "Price return over the last ~12 months.",
    # trend / structure
    "sma_stacked_20_50_150": "20-day > 50-day > 150-day average — a clean stacked uptrend.",
    "dist_52w_high": "How far below the 52-week high; nearer the high ranks better.",
    # valuation / growth
    "forward_pe": "Price ÷ next-year expected earnings; lower (cheaper) ranks better.",
    "trailing_pe": "Price ÷ last-year earnings; lower (cheaper) is a cheaper valuation.",
    "revenue_growth": "Year-over-year sales growth.",
    "earnings_growth": "Year-over-year earnings (profit) growth.",
    # swing derived / flow
    "ema_5_9_cross_score": "Freshness and strength of a bullish 5-over-9 EMA cross (0–1).",
    "rel_volume_20": "Today's volume vs its 20-day average; >1× is unusually active.",
    "macd_hist": "MACD line minus its signal line; positive = strengthening momentum.",
    "rsi_14": "14-day Relative Strength Index (0–100); ~70+ is overbought.",
    "rsi_health": "RSI rescaled to 0–1 with an overbought penalty — strong but not stretched.",
    "pullback_quality": "Health of a pullback to the 10/20-day EMAs (0–1).",
    "sector_strength_score": "The stock's sector ranked by 3-month return (0–1).",
    # tactical readouts (overextension + buy zone)
    "extension_state": "How stretched above trend: normal, extended, or parabolic.",
    "extension_score": "Overextension score (0–1); higher = more stretched above trend.",
    "in_buy_zone": "Whether the last close sits inside the descriptive buy-zone band.",
    "dist_to_buy_zone_pct": "Signed gap from the last close to the buy zone (0% if inside).",
    # headline price + company
    "price": "Most recent closing price.",
    "change_pct": "Percent change from the prior close (green up / red down).",
    "atr": "Average True Range (14-day): the typical daily price move, in dollars.",
    "atr_pct": "ATR as a percent of price — the typical daily move, normalized.",
    "market_cap": "Total market value of the company's shares.",
    "industry": "The company's specific industry (finer than its sector).",
}

# One-line, plain-English description per profile (keyed by ``Profile.name``).
# Feeds the profile radio's captions and the glossary intro. Descriptive only —
# never advice.
PROFILE_DESCRIPTIONS: "dict[str, str]" = {
    "long_term": (
        "Quality compounders in a durable uptrend: reasonable valuation, growing "
        "sales and earnings, a clean trend."
    ),
    "swing": (
        "Short-term setups: a fresh 5/9 EMA cross on heavy volume in a leading "
        "sector, with an earnings-date heads-up."
    ),
    "momentum": "Strongest trailing returns and trend, confirmed by volume.",
}


def feature_label(feature: str) -> str:
    """Human label for ``feature`` (explicit map, title-case fallback, no KeyError)."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def feature_description(feature: str) -> str:
    """Plain-English definition for ``feature`` (``""`` if unknown — never KeyError)."""
    return FEATURE_DESCRIPTIONS.get(feature, "")


def profile_description(name: str) -> str:
    """One-line description for a profile by ``Profile.name`` (``""`` if unknown)."""
    return PROFILE_DESCRIPTIONS.get(name, "")


def fit_score(score) -> int:
    """The 0..1 ``score`` as a 0..100 integer "fit" number (rounded, clamped).

    The headline metric the table and the detail panel lead with — it reads as a
    plain "fit out of 100" instead of a ``0.xxx`` fraction (the same composite the
    paid tools surface as their signature number). Fail-soft: a missing /
    non-finite score becomes ``0`` (never raises).
    """
    if _is_missing(score):
        return 0
    try:
        v = float(score)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(v):
        return 0
    return int(round(max(0.0, min(1.0, v)) * 100))
